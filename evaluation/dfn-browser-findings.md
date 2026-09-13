# Low-delay browser model probe

BandIt v2 produced useful offline stems but missed the live deadline. This
isolated probe asks a narrower question: can a small **low-look-ahead speech
model** run as a stereo Chrome AudioWorklet on 480-sample input frames?
It does not change or replace the extension's current audio path.

The probe uses [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet)
through the generated WASM bindings in
[`mezon-noise-suppression`](https://github.com/mezonai/mezon-noise-suppression),
pinned at `a5212661245a2184370fc3c3dd1f52dc4dffb2a5`. The package's own
worklet copies the first input channel to every output; our isolated worklet
creates two model handles so the original stereo channels stay distinct. This
is a **speech enhancer**, not a cinematic three-stem separator. Its raw output
must not become the default merely because it is fast.

## Reproduce locally

Clone the wrapper into the ignored `models/` folder and check out the pinned
commit:

```powershell
git clone https://github.com/mezonai/mezon-noise-suppression.git models/mezon-noise-suppression
git -C models/mezon-noise-suppression checkout a5212661245a2184370fc3c3dd1f52dc4dffb2a5
New-Item -ItemType Directory -Force models/dfn-assets | Out-Null
curl.exe -L --fail -o models/dfn-assets/df_bg.wasm https://cdn.mezon.ai/AI/models/datas/noise_suppression/deepfilternet3/v3/pkg/df_bg.wasm
curl.exe -L --fail -o models/dfn-assets/DeepFilterNet3_onnx.tar.gz https://cdn.mezon.ai/AI/models/datas/noise_suppression/deepfilternet3/v3/models/DeepFilterNet3_onnx.tar.gz
```

Verify SHA-256 before running. The evaluated WASM digest is
`440b5d12b6ea7d95008736f844221d7874ee15de5cb10d3015002470fdba0432`;
the model archive digest is
`c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616`.
The assets remain ignored by Git. The probe loads them from a temporary server
bound to `127.0.0.1`; audio is never sent to the CDN.

Install Playwright and its Chromium build if needed, then pass any 48 kHz WAV
inside this project:

```powershell
python -m pip install playwright
python -m playwright install chromium
python evaluation/dfn_browser_probe.py audio/separation/service_middle/original.wav audio/separation/tos_197_209.wav audio/separation/tos_208_220.wav
```

## Result on this machine

On an AMD Ryzen 9 9950X in isolated headless Chromium, with **two** 480-sample
model instances for stereo, 12 seconds of local audio rendered in about
1.0–1.3 seconds. The input/output cross-correlation peaked at about 39.3 ms
delay on each tested excerpt, consistent with the model's reported 40 ms
algorithmic latency. The [original DeepFilterNet paper](https://www.isca-archive.org/interspeech_2023/schroter23b_interspeech.pdf)
reports a real-time factor of 0.19 on a notebook CPU, but these figures are
from this project's browser probe and do **not** establish continuous Netflix
playback reliability or worst-case AudioWorklet scheduling.

Raw enhanced output still fails our quality target:

| Local 12 s excerpt | 12 dB attenuation limit | 50 dB attenuation limit |
| --- | ---: | ---: |
| Speech-dominant `service_middle` | -3.23 dB total RMS | -4.23 dB total RMS |
| Speech/music transition, `Tears of Steel` 197–209 s | -3.20 dB | -3.87 dB |
| Loud passage, `Tears of Steel` 208–220 s | -10.87 dB | -20.11 dB |

This is a promising **latency reference**, not a model selection. Feeding its
raw output to speakers would make ordinary dialogue audibly quieter and remove
too much programme audio.

## Residual-gated preview and controlled overlap

The next isolated experiment delayed the original by 1,888 samples (39.33 ms),
subtracted the model's speech estimate, and reduced that residual by up to 6 dB
only when its smoothed level crossed an event threshold. It uses attack,
release, hysteresis, and hold times to avoid a rapidly moving gain. The code is
in `dfn_probe_worklet.js`, accessible through `--preview-only`; the live
extension is unchanged. The thresholds and timings are experimental, not
production tuning.

On the same machine, 12-second stereo clips rendered in about 1.1-1.4 seconds.
On a speech-dominant *Tears of Steel* excerpt (24-36 s), total level changed by
less than 0.001 dB. On the quiet `service_middle` speech excerpt, it changed by
-0.004 dB. On the loud *Tears of Steel* passage (208-220 s), total level fell
5.57 dB. These isolated cases look encouraging, but total level cannot reveal
what happens to speech when music plays at the same time.

For a controlled overlap, `dfn_overlap_check.py` mixes the local speech excerpt
with a 12-second music clip from `cycles2015_original.wav`. The speech and music
inputs are -24.89 and -14.51 dBFS after scaling. Process the music alone and
the same music plus speech through separate fresh model instances. The
difference between their processed outputs is a **counterfactual speech
contribution**, not an isolated output stem; model and event-gate nonlinearity
can affect it. Here it is 3.24 dB lower than the known delayed speech input,
with 0.9574 correlation. This fails the goal that dialogue stay perceptually
unchanged during loud music.

As a quality reference, the existing BandIt v2 cinematic separator was run on
the same two inputs with a fixed 6 dB reduction of its music/effects estimates.
It is closer to the target, but much too slow for streaming:

| Isolated preview | Music-only level change | Speech contribution change | Correlation | Inference for each 12 s input |
| --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 residual gate, Chromium CPU | -5.73 dB | -3.24 dB | 0.9574 | 0.98-1.05 s offline render |
| BandIt v2 cinematic stems, RTX 5090 | -6.00 dB | -0.71 dB | 0.9812 | 17.9 s |

Neither figure measures uninterrupted browser playback. BandIt's multi-second
internal chunks make its end-to-end delay unacceptable even if inference were
faster. It establishes a useful quality reference for the next causal model
rather than an extension implementation.

Reproduce with these local clips (generated WAVs remain ignored):

```powershell
python evaluation/dfn_overlap_check.py prepare audio/separation/tos_speech_24_36.wav audio/eval/cycles2015_original.wav audio/separation/dfn_synthetic
python evaluation/dfn_browser_probe.py --preview-only --output-dir audio/separation/dfn_synthetic audio/separation/dfn_synthetic/music.wav audio/separation/dfn_synthetic/mixture.wav
python evaluation/dfn_overlap_check.py analyze audio/separation/dfn_synthetic
```

To reproduce the offline quality reference after installing the optional
BandIt environment described in `separation-trial.md`:

```powershell
models/separation-venv/Scripts/python.exe evaluation/separation_trial.py audio/separation/dfn_synthetic/music.wav audio/separation/bandit_synthetic_music --duration 12 --reduction-db 6 --device cuda
models/separation-venv/Scripts/python.exe evaluation/separation_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/bandit_synthetic_mixture --duration 12 --reduction-db 6 --device cuda
python evaluation/dfn_overlap_check.py analyze audio/separation/dfn_synthetic --delay-samples 0 --music-preview audio/separation/bandit_synthetic_music/reduced_preview.wav --mixture-preview audio/separation/bandit_synthetic_mixture/reduced_preview.wav
```

The direct and residual-gated variants therefore remain evaluation tools, not
extension defaults. The measured browser speed shows a small model can meet a
compute budget, but DeepFilterNet3 was built for speech enhancement rather
than film dialogue/effects separation. Next, seek or train a causal separator
for these sources, repeat the controlled overlap test, and then test continuous
AudioWorklet scheduling and lip sync in the extension. Offline rendering speed
alone does not prove stable live playback.
