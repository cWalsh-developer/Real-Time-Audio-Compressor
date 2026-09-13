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
too much programme audio. The next experiment should align its speech estimate
with a delayed original and reduce only a loud, estimated non-dialogue residual.
It must prove unchanged ordinary speech, natural overlap, no gain pumping,
stereo preservation, and stable continuous playback. If the speech residual
contains too much dialogue, a small causal model trained specifically for
cinematic dialogue/non-dialogue separation will be needed.
