# DTLN streaming ONNX trial

## Decision

Tested on 2026-09-14. DTLN passes the native speed screen and keeps isolated
speech close to its original level, but the tested remix fails dialogue
preservation during loud music. It remains an isolated evaluation tool.
It is not integrated into the Chrome extension.

This implements the next concrete test of the browser-only direction: use a
released, small, stateful ONNX model before spending time on browser integration.
DTLN is a **speech enhancer trained on noisy speech**, not a model trained to
separate cinematic dialogue, music, and effects. Passing this screen would
only justify a browser trial, not establish production suitability.

## Candidate and reproducibility

- [Official DTLN repository](https://github.com/breizhn/DTLN), pinned at
  `1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc`.
- Released `pretrained_model/model_1.onnx` and `model_2.onnx`, together about
  4 MB; external recurrent state; no conversion or vendor credentials needed.
- 16 kHz input, 512-sample analysis blocks, 128-sample hops (8 ms).
- [Upstream licence](https://github.com/breizhn/DTLN/blob/1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc/LICENSE): MIT.
- Downloads and generated audio remain in ignored `models/` and `audio/`.
  The runner verifies SHA-256 before loading the model. Its `--download` flag
  retrieves the pinned weights, licence, and reference example from GitHub;
  programme audio is never uploaded.

Use the optional evaluation environment from
[the GTCRN trial](causal-separator-findings.md), or create it from the root:

```powershell
python -m venv models/streaming-venv
models/streaming-venv/Scripts/python.exe -m pip install -r evaluation/streaming-requirements.txt -e .
```

The controlled inputs are prepared as described in
[the browser model findings](dfn-browser-findings.md). From the root:

```powershell
models/streaming-venv/Scripts/python.exe evaluation/dtln_trial.py audio/separation/dfn_synthetic/music.wav audio/separation/dtln_music --download
models/streaming-venv/Scripts/python.exe evaluation/dtln_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/dtln_mixture
models/streaming-venv/Scripts/python.exe evaluation/dtln_trial.py audio/separation/dfn_synthetic/speech.wav audio/separation/dtln_speech
models/streaming-venv/Scripts/python.exe evaluation/dfn_overlap_check.py analyze audio/separation/dfn_synthetic --delay-samples 0 --music-preview audio/separation/dtln_music/reduced_preview.wav --mixture-preview audio/separation/dtln_mixture/reduced_preview.wav
```

Each output directory contains `reduced_preview.wav`, `speech_estimate_16k.wav`,
and `report.json`. Reports include model and input hashes, dependency versions,
per-hop timing, level changes, and streaming correctness checks.

## Processing and correctness

The adapter follows the author's two-stage ONNX example: magnitude masking,
reconstruction with the input phase, learned time-domain processing, and
overlap-add. Each stereo channel has independent state. The file driver adds
zeros at the end to drain its delayed output, including partial final hops.

For comparison it removes the upstream file driver's **384-sample position
offset (24 ms at 16 kHz)**. That is not the live latency: each next 128-sample
hop must also arrive, giving the author's documented 512-sample/32 ms delay.
Browser scheduling, resampling, and playback buffering would add further delay.

After alignment, the estimated background is `lowband input - speech estimate`.
It is resampled back to 48 kHz, reduced by a fixed 6 dB, and subtracted from the
original mix. There is no speech boost, loudness normalisation, or event gating.
At zero reduction the remix is exactly the original input. Frequencies outside
the 16 kHz model's band remain in the original mix, so this is not a full-band
solution for gunfire or other high-frequency effects.

Validation performed:

- The adapter exactly matched the pinned upstream example over all 191,616
  samples processed by that example on the left channel of the 12-second
  speech input: maximum and RMS difference both zero before WAV encoding.
  The reference's file I/O was replaced in memory; its processing loop was
  executed unchanged. The original example leaves its final tail unwritten;
  the new adapter deliberately drains that tail.
- An identity analysis/synthesis pair reconstructs both channels, the first
  sample, and partial final hops; the raw output has the expected position offset.
- Recurrent state is independent between channels. Changing future input
  leaves all earlier completed raw output hops unchanged, with both the
  identity pair and the actual models.
- All 38 Python tests passed, including nine new model-free adapter tests.
  Tests do not download weights or require ONNX Runtime.

## Measurements

Ryzen 9 9950X, native ONNX Runtime CPU, one thread, two independent channel
states. Model loading and warm-up/correctness checks are excluded. The process
time includes offline external resampling, inference, tail draining, and remix.
These are file-driven throughput measurements, not sustained browser playback.

| Input (12 s stereo) | Process time | Stereo hop p99 | Maximum hop | Hops over 8 ms | Remix level change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Music | 0.426 s | 0.35 ms | 0.68 ms | 0 | -5.53 dB |
| Same music plus speech | 0.442 s | 0.74 ms | 0.86 ms | 0 | -5.30 dB |
| Speech alone | 0.437 s | 0.39 ms | 0.88 ms | 0 | -0.21 dB |

The controlled overlap comparison reports:

- Speech contribution change: **-2.38 dB**.
- Speech contribution correlation: **0.9482**.
- Relative difference from reference speech: **-8.66 dB**.
- Music-only reduction: **5.53 dB**.

The speech contribution is the processed mixture minus processed music alone,
not a directly isolated dialogue stem. Nonlinear changes in the music can
affect this measurement. It still fails our initial screen of remaining within
1 dB of the original speech while producing useful music reduction. Good
isolated-speech behaviour does not establish preservation during overlap.

## Implication for the local-browser route

This is another example of a downloadable, fast model that does not meet the
required quality screen in this configuration. It does not show that all
open models fail, and it does not justify substituting an untested model in
the live extension. An ONNX/WebGPU port would test deployment and speed; it
would not fix these separation errors.

The next model should demonstrate better overlap preservation on this screen
and additional held-out scenes before browser integration. If released models
continue to miss that target, training or fine-tuning a compact streaming model
on separate dialogue/music/effects material becomes a distinct research stage.
The existing evaluation clips must remain held out from that training.

## Additional research comparison: streaming Denoiser DNS64

The same three inputs were also screened with Meta's
[speech-specific streaming Denoiser](https://github.com/facebookresearch/denoiser),
which uses a Demucs architecture but is distinct from the music-stem models.
Its source is under CC-BY-NC 4.0, so this comparison does not establish a
commercially distributable backend. It also fails the overlap screen.

The reproducible research-only runner is `evaluation/denoiser_screen.py`.
It uses the upstream `DemucsStreamer` with independent channel state, `dry=0`,
`num_frames=1`, 256-sample input hops, and the default resampler settings.
The same 6 dB lowband-residual remix and offline external resampling are used
as above. There is no added speech boost or output normalisation; the upstream
model's own running input normalisation and output rescaling are retained.
EOF is drained with zero input rather than upstream `flush()`, which clears
the recurrent state. Dry-path identity and output invariance to input chunk
sizes (256 versus 113 samples) are checked before the screen runs.

Reproduction uses the existing optional PyTorch separation environment:

```powershell
git clone https://github.com/facebookresearch/denoiser.git models/denoiser
git -C models/denoiser checkout 8afd7c166699bb3c8b2d95b6dd706f71e1075df0
models/separation-venv/Scripts/python.exe -m pip install scipy==1.16.3
curl.exe -L --fail -o models/denoiser/dns64-a7761ff99a7d5bb6.th https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th
models/separation-venv/Scripts/python.exe evaluation/denoiser_screen.py
models/streaming-venv/Scripts/python.exe evaluation/dfn_overlap_check.py analyze audio/separation/dfn_synthetic --delay-samples 0 --music-preview audio/separation/dns64_music/reduced_preview.wav --mixture-preview audio/separation/dns64_mixture/reduced_preview.wav
```

The runner requires the pinned, unmodified source checkout and verifies weights
SHA-256 `a7761ff99a7d5bb69c16a61b3089469805cf3afd82e128103f0f77e79dfead5f`
before loading with `torch.load(weights_only=True)`. The weight file is about
134 MB. No upstream source or model weights are bundled with this repository.

On the same CPU with one PyTorch thread (`torch 2.11.0+cu128`, CPU inference):

| Input (12 s stereo) | Process time | Stereo hop p99 | Hops over 16 ms | Remix level change |
| --- | ---: | ---: | ---: | ---: |
| Music | 26.04 s | 36.77 ms | 748 | -5.43 dB |
| Same music plus speech | 26.08 s | 36.86 ms | 748 | -5.43 dB |
| Speech alone | 25.81 s | 36.24 ms | 748 | -0.20 dB |

The controlled speech contribution fell **2.63 dB**, with correlation **0.9086**.
The streamer requires 41.31 ms of input context; the first output was available
after feeding 48 ms with the chosen hop. Neither number is measured Chrome
playback latency. This particular CPU path is slower than real time; no ONNX
export or GPU speed claim is made. Faster execution would not address the
observed overlap-quality failure, so it was not pursued for browser integration.
