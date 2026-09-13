# Streaming separator trial: GTCRN

## Decision

The tested GTCRN model meets the native CPU throughput target but fails
dialogue preservation during music. It remains an isolated evaluation tool.
The Chrome extension does not use it.

The proposed causal cinematic separator was not available as a runnable
checkpoint in the sources checked on 2026-09-13:

- The [FasTUSS paper](https://arxiv.org/html/2507.11435v1) includes a causal
  model evaluated on cinematic separation. However, the
  [official TUSS repository](https://github.com/merlresearch/unified-source-separation)
  publishes the original medium/large checkpoints, not a causal checkpoint.
  The fast FasTUSS configurations and the paper's causal configuration are
  distinct; neither paper terminology nor chunked inference makes the released
  original model a low-delay streaming model.
- [Band-SCNet](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf)
  reports 92 ms latency, but is evaluated on musical stems. The available
  [unofficial implementation](https://github.com/olivepicker/band_scnet_pytorch)
  is marked work in progress and does not provide a ready cinematic checkpoint.
- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) provides a small streaming ONNX
  model. It is trained for speech enhancement on DNS3, not cinematic separation.
  We tested it as a practical alternative, not as the missing causal TUSS model.

## Reproduction

The optional evaluation environment and upstream assets are isolated under
ignored `models/`. The upstream code/model is MIT licensed. The trial executes
only the pinned ONNX graph and verifies its SHA-256 before loading it.

```powershell
git clone https://github.com/Xiaobin-Rong/gtcrn.git models/gtcrn
git -C models/gtcrn checkout 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73
python -m venv models/streaming-venv
models/streaming-venv/Scripts/python.exe -m pip install -r evaluation/streaming-requirements.txt -e .
```

Use the controlled input files prepared in `dfn-browser-findings.md`:

```powershell
models/streaming-venv/Scripts/python.exe evaluation/gtcrn_trial.py audio/separation/dfn_synthetic/music.wav audio/separation/gtcrn_synthetic_music
models/streaming-venv/Scripts/python.exe evaluation/gtcrn_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/gtcrn_synthetic_mixture
models/streaming-venv/Scripts/python.exe evaluation/gtcrn_trial.py audio/separation/dfn_synthetic/speech.wav audio/separation/gtcrn_synthetic_speech
python evaluation/dfn_overlap_check.py analyze audio/separation/dfn_synthetic --delay-samples 0 --music-preview audio/separation/gtcrn_synthetic_music/reduced_preview.wav --mixture-preview audio/separation/gtcrn_synthetic_mixture/reduced_preview.wav
```

Each output directory contains a `report.json`, a `reduced_preview.wav`, and
the native 16 kHz speech estimate. Reports record model/input hashes,
dependencies, reduction setting, correctness checks, and runtime measurements.

## Processing path and correctness

The trial downsamples stereo input to 16 kHz and processes each channel with
independent recurrent state, one 256-sample hop per call. It uses a 512-sample
square-root Hann STFT and overlap-add reconstruction. It subtracts the speech
estimate from the downsampled input, upsamples that background estimate, and
reduces it by 6 dB against the original 48 kHz mix. Frequencies above the model's
bandwidth remain in the original mix; this is a limitation for high-frequency
effects. No global volume normalization or speech boost is applied.

The ONNX wrapper is checked against upstream's stored streaming output before
each trial: maximum sample error 0.000639, RMS error 0.0000736. A deterministic
test changes the second half of a signal and verifies identical earlier output
outside the full STFT window's influence (32 ms). Prefix error is zero. Using
only the half-window as this bound would be incorrect because overlap-add
includes the next contributing analysis window.

The file driver uses offline resampling and boundary padding. Its aligned
output is for quality comparison; neither the 16 ms hop nor the 32 ms window
is a measured Chrome end-to-end delay. Browser scheduling, streaming resampling,
and sustained playback have not been evaluated for this model.

## Measurements on this machine

Ryzen 9 9950X, native ONNX Runtime CPU, one thread, warmed session, separate
state for both stereo channels. Model loading and correctness checks are
excluded from processing time. Each input is 12 seconds.

| Input | Processing time | Stereo hop p99 | Maximum hop | Total preview level change |
| --- | ---: | ---: | ---: | ---: |
| Music | 0.793 s | 1.17 ms | 1.35 ms | -5.81 dB |
| Same music plus speech | 0.804 s | 1.50 ms | 1.85 ms | -5.61 dB |
| Speech alone | 0.793 s | 1.19 ms | 1.61 ms | -0.60 dB |

No measured stereo hop exceeded its 16 ms audio duration. This is a native
throughput result, not a browser performance result.

Subtracting the processed music-only output from the processed mixture gives
a counterfactual speech contribution **4.07 dB quieter** than the known speech
input, with correlation **0.9838**. This is not a directly isolated speech stem:
nonlinear changes to music can also affect the difference. Nevertheless, it
fails the initial screening target of staying within 1 dB while lowering
music by approximately 6 dB. Isolated quiet speech alone does not expose this
failure. A gate that bypasses quiet passages would still leave the active
music/speech overlap problem.

## Next practical routes

We need either a released/licensed model with better overlap preservation or
a compact model trained for this specific task. Training requires independent
dialogue, music, and effects data with held-out speakers/scenes; the current
12-second evaluation mixture must not become training data used to claim
generalization. The offline BandIt output remains a quality reference, not a
low-delay runtime.

[Rapidly](https://github.com/rapidly-labs/rapidly-sdk) is a candidate for a
licensed evaluation: its published SDK offers separate dialogue/noise outputs,
and [the vendor demonstrates local WASM processing](https://www.rapidly.io/).
Its free no-key mode audibly watermarks output, so an unwatermarked trial key
is needed for a meaningful quality comparison. Availability of a browser SDK
and preservation of cinematic dialogue still need verification; the vendor's
latency claims do not establish our quality target. No account, trial signup,
purchase, or vendor contact has been made.

The user subsequently approved evaluating a licensed local model. See
`rapidly-trial.md` for the completed native demo benchmarks, the missing trial
key and browser SDK requirements, and a prepared access request.
