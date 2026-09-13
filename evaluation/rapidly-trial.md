# Rapidly local SDK evaluation

## Status

The user approved evaluating a commercially licensed model. The Windows SDK
has been downloaded and tested locally in its documented no-key demo mode.
This establishes API integration and native throughput, not audio quality.
Demo output is watermarked; the runner deliberately exports no preview WAV in
demo mode. No licence verification or watermark mechanism is altered.

The [SDK changelog](https://github.com/rapidly-labs/rapidly-sdk/blob/a75155656556b38709fe77a706958dcbe1f13624/CHANGELOG.md)
lists WebAssembly as a future release, although
[the vendor website](https://www.rapidly.io/) demonstrates local browser WASM.
The public Windows DLL cannot run inside a Chrome extension. A vendor-supplied
browser build is a separate requirement before extension integration.

## Reproduce the native performance trial

The SDK and model files stay under ignored `models/`; they are not distributed
with this repository. The adapter uses the vendor's public C API through
Python ctypes. It verifies the DLL/model hashes before loading them.

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/rapidly-labs/rapidly-sdk.git models/rapidly-sdk
git -C models/rapidly-sdk sparse-checkout set bindings/python include models bin/windows-x64 examples/process-file
git -C models/rapidly-sdk checkout a75155656556b38709fe77a706958dcbe1f13624
python evaluation/rapidly_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/rapidly_demo_32ms --demo --model 32ms
python evaluation/rapidly_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/rapidly_demo_96ms --demo --model 96ms
```

Each report includes asset/input hashes, model settings, SDK processor metadata,
block timing, and an explicit watermarked/not-quality-evaluated status. The
trial requests stereo 48 kHz processing, sets the `Dialogue` output gain to 1
and `Noise` gain to -6 dB, supplies 480 samples (10 ms) at a time, and drains
the output queue and delayed tail as demonstrated by the vendor's file example.
Unknown output names, invalid output, or failure to drain cause an error.

## Measured native performance

Ryzen 9 9950X; 12-second stereo speech/music mixture; SDK 1.1.0. Model loading
is excluded from processing time. Both models reported one model channel
despite accepting stereo input; preservation of stereo content must be checked
with unwatermarked output rather than assumed from the output channel count.

| Model | SDK maximum latency | Processing time | 10 ms block p99 | Maximum block | Blocks over 10 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| speech-denoise-32ms.v1.1 | 32 ms | 0.443 s | 0.76 ms | 3.39 ms | 0 |
| speech-denoise-96ms.v1.1 | 96 ms | 0.405 s | 0.86 ms | 2.19 ms | 0 |

The first output became available after supplying 40 ms and 90 ms of input,
respectively, with this host block size. Queue availability is different from
per-sample playback delay and must not replace the SDK latency metadata or an
end-to-end measurement. There is no sustained Chrome playback result yet.

## Unwatermarked evaluation

The vendor advertises a free 30-day trial on [its website](https://www.rapidly.io/).
We need an **offline `lk_` trial key covering both speech-denoise-32ms.v1.1 and
speech-denoise-96ms.v1.1**. Save only the key in
`models/rapidly-license.txt`. The entire `models/` directory is already ignored
by Git. The runner reads the key from that file, registers it before creating
the model, and never includes it in its reports or command-line arguments.
It rejects subscription/activation keys because those use different online
licensing flows that this local evaluation does not implement.

```powershell
python evaluation/rapidly_trial.py audio/separation/dfn_synthetic/music.wav audio/separation/rapidly_32ms_music --license-file models/rapidly-license.txt --model 32ms
python evaluation/rapidly_trial.py audio/separation/dfn_synthetic/mixture.wav audio/separation/rapidly_32ms_mixture --license-file models/rapidly-license.txt --model 32ms
python evaluation/rapidly_trial.py audio/separation/dfn_synthetic/speech.wav audio/separation/rapidly_32ms_speech --license-file models/rapidly-license.txt --model 32ms
```

The public API confirms whether an offline key was accepted, but does not
expose per-model entitlement status. Acceptance alone is not proof that a
specific model is unwatermarked. Confirm model coverage with the issued key.
Before comparing speech contributions, verify output alignment with a unity
mix (`--reduction-db 0`), then run the same controlled overlap comparison used
for the other models. Repeat at 96 ms if the 32 ms model loses dialogue.

Promotion requires ordinary speech and overlapping speech to remain within
the initial 1 dB screening range, useful background reduction, preserved stereo,
and listening checks on real scenes. Browser integration additionally requires
the WASM build, acceptable browser block timings, and sustained lip-sync tests.

## Ready-to-send access request (not sent)

> We are evaluating an on-device Chrome extension that leaves dialogue at its
> original level while reducing loud music and effects in film/TV audio. We
> have integrated the public Windows SDK and benchmarked the 32 ms and 96 ms
> speech-denoise v1.1 models. Could you provide an offline trial key covering
> both models for an unwatermarked quality evaluation, and confirm whether your
> WebAssembly SDK can be supplied for Chrome-extension evaluation? We also need
> to preserve stereo output and would appreciate confirmation of how the SDK
> handles stereo input for these models. All programme audio should stay local.

The [published SDK contact](https://github.com/rapidly-labs/rapidly-sdk/blob/a75155656556b38709fe77a706958dcbe1f13624/LICENSE)
for trial keys and technical support is `support@rapidly.io`. No signup,
purchase, or vendor message has been submitted.
