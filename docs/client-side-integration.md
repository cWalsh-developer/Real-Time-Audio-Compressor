# Client-side model integration

## Decision

The extension will target local inference first. Audio remains in the browser and no server or commercial SDK is required. ONNX Runtime Web with WASM is the initial portability target; WebGPU can be evaluated later as an optional execution provider.

Commercial SDKs such as AudioShake remain an evaluation option for a separate licensed product path. They are not part of the open extension's default runtime, and the current extension must continue to work without credentials or network access.

## Current artifact

`adaptive_audio.export_model` exports the trained spectral-mask baseline to ONNX and writes a SHA-256 sidecar manifest. The exported graph has input and output shape `[1, 257, 249]`, representing a four-second, 16 kHz spectrogram window.

The export was checked with ONNX Runtime CPU and matched PyTorch within `1.8e-7` maximum absolute error on a representative input.

## Why it is not in the live worklet yet

The current extension processes AudioWorklet blocks continuously. Feeding this model four seconds of audio would add a multi-second decision window and violate the lip-sync requirement. The model also has no streaming state and expects magnitude spectrograms, so the browser adapter would need STFT framing, overlap handling, inverse STFT, and a fallback path in addition to ONNX Runtime Web.

The next model phase is therefore a causal or short-frame model with explicit state and a bounded look-ahead. It must pass:

- sustained real-time factor above `1.0` on representative hardware;
- bounded input-to-output delay suitable for video playback;
- stereo or channel-safe processing;
- bypass to original audio when inference misses its deadline;
- listening checks for speech leakage, musical noise, pumping, and scene transitions.

Until those gates pass, the existing level-based AudioWorklet remains the live default and the exported ONNX file is a browser-compatibility artifact only.