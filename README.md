# Adaptive Audio

**Watch films and TV without constantly reaching for the volume control.**

Adaptive Audio is an experimental project for reducing loud music and sound effects while keeping dialogue at its original, comfortable level. The first target is a **Chrome extension** that processes a streaming tab's audio locally, in real time, while you watch.

The intended experience is simple: ordinary speech sounds the same when processing is enabled, but a loud theme tune, gunfire, an explosion, or a revving engine becomes less intrusive. Adjustments should be smooth, without audible volume pumping or loss of lip-sync.

**Current status:** a working Chrome prototype reduces loud passages using audio levels. AI source separation is being evaluated separately and is **not yet part of the live extension**. Preserving dialogue during overlapping music and effects remains the main unresolved challenge.

## What the project is aiming for

| Situation | Intended behaviour |
| --- | --- |
| Normal dialogue, deeper voices, or a moderately raised voice | Preserve the original level without dipping midway through a sentence. |
| Ordinary background music and everyday sounds | Leave comfortable audio unchanged. |
| Sustained loud music, theme tunes, or engines | Reduce the loud section consistently, without rising and falling between beats. |
| Sudden gunfire, impacts, or explosions | Reduce sharp peaks smoothly, without keeping the following dialogue quiet. |
| Dialogue overlapping loud music or effects | Preserve the speech while reducing the loud background. |

These are acceptance targets, not guarantees of the current prototype. The detailed listening criteria are in [the real-time audio target](docs/realtime-audio-target.md).

## What works today

| Component | Available now | Current limit |
| --- | --- | --- |
| **Chrome extension** | Captures tab audio, reduces loud passages, and supports instant comparison with original audio. | Uses levels rather than sound recognition. Loud speech can still be reduced, including during overlapping effects. |
| **Offline Python processor** | Processes WAV, MOV, and MP4 files with Cinema, Balanced, and Night modes. | Processes the whole mix; these modes are separate from the live extension's settings. |
| **Optional audio classifier** | Uses YAMNet to estimate speech, music, action, and other content for offline gain decisions. | Classifies sounds; it does not extract separate dialogue and effects tracks. |
| **Rolling Python prototype** | Processes successive audio blocks with configurable look-ahead. | Its default two-second buffer delays audio; it is not the Chrome playback engine. |
| **Source separation trials** | Compare estimated dialogue/background separation, audio quality, and processing speed. | No model has yet passed all quality, browser performance, and playback requirements. |

Local listening has confirmed playback on Netflix with acceptable sync in the tested setup. This is not a guarantee of compatibility with every title, streaming service, or device. A smart TV version is a longer-term possibility; there is no TV app in this repository.

## Try the Chrome extension

The current prototype requires **Chrome 116 or newer**. It needs no Python environment, model download, API key, or backend service. Captured audio is processed locally; the extension does not record or upload it.

1. Open `chrome://extensions` and enable **Developer mode**.
2. Choose **Load unpacked** and select this repository's `chrome-extension` folder.
3. Play a video, then click the extension icon to start processing. The badge shows `CMP`.
4. Press **Alt+Shift+P** to switch between processed audio (`CMP`) and original audio (`AUD`). Compare at the same player and speaker volume.
5. Click the icon again, or press **Alt+Shift+A**, to stop capture. The same shortcut starts capture again.

**Netflix fullscreen:** enter the player's fullscreen mode **before** starting capture, then use **Alt+Shift+A**. Chrome may prevent entering player fullscreen after capture starts. Configure shortcuts at `chrome://extensions/shortcuts` if needed.

See the [extension guide](chrome-extension/README.md) for badge meanings, troubleshooting, and browser verification instructions.

## Why source separation is the next step

A volume detector sees the level of the entire soundtrack. It cannot reliably distinguish a loud voice from loud music. If dialogue and an explosion overlap, turning down the mixed signal turns down both. Adding a classifier can help decide *when* to act, but does not solve that overlap problem.

The intended default is therefore to **estimate dialogue and music/effects separately**, then reduce only the loud non-dialogue content. The browser receives a mixed soundtrack, so these would be estimated sources, not access to the studio's original production tracks.

The planned processing path is:

```text
Captured tab audio
    → Low-latency dialogue / background separation
    → Smooth reduction of loud background events, with dialogue held at its original level
    → Recombine and play in sync with the video
```

Separation quality matters as much as speed: speech leaking into the estimated background can still become quieter when that background is reduced. A larger buffer alone cannot fix this and can make audio late relative to video. The target design should reproduce the original mix when no reduction is needed and fall back to original audio if separation fails or cannot keep up.

### What the model trials have shown

| Candidate | Finding from the recorded local trials |
| --- | --- |
| **BandIt v2** | Useful offline quality reference, but the tested processing time and chunking are unsuitable for live extension playback. |
| **DeepFilterNet3** | Fast in an isolated browser test, but the tested remix reduced dialogue during overlapping music. |
| **GTCRN** | Fast in a native streaming test, but also reduced dialogue during overlapping music. |
| **DTLN** | Downloadable streaming ONNX models; fast natively and close to the original on isolated speech, but the tested remix still lost dialogue during music. |
| **Rapidly SDK** | The native demo showed promising throughput. Unwatermarked quality, stereo preservation, and browser SDK integration still need evaluation. |

No licensed model has been selected, and the current extension does not depend on one. Native processing speed alone does not prove that a model is suitable for Chrome.

See the [offline separation findings](evaluation/separation-findings.md), [browser model findings](evaluation/dfn-browser-findings.md), [GTCRN findings](evaluation/causal-separator-findings.md), [DTLN and streaming Denoiser findings](evaluation/dtln-findings.md), and [licensed SDK trial](evaluation/rapidly-trial.md) for measurements and reproduction details.

## Run the offline tools

For development and file comparisons, install **Python 3.10+** and run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

adaptive-audio input.wav --mode balanced --output output.wav
adaptive-audio input.mov --mode balanced --output output.mov
adaptive-audio input.mp4 --mode night --output output.mp4
```

WAV input must be uncompressed 16-bit PCM. Video processing uses bundled FFmpeg to process the first audio track as stereo, copy the video, and encode the replacement audio as AAC. Always specify a MOV or MP4 output filename when processing video.

The [offline workflow guide](docs/offline-workflow.md) covers modes, optional classification, rolling processing, measurement reports, and the earlier *Tears of Steel* listening trials. Source media, generated previews, and downloaded models are kept outside Git; they are not included in a fresh checkout.

### Prepare training windows

The initial DnR v3 subset is pinned in `evaluation/datasets/dnr-v3-starter.json`. After downloading the planned files, prepare aligned 16 kHz stems and split-preserving four-second windows with:

```powershell
python -c "import json; from adaptive_audio.dataset_audio import prepare_dataset; plan=json.load(open('evaluation/datasets/dnr-v3-starter.json', encoding='utf-8')); prepare_dataset(plan, 'audio/datasets/dnr-v3', sample_rate=16000)"
```

The resulting `audio/datasets/dnr-v3/windows-16000-*.jsonl` indexes can be consumed with `adaptive_audio.training_data.iter_batches`. The loader returns batches containing `mixture`, `speech`, `music`, and `sfx` arrays and validates cache bounds and stem presence before yielding data.

### Train the baseline separator

Training is optional and uses the existing PyTorch separation environment, keeping PyTorch out of the lightweight offline install:

```powershell
$env:PYTHONPATH = "src"
models\separation-venv\Scripts\python.exe -m adaptive_audio.train_cli audio\datasets\dnr-v3 audio\datasets\dnr-v3\checkpoints --epochs 5 --batch-size 4 --device auto
```

The command learns a compact speech magnitude mask from the mixture and writes `checkpoint_latest.pt` plus per-epoch checkpoints. Continue an interrupted run with `--resume audio\datasets\dnr-v3\checkpoints\checkpoint_latest.pt --epochs 10`. This is a baseline for measuring separation quality, not yet a browser-ready or production separator.

Evaluate the held-out test split before changing the model or training budget:

```powershell
models\separation-venv\Scripts\python.exe -m adaptive_audio.eval_train_cli audio\datasets\dnr-v3\checkpoints\checkpoint_latest.pt audio\datasets\dnr-v3 audio\datasets\dnr-v3\evaluation-test.json --split test --device auto
```

The first five-epoch checkpoint scored `+0.46 dB` average SI-SDR improvement over the unprocessed mixture on 239 non-silent test windows. One silent-speech window is reported separately because SI-SDR is undefined there. This objective result is an initial baseline; listening tests, artifact checks, and latency measurements are still required.

Render a held-out listening set with the original mixture, clean reference, estimated speech, and a conservative background reduction:

```powershell
models\separation-venv\Scripts\python.exe -m adaptive_audio.eval_train_cli audio\datasets\dnr-v3\checkpoints\checkpoint_latest.pt audio\datasets\dnr-v3 audio\datasets\dnr-v3\evaluation-test.json --split test --preview-dir audio\datasets\dnr-v3\previews\test-0000 --preview-index 0 --background-gain 0.35 --device auto
```

The preview report records RMS, peak, and clipped-sample counts. Three representative test windows rendered without clipping; listen for speech loss, musical noise, and pumping before considering a larger model or browser integration.

Measure offline inference throughput with:

```powershell
models\separation-venv\Scripts\python.exe -m adaptive_audio.benchmark_train_cli audio\datasets\dnr-v3\checkpoints\checkpoint_latest.pt audio\datasets\dnr-v3 --split test --batch-size 4 --batches 20 --device cpu
```

The current checkpoint processed 320 seconds of held-out audio at `91.99x` real time (`43.5 ms` per four-second window) on CPU. This is an offline batch result; browser work still needs frame-level latency, sustained-load, stereo, fallback, and lip-sync trials.

### Client-side model packaging

The selected direction is local browser processing with ONNX Runtime Web or a comparable WASM/WebGPU runtime. Export the current baseline for compatibility experiments with:

```powershell
$env:PYTHONPATH = "src"
models\separation-venv\Scripts\python.exe -m adaptive_audio.export_model audio\datasets\dnr-v3\checkpoints\checkpoint_latest.pt models\browser\speech-mask.onnx
```

The exporter writes a SHA-256 sidecar manifest. This artifact is a packaging milestone only: the current model requires four-second spectrogram windows, so putting it directly in the live AudioWorklet would add unacceptable latency. A causal, short-frame model must pass latency and listening gates before replacing the existing reducer. No server or commercial SDK is required for this client-side path.

See the [client-side integration decision](docs/client-side-integration.md) for the browser runtime boundary and the remaining causal-model requirements.

## Next development stages

Keep each stage independently reviewable and committable:

1. **Qualify a separator.** Compare ordinary and overlapping speech, sustained music, and sharp effects against original audio at a fixed listening volume. Reject candidates with audible dialogue loss or separation artifacts.
2. **Prove browser performance.** Verify stereo preservation, sustained processing speed, total playback delay, and lip-sync on representative hardware before integrating a model as the default.
3. **Integrate background reduction.** Apply stable gain changes to estimated music/effects, handle scene transitions, and recover to original playback when processing cannot keep up.
4. **Validate real viewing.** Recheck dialogue, pumping, effects, fullscreen, and sync across longer sessions and different streaming content.
5. **Simplify installation.** Package a local processing path that avoids manual model or service setup wherever possible.

## Repository guide

- [`chrome-extension/`](chrome-extension/) — live tab capture and the current level-based audio processor.
- [`src/adaptive_audio/`](src/adaptive_audio/) — offline processing, classification, rolling processing, and stem remix utilities.
- [`evaluation/`](evaluation/) — scene definitions, model trial runners, and recorded findings.
- [`docs/`](docs/) — listening targets and detailed offline instructions.
- [`tests/`](tests/) — Python automated checks.

To run the Python checks after installing the project:

```powershell
python -m pip install pytest
python -m pytest -q
```

Automated checks help verify processing behaviour. Listening comparisons and browser playback checks are still required to establish dialogue preservation and a seamless viewing experience.
