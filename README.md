# Adaptive Audio MVP

A runnable offline baseline for reducing uncomfortable loudness changes while retaining some quiet/loud contrast. It processes **uncompressed 16-bit PCM WAV** files in Cinema, Balanced, or Night mode. This first stage uses short-window RMS levels and deterministic gain planning; it does **not** yet identify dialogue, music, or action, or measure LUFS.

## Set up

Python 3.10+ is required. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run it:

```powershell
adaptive-audio input.wav --mode balanced
adaptive-audio input.wav --mode night --output output.wav
```

The default output is `input_balanced.wav` (or the selected mode). The original is never overwritten. For a quick test from a video file, install FFmpeg separately and extract a PCM WAV:

```powershell
ffmpeg -i movie.mkv -map 0:a:0 -ac 2 -ar 48000 -c:a pcm_s16le movie.wav
adaptive-audio movie.wav --mode balanced
```

This version outputs audio only. It does not remux a processed track into a video file. It uses the same gain for all channels, smooths the gain curve with offline look-ahead, and limits peaks locally to a -1 dBFS sample-peak ceiling. The ceiling is **sample peak**, not true peak; check the output with a suitable meter before using it for critical listening.

WAV processing uses two passes: one scans 100 ms frame levels and peaks, then the other writes samples in chunks. Memory use is bounded by the chunk size plus the compact frame-level gain plan, so a full soundtrack does not need to be loaded at once.

## Tears of Steel test clip

The source MOV at `audio/ToS-4k-1920.mov/ToS-4k-1920.mov` is kept outside Git. A 60-second excerpt from 06:30–07:30 has been extracted and processed locally:

- `audio/tos_excerpt_original.wav`
- `audio/tos_excerpt_balanced.wav`
- `audio/tos_excerpt_night.wav`
- `audio/tos_excerpt_night_v2.wav` (revised Night settings with less dialogue reduction)

The full soundtrack has also been extracted to `audio/tos_full_original.wav` and processed in Balanced mode as `audio/tos_full_balanced.wav`. These local WAV files are ignored by Git.

Compare the original, Balanced, and either Night version at the **same player volume**. Listen for clear dialogue, the impact of louder events, pumping, and audible distortion. To recreate the excerpt with FFmpeg installed:

```powershell
ffmpeg -ss 00:06:30 -i "audio/ToS-4k-1920.mov/ToS-4k-1920.mov" -t 60 -map 0:a:0 -ac 2 -ar 44100 -c:a pcm_s16le audio/tos_excerpt_original.wav
adaptive-audio audio/tos_excerpt_original.wav --mode balanced --output audio/tos_excerpt_balanced.wav
adaptive-audio audio/tos_excerpt_original.wav --mode night --output audio/tos_excerpt_night_v2.wav
```

## Modes

Cinema makes small changes, Balanced gives quieter passages a gentle lift and reduces loud passages, and Night applies the strongest change. Mode values are provisional and should be tuned against real listening samples. Since this stage has no classifier, a quiet sound effect can be raised just like quiet dialogue. That is the reason for the next stage.

## Evaluation

The reproducible Tears of Steel windows are listed in `evaluation/scenes.json`. The first set samples the opening, middle, and closing minute; they are time selections, not verified content labels. Keep the source and generated WAV files in `audio/`, which Git ignores.

Compare two aligned WAV files with:

```powershell
adaptive-audio-report audio/tos_excerpt_original.wav audio/tos_excerpt_balanced.wav
adaptive-audio-report audio/tos_excerpt_original.wav audio/tos_excerpt_night_v2.wav --json
```

The report gives duration, sample peak, and the 10th, 50th, and 90th percentiles of 100 ms RMS levels. Its level range is P90 minus P10. These are repeatable **dBFS** measurements, not LUFS or standardized loudness range. They cannot measure dialogue intelligibility, pumping, or listening comfort; use the same player volume for those comparisons and record observations alongside the numbers.

## Next stage: semantic classifier

Add an independent component that outputs time-stamped probabilities for `speech`, `music`, `action`, and `other`. The decision engine can then use those labels with measured level to choose gains. Keep model inference separate from DSP so that the gain engine remains deterministic and can be tested with known labels. Compare the classifier-enabled output with this baseline on the same 10–20 scenes; prioritize fewer remote-control volume changes, dialogue clarity, preserved dynamics, and absence of pumping or clipping over classification accuracy alone.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```
