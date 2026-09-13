# Adaptive Audio MVP

A runnable offline processor for reducing uncomfortable loudness changes while retaining some quiet/loud contrast. It accepts **MOV, MP4, or uncompressed 16-bit PCM WAV** files in Cinema, Balanced, or Night mode. By default it uses short-window RMS levels and deterministic gain planning. An optional pretrained classifier can inform the gain plan. It does not yet measure LUFS.

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
adaptive-audio input.mov --mode balanced --output output.mov
adaptive-audio input.mp4 --mode balanced --output output.mp4
adaptive-audio input.mov --mode balanced --ai --output output_ai.mov
```

The default output adds `_balanced` before the input extension. The original is never overwritten. MOV and MP4 processing use a bundled FFmpeg binary: it extracts the **first audio track** to stereo PCM, processes it, then copies the original video stream and encodes the new audio as AAC. It copies subtitle streams when present. Other audio tracks are not included in the output.

To extract audio manually with a separate FFmpeg installation:

```powershell
ffmpeg -i movie.mkv -map 0:a:0 -ac 2 -ar 48000 -c:a pcm_s16le movie.wav
adaptive-audio movie.wav --mode balanced
```

The processor uses the same gain for all channels, smooths the gain curve with offline look-ahead, and limits peaks locally to a -1 dBFS sample-peak ceiling. The ceiling is **sample peak**, not true peak; check the output with a suitable meter before using it for critical listening.

WAV processing uses two passes: one scans 100 ms frame levels and peaks, then the other writes samples in chunks. Memory use is bounded by the chunk size plus the compact frame-level gain plan, so a full soundtrack does not need to be loaded at once.

## Tears of Steel test clip

The source MOV at `audio/ToS-4k-1920.mov/ToS-4k-1920.mov` is kept outside Git. A 60-second excerpt from 06:30–07:30 has been extracted and processed locally:

- `audio/tos_excerpt_original.wav`
- `audio/tos_excerpt_balanced.wav`
- `audio/tos_excerpt_night.wav`
- `audio/tos_excerpt_night_v2.wav` (revised Night settings with less dialogue reduction)

The full soundtrack has also been extracted to `audio/tos_full_original.wav` and processed in Balanced mode as `audio/tos_full_balanced.wav`. These local WAV files are ignored by Git.
The full MOV was processed directly as `audio/ToS-4k-1920_balanced.mov`; its video stream was verified byte-for-byte against the source stream.

After full-film listening showed the original Balanced curve lowered dialogue too far, `audio/tos_full_balanced_v2.wav` and `audio/ToS-4k-1920_balanced_v2.mov` were rendered with a revised Balanced curve. The old files remain for comparison. On 260 model-identified likely-speech windows, the median level change versus the original WAV is +0.93 dB in v2, compared with -4.65 dB in the old Balanced WAV. These are measured levels, not a substitute for listening. The v2 MOV retains the original duration and video stream.

`audio/tos_full_balanced_v3.wav` and `audio/ToS-4k-1920_balanced_v3.mov` add a small amount of reduction only toward the loud end of the Balanced curve. Across the full-film 100 ms measurements, input windows around -20 dBFS remain about unchanged, while windows at -10 dBFS or louder receive a median 7.48 dB reduction versus 5.98 dB in v2. The v3 MOV retains the original duration and video stream.

Compare the original, Balanced, and either Night version at the **same player volume**. Listen for clear dialogue, the impact of louder events, pumping, and audible distortion. To recreate the excerpt with FFmpeg installed:

```powershell
ffmpeg -ss 00:06:30 -i "audio/ToS-4k-1920.mov/ToS-4k-1920.mov" -t 60 -map 0:a:0 -ac 2 -ar 44100 -c:a pcm_s16le audio/tos_excerpt_original.wav
adaptive-audio audio/tos_excerpt_original.wav --mode balanced --output audio/tos_excerpt_balanced.wav
adaptive-audio audio/tos_excerpt_original.wav --mode night --output audio/tos_excerpt_night_v2.wav
```

## Modes

Cinema makes small changes, Balanced now keeps roughly -20 dBFS passages close to their original level while reducing loud passages, and Night applies the strongest change. Mode values are provisional and should be tuned against real listening samples. Without `--ai` or `--labels`, a quiet sound effect can be raised just like quiet dialogue.

## Evaluation

The reproducible Tears of Steel windows are listed in `evaluation/scenes.json`. The first set samples the opening, middle, and closing minute; they are time selections, not verified content labels. Keep the source and generated WAV files in `audio/`, which Git ignores.

Additional windows from the newly added MP4 and MOV files, with measurements and listening targets, are in `evaluation/additional_scenes.json` and `evaluation/additional_scenes.md`. Full Balanced v3 MP4 outputs are available locally as `audio/bbb_sunflower_balanced_v3.mp4` and `audio/service_balanced_v3.mp4`.

Compare two aligned WAV files with:

```powershell
adaptive-audio-report audio/tos_excerpt_original.wav audio/tos_excerpt_balanced.wav
adaptive-audio-report audio/tos_excerpt_original.wav audio/tos_excerpt_night_v2.wav --json
```

The report gives duration, sample peak, and the 10th, 50th, and 90th percentiles of 100 ms RMS levels. Its level range is P90 minus P10. These are repeatable **dBFS** measurements, not LUFS or standardized loudness range. They cannot measure dialogue intelligibility, pumping, or listening comfort; use the same player volume for those comparisons and record observations alongside the numbers.

## Optional semantic classifier

The classifier boundary is defined in `adaptive_audio.classification`: it returns time-stamped scores for `speech`, `music`, `action`, and `other`. `ManualClassifier` reads known labels from JSON; `evaluation/example_labels.json` shows the format. An optional [YAMNet ONNX model](https://huggingface.co/audiomagic/yamnet-onnx) adapter can generate scores from WAV audio:

```powershell
python -m pip install -e ".[ai]"
adaptive-audio-classify audio/tos_excerpt_original.wav --output audio/tos_excerpt.labels.json
```

The first run downloads pinned model files (about 16 MB) to `models/yamnet/` and verifies their SHA-256 hashes. Model files and generated labels are kept out of Git. The adapter uses FFmpeg to downmix and resample to the 16 kHz mono input expected by [Google's YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet). It groups selected AudioSet classes into the four project categories; these normalized category scores are **heuristic, not calibrated probabilities**. Model windows overlap.

Use `--ai` on WAV or MOV input to classify and process in one command, or pass existing scores with `--labels path/to/labels.json`. Without either option, the established level-only processing stays the same. The content-aware decision adds up to 3 dB of speech protection and 2 dB of action reduction before gain smoothing and peak limiting. Music currently receives the baseline treatment. The model-informed output is experimental and should be compared by ear at the same volume against the established Balanced and Night outputs.

For Tears of Steel, `audio/ToS-4k-1920_balanced_ai.mov` is a local full-film AI-assisted comparison. Its duration and copied video stream were verified against the source.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```
