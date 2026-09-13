# First BandIt v2 stem trial

Run on 2026-09-13 with the official `v2-multi` checkpoint, pinned
`bandit-infer` commit `d45cdec634bf1ee01cdd2acea74a2d100e639c8a`,
PyTorch 2.11.0+cu128, and an NVIDIA GeForce RTX 5090. All inputs were 48 kHz
stereo 16-bit WAV; inference used CUDA and a 12-second segment. The trial
applied a **fixed** 6 dB music/effects reduction to make source leakage easy
to audition. It did not run the intended dynamic event compressor.

| Segment | Inference for 12 s | Preview observation |
| --- | ---: | --- |
| `service_middle_original.wav`, 0–12 s | 16.16 s | Speech-dominant YAMNet windows; total RMS change below 0.01 dB. |
| `Tears of Steel`, 197–209 s | 15.52 s | Both estimated speech and music are present; total RMS change varies from -0.56 to -2.26 dB across 3-second windows. Dialogue quality during overlap still needs listening. |
| `Tears of Steel`, 208–220 s | 15.74 s | Model assigned nearly all signal to music; preview RMS fell about 5.95 dB. A local YAMNet window near 215.5 s labeled action, so the model's music/effects distinction is unreliable here. Both stems would be processed as non-dialogue. |

The first uncached model load took 22.5 seconds including download; later loads
took about 1.7 seconds. The three inference runs took 1.29–1.35 seconds per
second of audio. Since the model is slower than playback on this machine and
uses overlapping 8-second chunks, it cannot be the extension's real-time
default as-is. A faster **low-look-ahead** separator and browser integration
are still required. Faster average throughput by itself would not solve
lip-sync delay.

The local audition files are ignored by Git:

- `audio/separation/service_middle/{original,speech_estimate,music_estimate,effects_estimate,reduced_preview}.wav`
- `audio/separation/tos_transition/{original,speech_estimate,music_estimate,effects_estimate,reduced_preview}.wav`
- `audio/separation/tos_action/{original,speech_estimate,music_estimate,effects_estimate,reduced_preview}.wav`

The overall RMS comparisons are measurements of the mixed output, not proof
that dialogue stayed unchanged during overlap. Listen to the music and effects
stems for recognizable words, then compare original and preview at one speaker
volume. That listening result will determine whether this separator is useful
as an offline quality reference while a streaming model is sought.
