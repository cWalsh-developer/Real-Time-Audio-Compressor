# Additional source evaluation

Eight 60-second windows from the newly added videos are recorded in `additional_scenes.json`. Their local audio files live in `audio/eval/` and are ignored by Git. Each `<name>_original.wav` was extracted from the first audio track; each `<name>_balanced_v3.wav` was processed with the committed Balanced v3 curve. The Big Buck Bunny source has a second 5.1 audio track, which this evaluation does not use.

| Window | 100 ms RMS P90–P10 range, original → Balanced v3 | Sample peak, original → Balanced v3 | Dominant model category |
| --- | ---: | ---: | --- |
| bbb_opening | 18.79 → 18.04 dB | -1.76 → -1.00 dBFS | Music |
| bbb_middle | 14.23 → 7.13 dB | -0.39 → -1.00 dBFS | Music |
| bbb_late | 16.37 → 9.54 dB | -4.03 → -4.48 dBFS | Music |
| service_opening | 21.68 → 21.60 dB | -15.27 → -9.86 dBFS | Music |
| service_middle | 29.86 → 29.84 dB | -10.31 → -4.33 dBFS | Speech |
| service_late | 18.09 → 16.69 dB | -12.56 → -9.88 dBFS | Speech / music |
| reel2013 | 6.75 → 5.28 dB | 0.00 → -2.38 dBFS | Music |
| cycles2015 | 7.79 → 5.12 dB | 0.00 → -3.35 dBFS | Music |

The range is a percentile spread of 100 ms RMS dBFS measurements, **not LUFS loudness range**. Categories come from the experimental YAMNet grouping and have not been checked by a listener. In the `service_middle` window, model-identified likely-speech audio has a median level near -38 dBFS in the source and receives the current curve's maximum 6 dB lift. That source is mastered much more quietly than *Tears of Steel*, so listening at a fixed speaker volume matters more than comparing these peak numbers.

To regenerate a window, use its source and start time from the JSON manifest:

```powershell
ffmpeg -ss 00:18:00 -i "audio/service-mbrs-ntscrm-00009301-00009301.mp4" -t 60 -map 0:a:0 -ac 2 -c:a pcm_s16le audio/eval/service_middle_original.wav
adaptive-audio audio/eval/service_middle_original.wav --mode balanced --output audio/eval/service_middle_balanced_v3.wav
adaptive-audio-report audio/eval/service_middle_original.wav audio/eval/service_middle_balanced_v3.wav
```

For listening, compare each original and processed pair at the same speaker volume. Check whether speech remains clear in the two `service` windows, whether music changes feel natural in the reels and *Big Buck Bunny*, and whether there is audible pumping or distortion. The measurements alone do not establish a listening-quality verdict.
