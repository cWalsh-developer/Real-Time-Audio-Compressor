# Real-time listening target

The Chrome extension should leave ordinary programme audio, including normal
dialogue and modest changes in speaking level, perceptually unchanged. When
music, effects, engines, gunfire, or explosions become genuinely loud, it should
reduce the uncomfortable part smoothly and return to unity gain when the event
ends. Switching the extension on during ordinary speech should be difficult to
hear. Audio should remain synchronized with video.

## What must be measured

Sample peaks and short-term loudness answer different questions. A gunshot can
have a brief high peak; theme music or a revving engine can remain loud for
seconds without an exceptional individual sample. The live processor needs a
fast safety path for transients and a slower path for sustained loudness. Both
need stable gain behavior across brief dips and a controlled return to unity.

A volume detector sees the level of the **whole stereo mix**. It cannot know
whether a high-level passage is music or a raised voice. It also cannot lower
an explosion while holding simultaneous dialogue at exactly the original level:
one gain applied to a mixed signal changes both. Classifying the sound may help
decide when to change gain, but preserving speech *during* overlapping effects
requires source separation or access to separate dialogue and effects tracks.

The current extension is a level-only prototype. Do not treat its synthetic
sine-wave checks as proof that speech is preserved on actual film or TV audio.

## Evidence from the local soundtrack

`Tears of Steel` already has a local source WAV and YAMNet score windows. Run:

```powershell
python evaluation/level_overlap.py audio/tos_full_original.wav audio/tos_full.labels.json
```

With a 0.6 dominant-score cutoff, 260 likely-speech windows and 796
likely-music windows are available. Their approximately 1-second RMS levels overlap: the
speech P90 is about -12.3 dBFS and the music P90 about -11.6 dBFS. About 10%
of likely-speech windows and 12% of likely-music windows exceed -12 dBFS. A
single fixed dBFS threshold therefore cannot reliably choose music while
leaving all speech untouched. These YAMNet categories are heuristic, not
listener-verified labels; the finding is a warning about threshold tuning, not
a model accuracy claim.

## Acceptance checks for the next processor

- Normal and moderately raised **isolated speech**: compare bypass and
  processed playback at the same speaker level; no audible mid-sentence gain
  movement. Record measured gain over manually labeled speech windows.
- Sustained loud music and engines: reduce the uncomfortable level while
  avoiding audible up/down gain movement during brief quieter beats.
- Short gunfire and explosions: catch the onset and avoid clipping without a
  long reduction tail over the following dialogue.
- Scene transitions: regain the original speech level promptly after a loud
  event. Check lip-sync on Netflix and ordinary video tabs after each change.
- Overlapping speech and effects: report separately. A global gain processor
  cannot meet simultaneous speech-preservation and effect-reduction targets.

## Stem-first processing target

The intended default is to **estimate dialogue, music, and effects stems first**,
then apply event reduction to music and effects while passing the dialogue stem
at unity gain. The browser only receives the mixed soundtrack, not the original
production tracks. A separation model therefore estimates stems, and speech
leakage into music/effects could still make dialogue quieter. Judge that by
listening to the separated stems and to overlapping speech/effect scenes.

For an estimated music/effects stem, a remix can use the untouched input mix as
its reference and subtract only the requested reduction of those stems. At zero
reduction this is exactly the original input, even if estimated stems do not
sum perfectly. It does not remove leakage during reduction; that requires a
better separator or conservative gain decisions.

The next committable stages are (1) make an offline stem-first listening and
latency trial on local film/TV material; (2) compare dialogue level, leakage,
music/effects reduction, artifacts, and input-to-output delay against bypass;
(3) implement a streaming separator only if it can keep up on representative
hardware with acceptable playback delay and recovery; and (4) enable it as the
live default after browser listening and lip-sync checks. If separation misses
its deadline or fails, play the original audio rather than hold stale audio.
The existing level-only extension remains the working prototype until the
stem-first route passes those checks.
