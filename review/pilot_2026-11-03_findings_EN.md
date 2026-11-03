# Pilot 2026-11-03: what we learned (participant yycy / yycc, one session each)

Files: `data/yycy_simple_calib_trials.csv`, `data/yycy_simple_calib.json` (calibration inside the working-memory
task, `SFT_WM_nochange_EN.py`, targets 8.0 / 1.3, colour levels dE 2–90), `data/yycc_sound_perception.csv/.json`
(`sound_perception_EN.py`, pure perception, no memory load). Same person, non-native English listener.

## 1. The key-collection bug is fixed

Colour block: 0 of 48 trials without a response (19 Oct pilot: 7 of 72). Sound block: 2 of 48.
Cause and fix: DECISIONS 17 (keys are now collected from probe onset; PsychoPy 2026's
`getKeys(waitRelease=False)` only returns keys still held down, so quick presses were being dropped).

## 2. Colour inside the WM task: accuracy is clean, reaction time is flat

```
   probe dE        2      6      12     25     50     90     same (AA)
   accuracy       .00    .17    .71    1.00   1.00   1.00    .73  (false alarm .27)
   median RT      1.03   1.03   1.02   .89    .89    .89     1.21 s
```

- Accuracy rises from chance to ceiling between dE 6 and 25; the boundary is near dE 12.
- Fit (lnrm2, 30 "different" trials): L = 43 dE, H unsolvable in 89 % of posterior draws → clipped to 90.
  L = 43 is not a mistake: SFT wants both H and L at near-ceiling accuracy and only the speed to differ
  (Houpt et al., 2014, p. 321); a drift separation of 1.3 corresponds to ~97 % accuracy.
- **The problem is the RT.** From dE 25 to 90 the median is 0.89 s everywhere; even the wrong answers at
  dE 2–6 take 1.03 s. The SIC is built from RT differences between H and L; with H and L equally fast it is
  a flat line whatever the targets are. RTs cluster at 0.8–1.0 s, i.e. around the end of the 1 s probe
  (the syllable lasts 0.6 s): the response seems to wait for the probe to finish rather than track difficulty.
  This is a timing/instruction issue, not a calibration issue.

## 3. Sound inside the WM task: at chance at every level

```
   dissimilarity x   -2.6   -2.1   -1.6   -1.1   -0.7   -0.3    same (AA)
   accuracy           .50    .40    .57    .50    .50    .67     .70  (false alarm .22)
   median RT          1.58   1.49   1.45   1.45   1.16   1.57    1.29 s
```

Even the most dissimilar pairs (e.g. m vs theta) are at .67. The fit cannot solve for H or L (99.8 % / 96.6 %
of draws dropped); both were clipped to the ends of the axis. Nothing can be calibrated on this dimension as
the task stands. The participant also reported not attending to the sound during the colour block, which is
the expected strategy once the sound cannot be discriminated.

## 4. The same consonant pairs without memory load: discriminable

`sound_perception_EN.py`: syllable 1 (0.6 s) → 0.5 s silence → syllable 2 (0.6 s) → "same consonant?",
same talker, feedback, 108 trials.

```
   dissimilarity x   -2.6   -2.1   -1.6   -1.1   -0.7   -0.3
   hit rate           .50    .83    1.00   .92    1.00   1.00
   d'                 0.59   1.56   2.28   1.97   2.32   2.32
   median RT          1.41   1.22   1.02   1.33   1.03   1.21 s      false alarm .28, 1 no response
```

Five of the six levels are at .9–1.0 when nothing has to be remembered; inside the WM task the same levels sit
at .5. **The bottleneck is the memory load (remember two consonants, then compare), not the stimuli.**
Two caveats: d' peaks at only 2.3 and the false-alarm rate is .28, so even without load these pairs are only
"reasonably" separable for this listener; and the most similar level (x = −2.6) is at chance even here and
carries no information in the calibration.

## 5. Non-native listener

The participant's first language is not English and they report not hearing the consonant contrasts. Part of
the chance-level sound performance may be this rather than the design. To separate the two, run
`sound_perception_EN.py` on a native English speaker: if they reach d' ≈ 3–4 and keep it under load, the
population, not the task, is the issue. Recorded under open items in `DECISIONS.md`.

## 6. The target 8.0 / 1.3 and the stimulus ceiling

With targets 8.0 / 1.3 (chosen for coactive power, DECISIONS 3) the colour H for this participant lies beyond
dE 90 and was clipped. Whether a ceiling rule (H = the largest presentable difference, L scaled by the same
ratio) is worth adding depends on §2: if RT does not vary with dE, no choice of H and L gives a usable SIC.
Fix §2 first.

## 7. "Same" trials: should their RT be compared?

Under the task rule ("y only if both match") a "same" answer requires checking both dimensions, so "same"
RTs are expected to be slower than "different" RTs, and the participant's feeling of getting stuck on them is
the exhaustive decision, not an error. Observed: colour block same 1.21 s vs different 0.91 s; sound block
same 1.29 s vs different 1.45 s (sound "different" is slow because the decision is at chance).
For SFT the same cell (A × A) is a single cell with no salience factor, so it enters no SIC and no capacity
coefficient; its accuracy (here .70–.73, false alarms .22–.27) is the check that the participant is not
answering "different" by default, and its RT is reported descriptively. Nothing to change in the analysis.

## 8. What has to be decided (not code)

1. Reduce the memory load (e.g. 1 colour + 1 sound instead of 2 + 2) or keep 2 + 2 and change the sound
   stimuli to something far more distinct. §4 says load is the first lever.
2. Participant population: native English listeners, or stimuli that non-native listeners can separate.
3. Response timing: get RT to track difficulty (instruction "as fast as you can", or a probe that does not
   anchor responses at 1 s). Without this the SIC is flat regardless of calibration.
4. Only then: the ceiling rule for targets that fall outside the presentable range.
