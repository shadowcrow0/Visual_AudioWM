# Report: the sound dimension of the SFT experiment, from "confusion counts" to d′

2026-11-10. For the PI. Chinese version: `報告_聲音軸_從混淆次數到dprime.md`.
Every number here is traceable to a note in `review/` or a file in `review/data/`; this document gives the context and the conclusions.

---

## 0. One-page summary

We use SFT (Systems Factorial Technology) to ask whether colour and speech sounds held in working memory are processed in parallel, serially, or coactively.
SFT needs two salience levels per dimension, H and L, calibrated per participant.
Colour has a ready-made perceptual scale (ΔE) and calibrates fine; **sound did not**: in the first pilot (11-03) the sound dimension sat at chance under memory load.

One cause we identified is the sound scale itself: it was the log of Miller & Nicely (1955) confusion counts, not a perceptual unit.
After evaluating seven alternatives we chose to **fit GRT to the M&N confusion matrices and use the fitted d′ as the sound scale**.
Outcome: (1) a table of d′ for 240 consonant pairs replaces the old scale; (2) the fits also show which phonetic features are perceptually separable (voicing, nasality) and which are not (place, manner), so "number of differing features" cannot serve as a scale; (3) in the second pilot (11-10) the calibration found a sound L for the first time.
Unresolved: the H target (drift separation 8.0) was unreachable in both dimensions; with one participant the rule is left unchanged for now.

---

## 1. The experiment and what it demands of the stimulus scale

```
   One trial: remember 2 colours + 2 speech sounds (CV syllables) → probe: one colour with one sound → "did both appear?" Y/N
   SFT cells: colour {H, L} × sound {H, L}; the interaction of RT distributions (SIC, MIC) diagnoses the architecture
   How H / L are set: each participant is calibrated first; an LNRM (log-normal race) fits how drift v changes with
                      stimulus difference x, and the x giving drift separation 8.0 (H) and 1.3 (L) are solved for
```

The only demand on x: **a monotone perceptual scale with meaningful spacing**. Colour uses CIELAB ΔE, where one unit is roughly one just-noticeable difference. Sound needs an equivalent.

## 2. What we saw first (pilot 2026-11-03)

```
   Colour: accuracy rises cleanly with ΔE; RT flat (~0.9 s)
   Sound:  all six levels at chance under load (.40–.67, false alarms .22)
           same person, no load, pure perception: lowest level .50, the rest .83–1.00
```

Three candidate causes: (a) the sound scale orders pairs wrongly; (b) the memory load is too heavy; (c) the listener is Mandarin-L1 and English consonant contrasts are weak for them.

## 3. What was wrong with the old sound scale

Old scale: x = −log10(confusion count + 1), counts summed over three SNRs of M&N. Problems:

1. **Not a unit.** −1.6 means nothing, cannot be compared across studies or against ΔE.
2. **Voicing pairs at the far end.** Noise does not mask voicing, so p–b and k–g are almost never confused in noise (count 1) and were treated as "least similar"; in the memory literature voicing pairs are often the hardest.
3. **Different kinds of difference on one axis.** A place pair and a manner pair are compared by the same count with no reason to believe they share a scale.

## 4. Problem statement

Give the sound dimension an **objective, monotone, perceptually scaled** axis so that the adaptive calibration can find each participant's L (and H) as it does on ΔE.

## 5. Options evaluated

```
   Option                                 What was done                            Result / why not
   ───────────────────────────────────── ──────────────────────────────────────── ──────────────────────────────────────────────
   A Feature distance (no. of M&N features) coded as a candidate axis; perception test  GRT shows one differing feature spans d′ 0.46 (p–t)
                                                                                        to 2.5 (b–d); not a scale. Dropped.
   B Weighted feature distance            computed                                 only 27 distinct values; still assumes additivity. Dropped.
   C Confusion rate −ln p_ij              computed                                 continuous but contaminated by response bias. Dropped.
   D MDS coordinate distance              computed                                 dimensionality arbitrary; coordinates are not units. Dropped.
   E Luce similarity-choice −ln η         ML fit on 120 pairs, mean of 3 SNRs      handles bias, covers all pairs; but what is the unit? → pinned by F
   F GRT fit to M&N                       see §6                                   gives the d′ unit and the separability structure. **Adopted**
   G SNR continuum (one syllable in noise) the old GRTv3 design                     manipulates audibility, not the representational distance
                                                                                   between two remembered items. Fallback.
   H Per-participant 16×16 + GRT          estimated                                ≥ 2000 trials per person. Out of budget.
```

Literature checked in full text alongside:
- Schweppe, Grice & Rummer (2011): serial recall; the set sharing only manner (pa ta ka) hurts recall in every presentation/recall mode, the set sharing only place (da la za) only with spoken recall; the authors **argue against counting features**.
- Roodenrys et al. (2025): consonants differing by one articulatory feature interfere in serial recall, also under visual presentation; citing Wang & Bilger, half their pairs are not acoustically confusable → memory similarity is not purely acoustic.
- Cutler et al. (2004): non-native (Dutch) listeners ≈ 80% of native accuracy, the gap constant across SNR; the one clear L1 effect is **final-position voicing** (Dutch has no final voicing contrast). The corresponding prediction for Mandarin listeners is an inference.
- Conclusion: the M&N ordering can serve as a prior; absolute values cannot; the distance in memory must be measured by the calibration itself.

## 6. The adopted approach: GRT fitted to M&N (route B)

Why not fit the full 16×16: the available package (mdsdt) handles two dimensions; 16 consonants span five feature axes. So the other features are held fixed and feature-pair sub-designs are cut out.

```
   M&N 16×16 × 4 SNR
     │ 11 feature-pair sub-designs (voicing×place, voicing×manner, nasality×place, manner×place, in their contexts)
     v
   mdsdt::fit.grt (equal-variance Gaussian, decisional separability assumed): full / PS_x / PS_y / PI, compared by ΔAIC
     │
     ├─ separability (mainly −6 dB)
     │     voicing, nasality: means do not move with the other axis → separable; voicing independent of place/manner
     │     place: changes more than fourfold with voicing (p–t 0.46 vs b–d 2.5); manner changes with voicing (k–ʃ 3.4 vs g–ʒ 1.1)
     │     → the stimulus space is not a product of feature axes; distance must be looked up per pair
     └─ per-pair d′ = mean separation along that axis; 28 single-feature pairs

   Extending d′ to 120 pairs: on those 28 pairs Luce's −ln η is a line through the origin against the GRT d′
       d′ = 0.911 · (−ln η), Pearson .98
   → all 120 pairs use 0.911·(−ln η), written to stimuli/consonant_dprime.csv
```

For a GRT reader: we are **not** claiming the sound dimension is perceptually separable; the fits say the opposite. Collapsing sound to a scalar x is a convenience; x only has to be monotone. Selective influence is tested in the SFT data with the SIC check, not assumed from the scale.

## 7. Validation: pilot 2026-11-10

**(a) Pure perception, fixed 80 pairs:** hit rate .81→.94 across levels, but trial-level accuracy is unrelated to d′ (r = .06); the closest place pairs (d′ 1.5) were hit at .89, the same as voicing pairs; errors cluster on weak fricatives. Reading: with clean speech the whole scale is at ceiling, which is expected, not evidence of wrong ordering; token levels were checked and do not explain the errors.

**(b) Calibration on the new scale (under memory load):**

```
                     H                    L         accuracy by level
   colour   ΔE 90 (clipped)          39.4      ΔE 2–12 at chance, ≥ 25 all correct
   audio    d′ 4.5 (clipped; 14.7)   1.57      .60–1.00; RT falls from 1.7 s at low levels to 1.0 s at high
```

- **The sound dimension showed a slope for the first time**: both accuracy and RT vary with d′, and the LNRM found L = 1.57 (manner pairs). Three things changed since 11-03 (the "nochange" variant, the key-collection fix, the d′ scale), so the decisive one cannot be isolated.
- **H could not reach 8.0 in either dimension**: the drift function is quadratic (α₂ < 0) and has a peak; this participant's peak lies below 8.0, and the solved H is an extrapolation from the ~3% of posterior draws that have a solution, hence unstable. The program clips to the highest level.

## 8. What we have now

```
   Stimuli   stimuli/consonant_dprime.csv (d′ for 240 pairs); stimuli/sound_pairs.csv (80 pairs for the perception test)
   Code      calibration and main task use the d′ scale; LEVELS_AUDIO = 0.4 / 0.9 / 1.5 / 2.3 / 3.3 / 4.5
   Analysis  grt_mn_subdesigns.py / .R (re-runnable); review/data/grt_subdesigns/ (fit outputs)
   Notes     D1, D2; three full-text checks; two pilot notes; DECISIONS 19–21
   Data      one participant: perception ×2, calibration ×2
```

## 9. Open

1. **The H rule**: keep 8.0 and let the program clip to the top level (current; only the message changes), or fix H at the top level and calibrate only L. One participant is not enough to decide; wait for 3–5.
2. **Load**: sound showed a slope this time, but once, in one person; if the next participant is at chance again, reducing load is on the table (researcher's decision; the number of targets stays for now).
3. **A second listener** (ideally English-L1) on the same table, to separate "person" from "scale".
4. **Talker levels**: T10–T12 are 5–8 dB quieter than the others; align them at the next stimulus revision.

## 10. Limits

M&N: white noise, /ɑ/, American English natives; we use clean speech and Mandarin listeners. Sub-matrices are conditioned on within-subset responses. mdsdt's 2×2 path fixes σ = 1. Perceptual distance ≠ memory distance (Schweppe same direction, Roodenrys opposite). One participant.
