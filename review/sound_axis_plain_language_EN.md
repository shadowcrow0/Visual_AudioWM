# Where the sound axis comes from: a plain-language version for the advisor (2026-11-10)

Companion files: `review/consonant_distance_how.md` (the metrics), `review/子音距離_怎麼算.md` (Chinese),
`stimuli/sound_pairs.csv` (the fixed 80 pairs), `review/millerNicely1955_噪音混淆_逐SNR.md` §7–8 (evidence levels).

## 1. The story, in order

1. **A consonant is five switches** (vocal folds vibrating or not, air through the nose or not, a burst or a hiss, how long the hiss lasts, where the mouth is closed). This is Miller & Nicely's (1955) analysis.
2. **How alike two consonants sound depends on WHICH switches differ, not on how many.** A difference in "where the mouth is closed" is almost inaudible (p–k); a difference in "vocal folds" is easy to hear (p–b). This comes from M&N's confusion tables: 4,000 judgements at each of six noise levels.
3. **What was wrong before**: pairs that are rarely confused in M&N's table (p–b, k–g) were treated as "very different", and only six foils per consonant were kept, so the pairs that differ on the most switches (s–m, f–n) were never used.
4. **How distance is computed now**: the standard similarity-choice model of mathematical psychology (Luce 1963; Shepard 1957) splits the confusion table into "how similar two sounds are" and "which response the listener prefers", keeps only the first, and takes −log as the distance. It is the same quantity Nosofsky's categorisation and recognition-memory models use.
5. **Fixed stimuli**: computed once, 80 pairs (5 distance levels × 16 consonants), fixed talkers, written to a table; the experiment only reads the table.
6. **Not yet shown**: this distance is what native listeners hear in noise. Whether memory, and non-native listeners, order the pairs the same way has to be tested with our own perception test and WM data.

## 2. How other studies of speech memory compute the distance between consonants (from two literature checks; abstract level only)

```
   Method                              Who used it                                  How it is computed
   ──────────────────────────────────  ───────────────────────────────────────────  ──────────────────────────────────────────────
   Shared-feature count /              Wickelgren 1966; Schweppe et al. 2011;       Count how many articulatory features two
   feature overlap                     Roodenrys et al. 2025                        consonants share; use it to predict serial-recall
                                                                                    errors and interference
   Error rate per feature              Wickelgren 1966; Klatt 1968                  Split the recall-confusion matrix into one 2×2
                                                                                    table per feature; see which feature is lost most
                                                                                    in memory (Klatt: voicing retained, place lost)
   Perceptual confusions borrowed      Conrad 1964                                  Measure a letter-confusion matrix in noise first,
   as the similarity measure                                                        then correlate it with recall confusions
   MDS geometric distance              Shepard 1972 (on the M&N matrices)           Confusions → similarities → multidimensional
                                                                                    scaling coordinates; distance = distance in that space
   Similarity-choice model η           Luce 1963; Shepard 1957; Nosofsky 1986       Estimate "similarity" and "response bias" separately
                                                                                    from the confusion matrix; d = −ln η; Nosofsky uses
                                                                                    the same η for categorisation and recognition memory
   Segment overlap / rhyme             Baddeley 1966; Nairne & Kelley 1999          Word level: number of shared phonemes, rhyming or
                                                                                    not; no distance within a consonant
```

Two main traditions: **counting features** (the Wickelgren line, most common in memory research) and **estimating similarity from confusion matrices** (the Shepard/Luce line, used in perception and categorisation models). We computed both (`weighted` and `luce`, Spearman .73); they differ mainly in how far apart the voicing-only pairs are placed.

## 3. And GRT?

GRT does not use η; it uses **d′**, the distance between two perceptual distributions.

```
   Luce / Shepard (similarity-choice model)       GRT (Ashby & Townsend 1986)
   ──────────────────────────────────────         ─────────────────────────────────────────────
   Estimate one number η_ab = "how alike"          Each stimulus is a cloud (normal distribution)
   straight from the confusion matrix              in the head
   Says nothing about WHY they are alike           Confusion = overlap of two clouds + where the
   d = −ln η                                       decision bound is drawn
                                                   distance = gap between cloud centres ÷ cloud
                                                   width = d′

        a  ●────── d = −ln η ──────●  b              a  (  ●  )(  ●  )  b
                                                          ↑ the overlapping area is the confusion rate
```

- Luce's η is a *measured* number; GRT's d′ is *derived from a representation*. Ashby & Perrin (1988, *Psychological Review*; reference from memory, not verified) showed that with normal clouds the two agree in many cases — GRT says why things are alike, η does not.
- GRT does one more thing: it **splits a stimulus into dimensions** and tests whether they are independent (perceptual independence) and whether a judgement on one dimension is affected by the other (perceptual / decisional separability). M&N could only say the features "look almost independent" from information transmission; GRT can test it.

```
   GRT line (GRTv3_ada)    already measures d′: colour × sound, two levels each; are the dimensions separable?
   SFT line (SFT_WM)        calibration measures "drift separation" (LNRM's z₂ − z₁), itself a distribution-distance
                            quantity of the d′ family. The Luce distance only ORDERS which consonant pairs go in
                            which level; the real H / L still come from the calibrated drift separation.
```

The most consistent way to say it: **distance between consonants is defined by d′** (GRT's language); the M&N matrices are only used to guess in advance which pairs have large and small d′, which saves calibrating all 120 pairs.

## 4. "Fitting GRT to the M&N matrix" — what that would mean, in plain words

**The M&N matrix**: a 16 × 16 table of "how many times it was misheard". Numbers only, no "why".

**GRT's idea**: in the head a consonant is not a point but a cloud, with a centre and a size.

```
   the "vocal folds" axis in the head ─────────────────────────→
          (   p cloud   )(   b cloud   )           far apart, little overlap → rarely misheard

   the "place in the mouth" axis ────────────────────────────→
          (  p cloud ( t cloud ) k cloud )          crowded, much overlap → often misheard
```

More noise → fatter clouds → more overlap → more errors.

**Fitting = guessing the clouds backwards.** You see "how many times misheard"; you look for cloud positions and sizes whose overlaps reproduce exactly those error rates. Once found, you get:

1. **A d′ for every consonant pair**: how far apart the two centres are, in units of cloud width. That is "distance" in GRT's language, the same kind of quantity the GRT and SFT lines measure.
2. **Whether the five switches are really five separate axes**: if the p and b clouds on the "vocal folds" axis sit at the same position on the "place" axis, the two switches do not interfere (perceptual separability). M&N could only say "looks almost independent"; GRT computes whether it is.
3. **Whether an error comes from overlapping clouds or from a response preference**: GRT separates the clouds from where the decision line is drawn. Luce separates these too, but GRT adds the shape of the clouds.

The output would be: one d′ per feature per SNR, plus an independence test. You could then say "at 0 dB the vocal-folds axis has d′ ≈ 3, the place axis d′ ≈ 1, and the two are independent" — closer to what the experiment measures than "voicing transmitted 95 % of its information".

**Why it is not done now**: a full GRT model for 16 consonants × 5 features has many parameters; one 4,000-judgement matrix supports a simplified version (one axis per feature, equal variances), not the full one. And it would still answer about distance *in noise*, not in memory. A nice extension, not the current bottleneck.
