# Draft text for the paper: the consonant dissimilarity scale (Methods, Supplement, Results, Limitations)

2026-11-12. Follows `論文規格書_聲音軸.md`. Numbers come from the files named in brackets. `[TODO: …]` marks items the
authors must supply. Figures: `figures/figS1_dprime_vs_luce.png`, `figures/figS2_separability_dAIC.png`, `figures/figM1_feature_merge.png`;
tables: `figures/tableS1_dprime_matrix.md`, `figures/tableS2_levels_to_pairs.md`.

---

## Methods

### Stimuli: speech sounds

The auditory items were the 16 English consonants studied by Miller and Nicely (1955): /p t k f θ s ʃ b d g v ð z ʒ m n/.
Each consonant was presented as a consonant–vowel syllable spoken by one of 12 talkers (T01–T12), giving 192 tokens
(16 consonants × 12 talkers). [TODO: corpus or recording source, licence, talker sex/dialect, and the vowel.] Files were
mono, 16-bit, 15 kHz, 560–680 ms long. Whole-file RMS was closely matched across consonants (range 2.7 dB) but not
across talkers: tokens of T10–T12 were 5–8 dB quieter than those of the other talkers, and the consonant portion
(first 80 ms after acoustic onset) of /θ/ and /f/ was 20–25 dB below that of the nasals, as expected for the weakest
English consonants (`review/data/wav_levels.csv`). [TODO: headphones model; presentation level in dB SPL and how it was
set.] Audio output was routed to the headphone device selected by name (`audio_pick.py`), never by device index.

Within a trial the two remembered sounds were different consonants spoken by different talkers; the probe sound always
used the talker of the item it probed, so talker identity never cued the answer.

### A perceptual scale for consonant dissimilarity

Systems factorial technology requires two salience levels per dimension, calibrated for each participant. For colour
we used CIELAB ΔE. For speech sounds no comparable unit was available, so we constructed one from the consonant
confusion matrices of Miller and Nicely (1955; Tables I–VI, white noise at S/N −18 to +12 dB, 200–6500 Hz band, CV
syllables with /ɑ/). The six 16 × 16 tables were transcribed from the published scans and checked against the row
totals of ≈ 250 (`review/data/miller_nicely_1955_tables_I-VI.json`). [TODO: second independent transcription check.]

*Pairwise distance.* For each S/N level we fitted Luce's (1963) similarity-choice model, P(j | i) = b_j η_ij / Σ_k b_k η_ik,
with 120 free similarity parameters η_ij (η_ii = 1) and 15 free response biases b_j, by maximum likelihood
(closed-form gradient ascent started from Shepard's (1957) closed-form estimate; `consonant_distance.py`). Distance was
defined as −ln η_ij (Shepard, 1987) averaged over the −12, −6 and 0 dB tables; the three higher S/N tables were not
used because at 0, +6 and +12 dB respectively 73, 80 and 92 of the 120 pairs were never confused in either
direction, leaving the estimate to the smoothing constant. Averaging over −6/0/+6 dB instead changes the rank order
of the 120 pairs by Spearman ρ = .97 relative to the chosen set; using +12 dB alone gives ρ = .58.

*Units.* To express the Luce distances in signal-detection units we fitted general recognition theory (GRT; Ashby &
Townsend, 1986) models to the same data. GRT represents each stimulus as a bivariate Gaussian perceptual distribution
and tests perceptual separability (PS: the distribution along one dimension does not move with the level of the other)
and perceptual independence (PI: within a stimulus the perceptual noise on the two dimensions is uncorrelated). The
available fitting software, mdsdt (Silbert & Hawkins, 2016), handles full-report identification designs with two
dimensions and the same number of levels on each, typically a 2 × 2 design and its 4 × 4 confusion matrix. The 16
consonants lie on four feature axes (Table M1), so sub-designs of two features were cut out with the other features held
fixed.

*From five features to four axes (Fig. M1).* Miller and Nicely described the 16 consonants with five articulatory
features (pp. 346–347, Table XIX): **voicing** — whether the vocal cords vibrate (/b d g v ð z ʒ m n/ vs /p t k f θ s ʃ/);
**nasality** — the velum lowered and air released through the nose (/m n/ only); **affrication** — the articulators
brought close enough to produce turbulent friction noise (/f θ s ʃ v ð z ʒ/ vs the stops and nasals); **duration** — a
name they "arbitrarily adopted" for the difference between /s ʃ z ʒ/, "long, intense, high-frequency noises", and the
other 12 consonants; and **place** — front (/p b f v m/), middle (/t d θ ð s z n/) and back (/k g ʃ ʒ/). They treated
each feature as a communication channel and computed information transmission per channel, noting explicitly that the
inputs to the five channels are not independent, so that "even if the channels themselves are independent, the amounts
transmitted through each channel will be related" (p. 348): the 16 consonants do not fill all combinations of the five
features.

This matters for GRT. Perceptual separability asks whether the perceptual distribution of feature A is the same at every
level of feature B; to answer it, every level of A must occur at every level of B — A × B must be a **full factorial
design**, with at least one stimulus in every cell. Affrication × duration fails this test (Fig. M1A): cell (0,0) holds
the eight stops and nasals, (1,0) holds /f θ v ð/, (1,1) holds /s ʃ z ʒ/, but (0,1) — "long but not fricated" — contains
**no English consonant**. Three of four cells are filled; duration is defined only for consonants with affrication = 1;
the two factors are **nested**, not crossed. Nested factors cannot be tested for separability against each other and
cannot each carry an independent d′ axis.

Three treatments were considered: (a) keep five features and write a five-dimensional likelihood ourselves — feasible,
but it abandons validated fitting software and the (0,1) cell stays empty, so the separability test is still impossible;
(b) drop duration — this merges /s/ with /θ/ and /ʃ/ with /θ/, whose discriminabilities differ greatly (after fitting,
d′ = 0.97 for θ–s vs 0.46 for f–θ); (c) merge the two features into one **three-level ordinal manner axis**: 0 = closure
(stops and nasals), 1 = short friction (/f θ v ð/), 2 = long, intense friction (/s ʃ z ʒ/) (Fig. M1B). Option (c) loses
no information — the three occupied cells map one-to-one onto three levels and the empty cell never needs a code — its
order has acoustic meaning (increasing friction and duration), and it matches the three-way manner/openness coding
customary in the speech short-term-memory literature (Wickelgren, 1966). We adopted (c).

Four axes remain: voicing (2), nasality (2), manner (3), place (3). Fig. M1C lays the 16 consonants out on
voicing × manner × place with the nasals apart: cells are still empty (no front sibilant, no back short fricative, nasals
only at front and middle), so **the four axes together are still not a full factorial** and cannot be handed to any model
that requires one in more than two dimensions. What can be done is to find, in that layout, the rectangles in which every
cell holds exactly one consonant — the sub-designs below.

**Table M1. Coding of the 16 consonants on the four axes**

| | voicing | nasality | manner | place |
|---|---|---|---|---|
| levels | 0 voiceless / 1 voiced | 0 oral / 1 nasal | 0 stop or nasal / 1 short fricative / 2 sibilant | 0 front / 1 middle / 2 back |
| p t k | 0 | 0 | 0 | 0 1 2 |
| f θ | 0 | 0 | 1 | 0 1 |
| s ʃ | 0 | 0 | 2 | 1 2 |
| b d g | 1 | 0 | 0 | 0 1 2 |
| v ð | 1 | 0 | 1 | 0 1 |
| z ʒ | 1 | 0 | 2 | 1 2 |
| m n | 1 | 1 | 0 | 0 1 |

*Enumeration rule* (`grt_mn_subdesigns.py`). Choose two axes f and g; fix the other two at one combination of values
(the context); within that context find level subsets L_f and L_g such that every cell of L_f × L_g holds exactly one
consonant (a full factorial); keep only maximal subsets. One context can hold two maximal subsets (both manner × place
contexts do). This gives 11 sub-designs (Table M2). No voicing × nasality or nasality × manner design exists because
all nasals are voiced and all are manner 0.

**Table M2. The 11 feature-pair sub-designs** (−6 dB; retained = share of each consonant's responses that fall inside
the subset).

| f × g | context | consonants | size | retained |
|---|---|---|---|---|
| voicing × manner | oral, front | p b f v | 2 × 2 | .66 |
| voicing × manner | oral, middle | t d θ ð s z | 2 × 3 | .50 |
| voicing × manner | oral, back | k g ʃ ʒ | 2 × 2 (manner 0, 2) | .57 |
| voicing × place | oral, stops | p b t d k g | 2 × 3 | .72 |
| voicing × place | oral, short fricatives | f v θ ð | 2 × 2 | .69 |
| voicing × place | oral, sibilants | s z ʃ ʒ | 2 × 2 (place 1, 2) | .72 |
| nasality × place | voiced, manner 0 | b m d n | 2 × 2 | .73 |
| manner × place | voiceless, oral | p f t θ | 2 × 2 (manner 0, 1) | .74 |
| manner × place | voiceless, oral | t s k ʃ | 2 × 2 (manner 0, 2; place 1, 2) | .72 |
| manner × place | voiced, oral | b v d ð | 2 × 2 (manner 0, 1) | .77 |
| manner × place | voiced, oral | d z g ʒ | 2 × 2 (manner 0, 2; place 1, 2) | .84 |

*Why a subset is admissible.* Each sub-matrix keeps only the cells whose stimulus and response both lie in the subset,
i.e. the identification data are conditioned on the response falling among these four (or six) candidates. Under the
GRT decision model, if the response criteria are not changed by our later selection of a subset — which decisional
separability guarantees — the conditioned 4 × 4 matrix is the data the sub-design itself would have produced, with fewer
trials. The cost is that d′ now means discriminability among these candidates, and that low retention (.45 at −12 dB)
degrades the estimates; hence the main conclusions use −6 dB.

**Table M3. How the two 2 × 3 designs were split into adjacent 2 × 2 designs.**

| original | 2 × 3 consonants | split along | sub-design (a) | sub-design (b) |
|---|---|---|---|---|
| voicing × manner, middle | t d / θ ð / s z | adjacent manner levels | manner 0 ∣ 1: t d θ ð | manner 1 ∣ 2: θ ð s z |
| voicing × place, stops | p b / t d / k g | adjacent place levels | place 0 ∣ 1: p b t d | place 1 ∣ 2: t d k g |

Each 2 × 2 sub-table is conditioned once more on its own four consonants (e.g. t d θ ð at −6 dB retains 353 of the 737
responses of the 2 × 3 design).

*Why the split is admissible.* (1) Software: mdsdt's n × n routine takes √(number of stimuli) as the number of levels per
axis (`create_n_by_n_mod`), so six stimuli cannot be fitted. (2) The quantity needed is the d′ between adjacent levels —
t–θ and θ–s separately, not three points on one line — and the PS question is likewise asked step by step (does the
voicing distribution move when manner goes from 0 to 1, and from 1 to 2?). (3) The split is just one more conditioning,
with the same logic as above; what is lost is the information whether the three distributions are collinear and whether
the three levels share one set of criteria. The empirical check is that a pair estimated in different sub-designs should
give similar d′: of the 28 single-feature pairs, 21 recur across designs; 16 agree within 0.4, and the largest
discrepancies are /θ–ð/ (2.48–3.13), /d–ð/ (1.40–2.04), /b–d/ (2.16–2.78), /k–ʃ/ (3.12–3.59) and /t–d/ (3.94–4.35),
at most 0.65; no pair changes rank.

Each 4 × 4 sub-matrix was fitted with mdsdt 1.3 (R [TODO: version]) under the equal-variance Gaussian GRT model with
decisional separability, in seven versions: unconstrained (full), PS on axis f, PS on axis g, PS on both, one shared
correlation ρ, ρ = 0 (PI), and PS on both with ρ = 0. Models were read with ΔAIC = AIC(constrained) − AIC(full): ≤ 2
taken as the constraint holding, > 10 as rejected. The −6 dB table was used because higher S/N tables were too sparse
for stable fits.

The fits gave a d′ for each of 28 single-feature pairs (the mean separation along the fitted axis; of the 21 pairs recurring in
more than one sub-design, 16 agreed within 0.4, the largest discrepancy being 0.65). Across these 28 pairs the GRT d′ was a linear function of
the Luce distance through the origin, d′ = 0.003 + 0.911 (−ln η), Pearson r = .98 (Fig. S1), the relation expected
between η and d′ under equal-variance Gaussian assumptions. We therefore defined the dissimilarity of every one of the
120 pairs as d′ = 0.911 × (−ln η) (Table S1; `stimuli/consonant_dprime.csv`). The scale runs from 0.26 (/p–k/) to 4.95.

*How we got here: from counting features to a per-pair table.* GRT was not the starting point. The first sound axis
was the log of Miller and Nicely's confusion counts summed over three S/N levels; in the first pilot the sound dimension
sat at chance under memory load, which sent us back to the axis. Miller and Nicely (1955) analysed their matrices feature
by feature with information transmission and concluded that the five features "were perceived almost independently of
one another" (p. 348): voicing and nasality remained discriminable at −12 dB whereas place was hard to distinguish below
+6 dB, a difference of some 18 dB in efficiency (p. 349; Table XXI, relative transmission at −12 dB: voicing 52 %,
nasality 49 %, place 4 %). The picture of features as independent channels invites a feature-distance scale: two
consonants are as dissimilar as the number of features on which they differ. The speech short-term-memory literature
speaks the same language: Wickelgren (1966) predicted recall confusions from voicing, nasality, manner (openness) and
place and concluded that consonants are stored as feature bundles whose features are forgotten semi-independently;
Klatt (1968) re-analysed the same data with a feature set containing voicing but not place.

The same literature warns against *counting* features. Schweppe, Grice and Rummer (2011) manipulated which feature is
shared, not how many: a set sharing only manner (/pa ta ka/) impaired serial recall in every presentation and recall
mode, a set sharing only place (/da la za/) only with spoken recall, and the distinct set, which differed additionally
in place, was not recalled better — "as might be expected if a simple count of differences across feature values were
involved" (p. 264); they cite Brady, Mann and Schmidt (1987), who found no consistent relation between recall and sharing
0, 1 or 2 features. Roodenrys et al. (2025) used only a binary "differs by exactly one articulatory feature" definition,
36 of their 60 interfering pairs were voicing pairs, and the three feature types were never analysed separately. Cutler
et al. (2004) showed that feature weights follow the listener's first language: Dutch listeners transmitted voicing
worst, but only in the syllable-final position where Dutch has no voicing contrast. Taken together: which feature
differs matters; how many differ is not a scale. Our own data pointed the same way: the old axis placed the voicing-only
pairs /p–b/ and /k–g/ at the "least similar" end (white noise does not mask voicing), yet a non-native participant's
d′ for these pairs in a pure perception test was only 2.3, no better than for other pairs.

What was needed, then, was a scale that presupposes no feature weights and estimates the discriminability of every pair
from human identification data. There are two mature ways to turn a confusion matrix into distances, with
complementary strengths:

| | Luce's similarity-choice model (Luce, 1963; Shepard, 1957) | General recognition theory (Ashby & Townsend, 1986) |
|---|---|---|
| Model | P(j∣i) = b_j η_ij / Σ_k b_k η_ik: symmetric similarity η_ij, response bias b_j | one multivariate Gaussian perceptual distribution per stimulus; decision regions partition the space; d′ = mean separation / SD |
| Structure assumed | none: η are free parameters, no dimensions | explicit perceptual dimensions; PS, PI and DS are testable |
| Handles at once | the whole 16 × 16 (135 parameters) | a factorial design with two dimensions and equal numbers of levels (here: sub-designs) |
| Yields | distances −ln η for 120 pairs (Shepard, 1987) | d′ for single-feature pairs (28) plus which features are separable / independent |
| Lacks | a unit: −ln η is on an arbitrary scale | coverage: no direct value for pairs differing on several features |
| Shared | both absorb response bias; both are estimated from the same matrix | |

They are complements, not rivals: Luce gives coverage, GRT gives the unit and the structure. They can be joined because
under equal-variance Gaussian assumptions they describe the same thing — the similarity-choice model is the logistic
approximation to signal detection theory (Luce, 1963; Shepard, 1957; Nosofsky, 1986, linking η_ij = exp(−c·d_ij) to
multidimensional-scaling distance). For two-alternative identification the Gaussian model gives P(correct) = Φ(d′/2)
and the choice model 1/(1 + η); with Φ(x) ≈ logistic(1.7x) this yields −ln η ≈ 0.85·d′, i.e. −ln η should be a line
through the origin in d′. On the 28 single-feature pairs we measured d′ = 0.911·(−ln η) (r = .98, Fig. S1), a slope of
the same order as the theoretical approximation (1.18) but not identical to it — the GRT d′ uses −6 dB only, the Luce
distance averages three S/N levels, and the sub-matrices are conditioned — so the empirical slope was used rather than
the theoretical constant.

*What the GRT fits told us.* Beyond the unit, they answer whether a feature can serve as an axis. Voicing and nasality
were perceptually separable from the other features (ΔAIC of the separable model ≤ 0 in every context), and voicing was
perceptually independent of place and manner at −6 dB. Place and manner were not separable: the d′ for place among
voiceless stops was 0.46 (/p–t/) but 2.5 among voiced stops (/b–d/) — Miller and Nicely themselves noted that the
aspiration noise of voiceless stops masks the formant transitions that signal place (p. 347) — and the d′ for manner was
3.4 for /k–ʃ/ but 1.1 for /g–ʒ/ (Fig. S2). A difference of "one feature" thus spans a fivefold range of discriminability;
the stimulus space is not a product of feature axes. This is why dissimilarity must be specified per pair, which is
exactly what the Luce–GRT d′ table is; it also explains why Schweppe et al. found a manner–place asymmetry and why
Roodenrys et al.'s voicing pairs behaved differently from the rest — none of which a count can express.

*Levels.* Calibration used six nominal levels, d′ = 0.4, 0.9, 1.5, 2.3, 3.3 and 4.5. On each trial the probe
consonant was the foil of the remembered target whose d′ to it was closest to the nominal level, and the realised d′
was recorded and used in the fit (Table S2). The lowest two levels are place contrasts, the middle levels manner
contrasts, and the highest level voicing contrasts.

### Calibration

Both dimensions were calibrated with the task itself (two colours and two sounds to remember; a colour–sound probe;
"did both appear?"). While one dimension was calibrated the two targets of the other dimension stayed fixed for the
block, and the probe's value on that dimension was drawn from its target, a provisional high level (ΔE 30; d′ 4.3) or a
provisional low level (ΔE 6; d′ 1.0). Each dimension received 10 practice trials with feedback, then 30 non-match
trials (6 levels × 5), 10 match trials and 8 filler trials, in random order without feedback.

Accuracy and response time were fitted with a two-accumulator log-normal race (LNRM; [TODO: adaptiveSFT citation]) in
which the drift separation between the "different" and "same" accumulators is a quadratic function of the stimulus
difference, v(x) = α x + α₂ x². Models were fitted in PyMC 5.28 with the nutpie sampler (4 chains, 1000 warm-up and
1000 kept draws each; `adaptivesft.models_v2.fit_lnrm_v2`). Colour entered the model as ΔE/10 and sound as d′. The
high (H) and low (L) levels were the stimulus differences at which the posterior drift separation equalled 8.0 and
1.3, solved draw by draw (smaller root; draws with α₂ ≥ 0 excluded) and averaged. When a target lay beyond the
presentable range the level was set to the range limit and flagged. [TODO: final rule for H once ≥ 5 participants are
available; see Results.]

---

## Supplement

**S1. Transcription of Miller & Nicely (1955).** Source: Tables I–VI (S/N −18, −12, −6, 0, +6, +12 dB). Row sums
3997–4004 over 16 rows per table (≈ 250 per consonant). Proportion correct .078, .270, .465, .705, .834, .908. Pairs
never confused in either direction: 0, 21, 88, 163, 175, 199 of 240 ordered cells.

**S2. Luce model fits.** Per S/N level: log-likelihood of the fitted and the saturated model
[TODO: paste from `consonant_distance.luce_result(snr)`]. η values above 1 (a pair confused more often than the diagonal
predicts; a few at −18 dB) were clipped to 1 before taking −ln η.

**S3. GRT sub-designs and fits.** List of the 11 designs with consonants, context and retained proportion
(`review/data/grt_subdesigns/index.csv`). Models: full; PS on axis f; PS on axis g; PS on both; PI with one shared ρ;
PI with ρ = 0; PS on both with ρ = 0. 392 fits attempted, 336 converged; failures (non-positive-definite covariance)
were almost all PS or PI models at 0 and +6 dB. The 2 × 2 routine of mdsdt fixes the perceptual variances at 1 and adds
1 to every cell. Note: mdsdt 1.3's `mriTest` uses the wrong denominator in its second-axis statistic
(`grt_base.R`, `mriTest`), so marginal response invariance on that axis was judged from ΔAIC, not from that test.
Fig. S2 shows ΔAIC (PS model − full) for every design at −6 dB.

**S4. Per-pair d′ from GRT.** 28 pairs, −6 dB, with the context in which each was estimated and the range across
contexts (`review/data/grt_subdesigns/pair_dprime_mdsdt.csv`).

**S5. Scale table.** Table S1 (16 × 16). Table S2 (foil chosen at each nominal level for each target).

---

## Results (pilot; n = 1 at the time of writing — [TODO: replace with the full sample])

*Perception without memory load.* One Mandarin-L1 participant judged 80 different and 40 identical pairs (same talker,
fixed table `stimuli/sound_pairs.csv`). Hit rate by distance quintile was .81, .75, .81, .88 and .94 (false-alarm rate
.17), but trial-level accuracy was unrelated to d′ (point-biserial r = .06, p = .61) and pairs differing only in place
(mean d′ 1.5) were identified as well as voicing pairs (.89 each). With clean speech the scale is thus at ceiling in
perception, as expected; its dynamic range has to come from memory load.

*Calibration under load.* For the same participant the LNRM located the low level of the sound dimension at
d′ = 1.57 (accuracy .60–1.00 across levels; median RT falling from 1.7 s at the lowest levels to 1.0 s at the highest)
and of the colour dimension at ΔE 39 (accuracy at chance for ΔE ≤ 12, 1.0 for ΔE ≥ 25). The high level could not be
reached in either dimension: the fitted drift function peaked below a separation of 8.0, the solution for H existed in
only 1–3 % of posterior draws, and H was set to the top of the presentable range (ΔE 90; d′ 4.5). In an earlier pilot
with the previous confusion-count scale, a different trial structure and a key-logging fault, the same participant's
sound accuracy had been at chance at all levels; the three changes cannot be separated with these data.

---

## Discussion: limitations of the scale

The scale is derived from native listeners' identification in white noise and applied to clean speech heard by
non-native listeners; only its ordering is assumed to transfer, and even that is a working assumption (Cutler et al.,
2004, report non-native accuracy at about 80 % of native across S/N levels, with per-phoneme accuracies correlated
.83–.91 between groups and one L1-specific effect, final-position voicing). The GRT sub-matrices condition on responses
inside each subset; the 2 × 2 fits fix variances and assume decisional separability; correlations at the higher S/N
levels are not interpretable. Perceptual distance is not memory distance: Schweppe, Grice and Rummer (2011) found
that the set hardest to recall (/pa ta ka/) is the set most confused in noise, whereas Roodenrys et al. (2025) found
interference from articulatorily similar consonants under visual presentation, where acoustic confusability cannot
operate. The present scale is therefore a prior for the adaptive calibration, not a claim about memory; its adequacy is
tested by whether accuracy and response time vary with it under load. Finally, the drift-separation target for the high
level (8.0) lay beyond the fitted function for the participant tested so far, so H was the highest presentable level;
this leaves H un-equated across participants in drift units, a departure from the intended design that we will
[TODO: keep / revise] once the full sample is calibrated.

---

## References to add

Ashby & Townsend (1986) Psychol Rev 93, 154. · Cutler, Weber, Smits & Cooper (2004) JASA 116, 3668. · Houpt, Blaha,
McIntire, Havig & Townsend (2014) Behav Res Methods 46, 307. · Luce (1963) in Handbook of Mathematical Psychology. ·
Miller & Nicely (1955) JASA 27, 338. · Roodenrys, Miller, Guitard & Neath (2025) Memory 33, 248. · Schweppe, Grice &
Rummer (2011) JML 64, 256. · Shepard (1957) Psychometrika 22, 325; (1987) Science 237, 1317. · Nosofsky (1986) JEP:General 115, 39. · Brady, Mann & Schmidt (1987) Mem Cognit 15, 444. · Klatt (1968) JASA 44, 401. · Silbert & Hawkins
(2016) J Math Psychol 73, 110 (mdsdt). · Thomas (2001) Percept Psychophys 63, 625 (MRI test). · Wickelgren (1966)
JASA 39, 388. · [TODO: adaptiveSFT / LNRM citation; PyMC; nutpie.]
