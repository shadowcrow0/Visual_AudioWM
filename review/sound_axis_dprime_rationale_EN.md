# Why the sound dimension is scaled in d′ (note for the PI)

2026-11-10. Audience: a mathematical psychologist who works on GRT. Sources: `GRT擬合MN_B路線_D1.md` / `_D2.md`,
`pilot_2026-11-10_*.md`, `Cutler2004_全文查核.md`, `Schweppe2011_全文查核.md`, `Roodenrys2025_全文查核.md`.

**Claim.** The stimulus scale x for the sound dimension is d′, obtained from a GRT fit to the Miller & Nicely (1955)
confusion matrices. It plays exactly the role ΔE plays for colour: it is only the independent variable of the drift
function v(x) in the LNRM calibration. H and L are then defined in drift-separation units and calibrated per participant.

## How it was computed

```
   M&N 16×16 × 4 SNR (−12/−6/0/+6 dB)
     │ hold the other features fixed → 11 feature-pair sub-designs (2×2, or 2×3 split into two 2×2);
     │ responses conditioned on the subset
     v
   mdsdt::fit.grt: equal-variance Gaussian, DS assumed; full / PS_x / PS_y / PI models compared by ΔAIC
     │
     ├─ separability: the voicing and nasality means do not move with the other axis (ΔAIC ≤ 0);
     │   place moves with voicing (p–t 0.46 vs b–d 2.5 at −6 dB); manner moves with voicing (k–ʃ 3.4 vs g–ʒ 1.1)
     │   → the stimulus space is not a product of feature axes; "number of differing features" is not a scale
     └─ per-pair d′ = mean separation along that axis at −6 dB; 28 single-feature pairs;
        pairs that recur across contexts agree within 0.4 (d–ð the exception)

   All 120 pairs: −ln η from Luce's similarity-choice model (ML fit, mean of −12/−6/0 dB)
   is a straight line through the origin against those 28 GRT d′ values: d′ = 0.911·(−ln η), r = .98
   → 0.911·(−ln η) is used as d′ for all 120 pairs (stimuli/consonant_dprime.csv)
```

He will recognise the monotone relation between η and d′ under equal-variance Gaussian assumptions as the old
Shepard–Luce result; the .98 only confirms it holds on this table, and the GRT sub-designs pin the unit to d′.

## The point that matters for a GRT reader

We are not claiming the sound dimension is perceptually separable. The fits say the opposite: only voicing and
nasality are separable; place and manner are not. Collapsing sound to a scalar x is therefore a convenience: x only
has to be monotone with interpretable spacing. Selective influence is tested in the SFT data (the SIC selective-influence
check), not assumed from the scale.

## Assumptions and limits, stated before he asks

1. M&N: white noise, CV with /ɑ/, five native speakers. We use clean speech and Mandarin-L1 listeners. The ordering is
   assumed to transfer; the absolute values are not (Cutler et al. 2004: non-native ≈ 80% of native accuracy, per-phoneme
   correlation .83–.91 between groups).
2. Sub-matrices are conditioned on responses inside the subset (retention .55–.80 at −6 dB). 0/+6 dB are sparse and
   held up by +1 smoothing; ρ is not trustworthy there, so −6 dB is used.
3. mdsdt's 2×2 path fixes σ = 1, so "separable" means "means do not move". Its `mriTest` has a bug in the y-axis
   formula (version 1.3); only ΔAIC was used.
4. Perception ≠ memory: Schweppe et al. (2011) point the same way (pa-ta-ka is the hardest set in both), Roodenrys et
   al. (2025) the other way (articulatory similarity interferes under visual presentation too). The scale's validity
   therefore rests on whether it produces a slope under load.
5. Data: one participant. In pure perception accuracy is flat across x (r = .06) — the clean-speech ceiling, as expected.
   Under load (11-10) there was a slope for the first time: L = 1.57; H could not reach a drift separation of 8.0 and
   was clipped at 4.5.

## Alternatives he may raise

- *Measure each participant's own 16×16 and fit GRT per person.* ≥ 2000 trials per person; out of budget. M&N as the
  prior plus adaptive per-person calibration of L is the compromise.
- *Use an SNR continuum.* SNR manipulates signal audibility, not the representational distance between two remembered
  items; we want the latter. If the d′ scale produces no slope under load, SNR is the fallback.
- *Why can H not be reached?* v(x) is quadratic with α₂ < 0, so it has a maximum; 8.0 lies above this participant's
  peak. The inverted H is an extrapolation from the ~3% of posterior draws that have a solution, hence unstable. For now
  only the message is changed, not the rule; the rule is decided after more participants.
