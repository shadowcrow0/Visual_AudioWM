"""Generate a FIXED table of consonant pairs for the sound dimension: stimuli/sound_pairs.csv.

Why fixed: the experiment should not draw different foils for every participant and run. This script
picks the pairs once, deterministically, from the Luce/Shepard distance (consonant_distance.py, metric
"luce", mean of -12/-6/0 dB), and the experiment scripts just read the table.

    python make_sound_pairs.py            # rewrites stimuli/sound_pairs.csv and prints the table

Design of the table:
  * 5 levels of dissimilarity, cut points from consonant_distance.propose_bins (luce, 5 levels):
        1 = most similar (e.g. p-k, f-theta) ... 5 = most dissimilar (e.g. s-m, sh-n)
  * every one of the 16 consonants appears once as TARGET at every level  -> 16 x 5 = 80 pairs
  * the foil for (target, level) is the consonant whose luce distance to the target is closest to the
    level's midpoint; if the target has no partner inside the level the nearest available foil is used
    and the row is marked snapped = 1 (the actual distance is in the x column)
  * talker: the 12 talkers are assigned round-robin over the 80 rows, so every talker is used 6-7 times
    and no (target, level) shares a talker with its neighbours; target and foil use the SAME talker
  * columns: level, x (luce distance), target, foil, talker, snd_target, snd_foil, feature_distance,
    features_differing, snapped, dprime, dprime_mdsdt
  * dprime = x in d' units: on the 28 single-feature pairs the mdsdt GRT fits of Miller & Nicely at -6 dB give
    d' = 0.911 * luce (Pearson .98, review/GRT擬合MN_B路線_D2.md §8); dprime_mdsdt is that directly fitted value
    where one exists (review/data/grt_subdesigns/pair_dprime_mdsdt.csv), blank otherwise. Re-cutting the levels in
    d' units reproduces the same table (monotone transform), so the pairs are unchanged.
"""
import csv
import os

import numpy as np

import consonant_distance as cd
from adaptive_vawm_EN import MN_FEATURES, TALKERS, feature_distance, sound_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "sound_pairs.csv")
MDSDT_PAIRS = os.path.join(HERE, "review", "data", "grt_subdesigns", "pair_dprime_mdsdt.csv")
LUCE_TO_DPRIME = 0.911   # slope of mdsdt d' (-6 dB) on luce distance, 28 single-feature pairs, intercept 0.003
N_LEVELS = 5
FEATURE_NAMES = ["voicing", "nasality", "affrication", "duration", "place"]


def differing(a, b):
    return "+".join(n for n, x, y in zip(FEATURE_NAMES, MN_FEATURES[a], MN_FEATURES[b]) if x != y)


def mdsdt_dprime(snr=-6):
    """{frozenset({a, b}): mean fitted d'} from the mdsdt 2x2 fits at the given SNR; {} if the file is missing."""
    out = {}
    if not os.path.exists(MDSDT_PAIRS):
        return out
    with open(MDSDT_PAIRS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(float(r["snr"])) == snr:
                out.setdefault(frozenset((r["c1"], r["c2"])), []).append(float(r["dprime"]))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    M = cd.matrix("luce")
    direct = mdsdt_dprime()
    C = cd.CONSONANTS
    bins = cd.propose_bins(M, N_LEVELS)
    edges = bins["edges"]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(N_LEVELS)]
    rows = []
    k = 0
    for level in range(1, N_LEVELS + 1):
        lo, hi = edges[level - 1], edges[level]
        for target in C:
            i = C.index(target)
            cands = [(abs(M[i, j] - mids[level - 1]), M[i, j], C[j]) for j in range(16) if j != i]
            inside = [c for c in cands if lo <= c[1] <= hi] if level < N_LEVELS else [c for c in cands if c[1] >= lo]
            if level == 1:
                inside = [c for c in cands if c[1] <= hi]
            pick = min(inside) if inside else min(cands)
            snapped = 0 if inside else 1
            _, x, foil = pick
            talker = TALKERS[k % len(TALKERS)]
            k += 1
            rows.append(dict(level=level, x=round(float(x), 3), target=target, foil=foil, talker=talker,
                             snd_target=sound_file(talker, target), snd_foil=sound_file(talker, foil),
                             feature_distance=feature_distance(target, foil), features_differing=differing(target, foil),
                             snapped=snapped, dprime=round(LUCE_TO_DPRIME * float(x), 2),
                             dprime_mdsdt=round(direct[frozenset((target, foil))], 2) if frozenset((target, foil)) in direct else ""))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("level edges (luce):", " | ".join(f"{e:.2f}" for e in edges))
    print(f"{'level':>5} {'target':>6} {'foil':>6} {'x':>5} {'dprime':>6} {'mdsdt':>5} {'feat':>4}  features differing          talker  snapped")
    for r in rows:
        print(f"{r['level']:>5} {cd.IPA[r['target']]:>6} {cd.IPA[r['foil']]:>6} {r['x']:>5.2f} {r['dprime']:>5.2f} {str(r['dprime_mdsdt']):>5} {r['feature_distance']:>4}  "
              f"{r['features_differing']:<26} {r['talker']:>6}  {'*' if r['snapped'] else ''}")
    print(f"wrote {OUT}: {len(rows)} pairs, {sum(r['snapped'] for r in rows)} snapped")


if __name__ == "__main__":
    main()
