"""Build the ONLINE calibration pilot (online/calib_pilot/): 216 different + 108 same consonant pairs at ONE SNR,
spread evenly along the acoustic axis (spectrogram difference in our noise), every syllable pre-mixed.

    python analyze/acoustic_scale_compute.py             # once: writes analyze/out/acoustic_scale_pairs.csv (about 25 s)
    python online/make_calib_pilot_stimuli.py            # writes calib_pilot/stim/*.wav, conditions.json, design.csv, design.png
    python online/pack_stimuli.py calib_pilot            # stim/ -> bundles/

Why a second pilot instead of changing online/snr_pilot/: that one produced WEB01's data and must stay rebuildable.
What is different here (review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md §7, step 1):
- One SNR (+6 dB). A single 72-trial block per SNR could not be calibrated for either listener; ~216 trials at one SNR can.
- The unit is pair x talker and its x is the acoustic difference of THAT talker's two files at this SNR
  (analyze/acoustic_scale_compute.py, columns ex_T01..ex_T12), not the d' table. Trials fill 1-dB bins from 0.5 to 8.5 dB
  as evenly as the material allows, so the slope AND the ceiling are sampled.
- The seven voicing-only pairs (b-p, d-t, g-k, f-v, s-z, th-dh, sh-zh) are forced in, both orders, one per voice group:
  the d' table calls them easy, the acoustic scale calls them hard, and no listener has been tested on them in noise.
- The 12 talker labels are three near-identical voices (T01-T06, T07-T09, T10-T12; analyze/stimuli_voice_check.py),
  so sampling is balanced over the three GROUPS and the group is logged.
- Within each bin the pair types (stop/fricative/nasal, within or across class) are spread as far as the material allows,
  so that "large difference" is not just "contains sh" and "small" is not just "b v k p". design.csv shows what was achieved.
- Seeds are written as STRINGS: a browser stores 64-bit integers as doubles and rounds them.

The plan is fixed by PLAN_SEED (same for every participant); the browser shuffles the order per participant.
"""
import csv
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
import sft_noise  # noqa: E402
import make_snr_pilot_stimuli as snr_pilot  # noqa: E402   (headphone-check tones: same generator)
from adaptive_vawm_EN import TALKERS, audio_x, feature_distance, load_dprime, sound_file  # noqa: E402

SNR = 6.0
N_DIFF, N_SAME = 216, 108
EDGES = np.arange(0.5, 9.0, 1.0)                 # 8 bins: 0.5-1.5, ..., 7.5-8.5 dB
PLAN_SEED = 20261125
OUT = os.path.join(HERE, "calib_pilot")
STIM = os.path.join(OUT, "stim")
PAIRS_CSV = os.path.join(ROOT, "analyze", "out", "acoustic_scale_pairs.csv")

GROUPS = {"G1": TALKERS[0:6], "G2": TALKERS[6:9], "G3": TALKERS[9:12]}       # analyze/stimuli_voice_check.py
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
CLASS = {**{c: "S" for c in "ptkbdg"}, **{c: "F" for c in ("f", "theta", "s", "sh", "v", "eth", "z", "zh")}, "m": "N", "n": "N"}
VOICING = [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("theta", "eth"), ("s", "z"), ("sh", "zh")]
pair_key = lambda a, b: "-".join(sorted([a, b]))
pair_type = lambda a, b: "".join(sorted([CLASS[a], CLASS[b]], key="SFN".index))        # SS, SF, SN, FF, FN, NN


def load_acoustic():
    """{(pair, talker): x} and {pair: 12-talker mean} at SNR, from analyze/acoustic_scale_compute.py."""
    if not os.path.exists(PAIRS_CSV):
        raise SystemExit(f"missing {PAIRS_CSV}: run  python analyze/acoustic_scale_compute.py  first")
    x, mean = {}, {}
    with open(PAIRS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["snr"] in ("clean",) or float(r["snr"]) != SNR:
                continue
            p = pair_key(r["a"], r["b"])
            mean[p] = float(r["excess"])
            for t in TALKERS:
                x[(p, t)] = float(r[f"ex_{t}"])
    if len(mean) != 120:
        raise SystemExit(f"{PAIRS_CSV}: expected 120 pairs at {SNR:g} dB, found {len(mean)}")
    return x, mean


def bin_of(v):
    return int(np.clip(np.searchsorted(EDGES, v, side="right") - 1, 0, len(EDGES) - 2))


def make_plan(rng, X, XMEAN, table):
    dp = {(c, f): d for c, lst in table.items() for f, d in lst}
    nb = len(EDGES) - 1
    use_pair, use_cons, use_talker, use_unit = {}, {}, {}, {}
    first = {}                                         # per pair: how often its alphabetically first consonant was heard first
    bins = [[] for _ in range(nb)]

    def add(p, t, forced_order=None, voicing=False):
        a, b = p.split("-")
        if forced_order is None:
            k = first.get(p, 0)
            forced_order = (a, b) if k <= 0 else (b, a)          # alternate the order within a pair
        first[p] = first.get(p, 0) + (1 if forced_order[0] == a else -1)
        for d, key in ((use_pair, p), (use_talker, t), (use_unit, (p, t)), (use_cons, a), (use_cons, b)):
            d[key] = d.get(key, 0) + 1
        bins[bin_of(X[(p, t)])].append(dict(consonant=forced_order[0], foil=forced_order[1], talker=t, voicing_only=int(voicing)))

    # 1. the voicing-only pairs: one talker per voice group, both orders -> 6 trials per pair
    vkeys = [pair_key(a, b) for a, b in VOICING]
    for p in vkeys:
        a, b = p.split("-")
        for g, ts in GROUPS.items():
            t = min(ts, key=lambda u: (use_talker.get(u, 0), rng.random()))
            add(p, t, (a, b), True); add(p, t, (b, a), True)

    # 2. fill the bins to the same height (water-filling: the voicing trials already sit in some of them)
    have = np.array([len(b) for b in bins]); left = N_DIFF - have.sum(); target = have.copy()
    while left > 0:
        i = int(np.argmin(target)); target[i] += 1; left -= 1
    cands = [(p, t) for (p, t) in X if p not in vkeys]
    for bi in np.argsort(-(target - have)):            # the emptiest bins first: they constrain the usage counts least
        pool = [(p, t) for (p, t) in cands if bin_of(X[(p, t)]) == bi and EDGES[0] <= X[(p, t)] < EDGES[-1]]
        while len(bins[bi]) < target[bi]:
            if not pool:
                raise SystemExit(f"bin {bi} ran out of pair x talker units")
            here = bins[bi]
            n_type = lambda ty: sum(pair_type(r["consonant"], r["foil"]) == ty for r in here)
            n_group = lambda g: sum(GROUP_OF[r["talker"]] == g for r in here)
            n_pair = lambda p: sum(pair_key(r["consonant"], r["foil"]) == p for r in here)

            def cost(u):
                p, t = u; a, b = p.split("-")
                return (use_unit.get(u, 0) * 100               # the same two files again only as a last resort
                        + 6 * n_pair(p) + 4 * n_type(pair_type(a, b)) + 3 * n_group(GROUP_OF[t])
                        + 2 * use_pair.get(p, 0) + max(use_cons.get(a, 0), use_cons.get(b, 0)) / 4 + use_talker.get(t, 0) / 6
                        + rng.random() * 0.5)
            u = min(pool, key=cost)
            if use_unit.get(u, 0) >= 2:
                pool.remove(u); continue                         # both orders of this unit are used
            add(*u)
    diff = [r for b in bins for r in b]

    # 3. same trials: every consonant twice in every voice group (96) + 12 more, 4 per group, on 12 different consonants
    cons = sorted(table)
    same = [(c, g) for c in cons for g in GROUPS for _ in range(2)]
    extra = list(rng.permutation(cons))[:12]
    same += [(c, list(GROUPS)[i % 3]) for i, c in enumerate(extra)]
    for c, g in same[:N_SAME]:
        t = min(GROUPS[g], key=lambda u: (use_talker.get(u, 0), rng.random()))
        use_talker[t] = use_talker.get(t, 0) + 1
        diff.append(dict(consonant=c, foil=c, talker=t, voicing_only=0, is_same=1))

    order = rng.permutation(len(diff))
    rows = []
    for i, k in enumerate(order, 1):
        r = diff[k]; c, f, t = r["consonant"], r["foil"], r["talker"]; same_ = int(r.get("is_same", 0)); p = pair_key(c, f)
        x = None if same_ else round(X[(p, t)], 4)
        rows.append(dict(row=i, is_same=same_, consonant=c, foil=f, talker=t, group=GROUP_OF[t],
                         pair=None if same_ else p, pair_type=None if same_ else pair_type(c, f), voicing_only=r["voicing_only"],
                         x_acoustic=x, x_acoustic_pair=None if same_ else round(XMEAN[p], 4), bin=None if same_ else bin_of(X[(p, t)]),
                         mn_dprime=None if same_ else audio_x(dp[(c, f)]), feature_distance=None if same_ else feature_distance(c, f),
                         snd1=sound_file(t, c), snd2=sound_file(t, f),
                         seed1=str(sft_noise.seed_for("calib_pilot", i, 1)), seed2=str(sft_noise.seed_for("calib_pilot", i, 2))))
    return rows


def design_table(rows):
    """One line per bin: how many trials, how the pair types / voice groups / consonants are spread."""
    out = []
    types = ["SS", "SF", "SN", "FF", "FN", "NN"]
    for bi in range(len(EDGES) - 1):
        sub = [r for r in rows if not r["is_same"] and r["bin"] == bi]
        cons = [c for r in sub for c in (r["consonant"], r["foil"])]
        top = sorted({c: cons.count(c) for c in set(cons)}.items(), key=lambda kv: -kv[1])[:3]
        out.append(dict(bin=f"{EDGES[bi]:.1f}-{EDGES[bi + 1]:.1f}", n=len(sub), voicing_only=sum(r["voicing_only"] for r in sub),
                        distinct_pairs=len({r["pair"] for r in sub}), **{ty: sum(r["pair_type"] == ty for r in sub) for ty in types},
                        **{g: sum(r["group"] == g for r in sub) for g in GROUPS},
                        most_frequent=" ".join(f"{c}:{n}" for c, n in top), share_with_sh=round(np.mean([("sh" in (r["consonant"], r["foil"])) for r in sub]), 2) if sub else 0))
    return out


def design_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "figure.dpi": 200, "axes.unicode_minus": False})
    types = [("SS", "塞音-塞音", "#1f618d"), ("SF", "塞音-擦音", "#5dade2"), ("SN", "塞音-鼻音", "#a9cce3"),
             ("FF", "擦音-擦音", "#b9770e"), ("FN", "擦音-鼻音", "#f0b27a"), ("NN", "鼻音-鼻音", "#7d3c98")]
    tab = design_table(rows); xs = np.arange(len(tab)); bottom = np.zeros(len(tab))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for ty, lab, col in types:
        v = np.array([r[ty] for r in tab]); ax.bar(xs, v, 0.7, bottom=bottom, color=col, label=lab); bottom += v
    for x_, r in zip(xs, tab):
        ax.text(x_, r["n"] + 0.4, f"{r['n']}" + (f"（清濁 {r['voicing_only']}）" if r["voicing_only"] else ""), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels([r["bin"] for r in tab]); ax.set_xlabel("頻譜圖上的差別 x (dB，該說話者，+6 dB 噪音)")
    ax.set_ylabel("「不同」題數"); ax.set_ylim(0, max(bottom) + 5); ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title(f"校準前測的 {N_DIFF} 題「不同」：每 1 dB 一格的題數與子音類別組成", loc="left", fontsize=10.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    os.makedirs(STIM, exist_ok=True)
    X, XMEAN = load_acoustic()
    rng = np.random.default_rng(PLAN_SEED)
    rows = make_plan(rng, X, XMEAN, load_dprime())
    for r in rows:
        for slot in (1, 2):
            fn = f"r{r['row']:03d}_s{slot}.wav"
            sft_noise.rebuild(r[f"snd{slot}"], SNR, int(r[f"seed{slot}"]), os.path.join(STIM, fn))
            r[f"file{slot}"] = "stim/" + fn
    snr_pilot.STIM = STIM                                           # write the headphone tones next to our syllables
    hp = snr_pilot.headphone_check(np.random.default_rng(PLAN_SEED + 1))
    tab = design_table(rows)
    cond = dict(design="calib_pilot", snr_db=SNR, n_different=N_DIFF, n_same=N_SAME, bin_edges=[float(e) for e in EDGES], plan_seed=PLAN_SEED,
                groups=GROUPS, trials=rows, headphone_check=hp, built="2026-11-25")
    with open(os.path.join(OUT, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=1)
    with open(os.path.join(OUT, "design.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)
    design_figure(rows, os.path.join(OUT, "design.png"))
    d = [r for r in rows if not r["is_same"]]
    xs = np.array([r["x_acoustic"] for r in d])
    print(f"{len(d)} different + {len(rows) - len(d)} same at {SNR:+g} dB; x range {xs.min():.2f}-{xs.max():.2f} dB; "
          f"{len({r['pair'] for r in d})} distinct pairs; groups {dict((g, sum(r['group'] == g for r in d)) for g in GROUPS)}")
    for r in tab:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    for a, b in VOICING:
        s = [r for r in d if r["pair"] == pair_key(a, b)]
        print(f"  voicing {a}-{b}: {len(s)} trials, x {min(r['x_acoustic'] for r in s):.2f}-{max(r['x_acoustic'] for r in s):.2f} dB, M&N d' {s[0]['mn_dprime']:.2f}")
    mb = sum(os.path.getsize(os.path.join(STIM, x)) for x in os.listdir(STIM)) / 1e6
    print(f"{2 * len(rows)} syllable files + {len(hp) * 3} headphone tones, {mb:.1f} MB in {STIM}")


if __name__ == "__main__":
    main()
