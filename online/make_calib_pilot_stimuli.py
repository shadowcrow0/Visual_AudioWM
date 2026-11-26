"""Build the ONLINE calibration pilot (online/calib_pilot/): 216 different + 108 same consonant pairs, spread along
the acoustic axis (spectrogram difference in our noise), every syllable pre-mixed.

    python analyze/acoustic_scale_compute.py             # once: writes analyze/out/acoustic_scale_pairs.csv (about 25 s)
    python online/make_calib_pilot_stimuli.py            # version 2: calib_pilot/stim/*.wav, conditions.json, design.csv, design.png
    python online/pack_stimuli.py calib_pilot            # stim/ -> bundles/
    python online/make_calib_pilot_stimuli.py --version 1 [--stim DIR]   # the plan CAL01 ran -> conditions_v1.json (and its sounds into DIR)

Why a second pilot instead of changing online/snr_pilot/: that one produced WEB01's data and must stay rebuildable.
What is different here (review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md §7, step 1):
- The unit is pair x talker and its x is the acoustic difference of THAT talker's two files at the trial's SNR
  (analyze/acoustic_scale_compute.py, columns ex_T01..ex_T12), not the d' table. Trials fill 1-dB bins from 0.5 to 8.5 dB
  as evenly as the material allows, so the slope AND the ceiling are sampled.
- The seven voicing-only pairs (b-p, d-t, g-k, f-v, s-z, th-dh, sh-zh) are forced in, both orders, one per voice group:
  the d' table calls them easy, the acoustic scale calls them hard, and no listener has been tested on them in noise.
- The 12 talker labels are three near-identical voices (T01-T06, T07-T09, T10-T12; analyze/stimuli_voice_check.py),
  so sampling is balanced over the three GROUPS and the group is logged.
- Within each bin the pair types (stop/fricative/nasal, within or across class) are spread as far as the material allows,
  so that "large difference" is not just "contains sh" and "small" is not just "b v k p". design.csv shows what was achieved.
- Seeds are written as STRINGS: a browser stores 64-bit integers as doubles and rounds them.

Two versions, both rebuilt bit-for-bit by this file:
- Version 1 (PLAN_SEED, 2026-11-25; CAL01 ran it): everything at +6 dB, eight 1-dB bins x 27 trials. Kept as
  calib_pilot/conditions_v1.json so that CAL01's data can always be matched to its plan.
- Version 2 (PLAN_SEED_V2, 2026-11-26; the default): +6 dB and 0 dB trials interleaved. Why: at +6 dB CAL01 was at ceiling
  in six of the eight bins and the lower end of his curve was not sampled (review/pilot_2026-11-26_CAL01_calib_pilot.md).
  Version 2 IS version 1 with three changes, so everything else stays comparable with CAL01:
    1. the +6 dB trials of the two top bins (6.5 dB and above, 54 trials, all at ceiling) give way, except N_ANCHOR
       of them that stay as a check of the ceiling;
    2. their rows are filled with 0 dB trials below ZERO_EDGES[-1] dB, spread over half-dB bins, pair types and the
       three voice groups, plus the seven voicing-only pairs once in each order on talkers version 1 already uses;
    3. N_SAME_ZERO of the 108 same trials are mixed at 0 dB (one same trial per two different trials at either SNR, so
       the noise level says nothing about the answer; each SNR's false-alarm rate is the floor of its own curve).
  Rows that did not change keep their row number, seeds and file name, so their sounds are the same bytes as in
  version 1. Changed rows get seeds from "calib_pilot_v2" and a file name ending in _snr0: a version-1 bundle next to a
  version-2 plan (a half-finished upload) is then caught by the page's own check instead of playing the wrong sounds.
  The numbers N_ANCHOR / ZERO_EDGES / N_SAME_ZERO were chosen by simulation: analyze/calib_pilot_design_check.py --allocations.

The plan is fixed by the seeds (same for every participant); the browser shuffles the order per participant, so the
two SNRs are interleaved at random.
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
SNR_LOW = 0.0                                    # version 2: the second, harder SNR
N_DIFF, N_SAME = 216, 108
EDGES = np.arange(0.5, 9.0, 1.0)                 # 8 bins: 0.5-1.5, ..., 7.5-8.5 dB
PLAN_SEED = 20261125
PLAN_SEED_V2 = 20261126                          # the revision of 2026-11-26; version-2 data files carry this plan_seed
# Version 2, chosen by simulation (analyze/calib_pilot_design_check.py --allocations; online/calib_pilot/README.md):
REPLACE_BINS = (6, 7)                            # the +6 dB bins that give way (6.5-7.5 and 7.5-8.5 dB)
N_ANCHOR = 6                                     # of their 54 trials, this many stay (one per bin x voice group)
ZERO_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)      # half-dB bins the 0 dB different trials are spread over
ZERO_CAP = {0: 4}                                # at most 4 trials below 0.5 dB: only 12 such units exist, nearly all stop-stop in G1
VOICING_AT_ZERO = True                           # the seven voicing-only pairs once in each order at 0 dB as well
N_SAME_ZERO = 24                                 # same trials at 0 dB (1 same : 2 different at both SNRs)
OUT = os.path.join(HERE, "calib_pilot")
STIM = os.path.join(OUT, "stim")
PAIRS_CSV = os.path.join(ROOT, "analyze", "out", "acoustic_scale_pairs.csv")

GROUPS = {"G1": TALKERS[0:6], "G2": TALKERS[6:9], "G3": TALKERS[9:12]}       # analyze/stimuli_voice_check.py
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
CLASS = {**{c: "S" for c in "ptkbdg"}, **{c: "F" for c in ("f", "theta", "s", "sh", "v", "eth", "z", "zh")}, "m": "N", "n": "N"}
VOICING = [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("theta", "eth"), ("s", "z"), ("sh", "zh")]
pair_key = lambda a, b: "-".join(sorted([a, b]))
pair_type = lambda a, b: "".join(sorted([CLASS[a], CLASS[b]], key="SFN".index))        # SS, SF, SN, FF, FN, NN


def load_acoustic(snr=SNR):
    """{(pair, talker): x} and {pair: 12-talker mean} at `snr`, from analyze/acoustic_scale_compute.py."""
    if not os.path.exists(PAIRS_CSV):
        raise SystemExit(f"missing {PAIRS_CSV}: run  python analyze/acoustic_scale_compute.py  first")
    x, mean = {}, {}
    with open(PAIRS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["snr"] in ("clean",) or float(r["snr"]) != snr:
                continue
            p = pair_key(r["a"], r["b"])
            mean[p] = float(r["excess"])
            for t in TALKERS:
                x[(p, t)] = float(r[f"ex_{t}"])
    if len(mean) != 120:
        raise SystemExit(f"{PAIRS_CSV}: expected 120 pairs at {snr:g} dB, found {len(mean)}")
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


def make_plan_v2(rows1, X0, XMEAN0, table, replace_bins=REPLACE_BINS, n_anchor=N_ANCHOR, zero_edges=ZERO_EDGES, zero_cap=ZERO_CAP,
                 voicing_at_zero=VOICING_AT_ZERO, n_same_zero=N_SAME_ZERO, seed=PLAN_SEED_V2):
    """Version 2 = the version-1 rows (make_plan) with some rows re-used for 0 dB trials. See the module docstring.
    The keyword arguments exist so that analyze/calib_pilot_design_check.py can compare other allocations."""
    rng = np.random.default_rng(seed)
    dp = {(c, f): d for c, lst in table.items() for f, d in lst}
    rows = [dict(r, snr_db=SNR) for r in rows1]
    zero_edges = np.asarray(zero_edges, float); nb0 = len(zero_edges) - 1
    zbin = lambda v: int(np.searchsorted(zero_edges, v, side="right") - 1) if zero_edges[0] <= v < zero_edges[-1] else -1

    # 1. the +6 dB trials that give way: the replaced bins, minus the anchors (bin x voice group in turn, pair types spread)
    top = [r for r in rows if not r["is_same"] and r["bin"] in replace_bins]
    cells = [(b, g) for b in replace_bins for g in GROUPS]
    anchors, k = [], 0
    while len(anchors) < n_anchor:
        b, g = cells[k % len(cells)]; k += 1
        pool = [r for r in top if r["bin"] == b and r["group"] == g and r["row"] not in {a["row"] for a in anchors}]
        if pool:
            types = [a["pair_type"] for a in anchors]
            anchors.append(min(pool, key=lambda r: (types.count(r["pair_type"]), rng.random())))
        elif k > 50 * len(cells):
            raise SystemExit("not enough trials in the replaced bins for that many anchors")
    gone = sorted({r["row"] for r in top} - {a["row"] for a in anchors})
    kept = [r for r in rows if not r["is_same"] and r["row"] not in gone]

    # 2. the 0 dB different trials that take their rows
    use_pair, use_cons, use_talker, first = {}, {}, {}, {}
    for r in kept:
        a, _ = r["pair"].split("-")
        first[r["pair"]] = first.get(r["pair"], 0) + (1 if r["consonant"] == a else -1)
        for d, key in ((use_pair, r["pair"]), (use_talker, r["talker"]), (use_cons, r["consonant"]), (use_cons, r["foil"])):
            d[key] = d.get(key, 0) + 1
    chosen = []

    def add(p, t, order=None, voicing=False):
        a, b = p.split("-")
        if order is None:
            order = (a, b) if first.get(p, 0) <= 0 else (b, a)
        first[p] = first.get(p, 0) + (1 if order[0] == a else -1)
        for d, key in ((use_pair, p), (use_talker, t), (use_cons, a), (use_cons, b)):
            d[key] = d.get(key, 0) + 1
        chosen.append(dict(consonant=order[0], foil=order[1], talker=t, pair=p, voicing_only=int(voicing), zbin=zbin(X0[(p, t)])))

    vkeys = [pair_key(a, b) for a, b in VOICING]
    if voicing_at_zero:
        # each voicing pair once in each order, on two voice groups, with the talker the +6 dB trials of that pair
        # already use in that group: the same two files are then heard at both SNRs
        for i, p in enumerate(vkeys):
            a, b = p.split("-")
            for j, order in enumerate(((a, b), (b, a))):
                g = list(GROUPS)[(i + j) % 3]
                t = next(r["talker"] for r in kept if r["pair"] == p and r["group"] == g)
                add(p, t, order, True)
    have = np.array([sum(c["zbin"] == i for c in chosen) for i in range(nb0)])
    cap = np.array([zero_cap.get(i, 10 ** 6) for i in range(nb0)])
    left = len(gone) - len(chosen); target = have.copy()
    if left < 0:
        raise SystemExit("fewer rows to fill than voicing trials at 0 dB")
    while left > 0:                                    # water-filling, as in version 1, within the caps
        i = min((i for i in range(nb0) if target[i] < cap[i]), key=lambda i: target[i]); target[i] += 1; left -= 1
    units = [(p, t) for (p, t) in X0 if p not in vkeys]
    pools = {i: [u for u in units if zbin(X0[u]) == i] for i in range(nb0)}
    for bi in sorted(range(nb0), key=lambda i: len(pools[i])):          # the scarcest bins first
        pool = pools[bi]
        while sum(c["zbin"] == bi for c in chosen) < target[bi]:
            if not pool:
                raise SystemExit(f"0 dB bin {bi} ran out of pair x talker units")
            here = [c for c in chosen if c["zbin"] == bi]
            n_type = lambda ty: sum(pair_type(c["consonant"], c["foil"]) == ty for c in here)
            n_group = lambda g: sum(GROUP_OF[c["talker"]] == g for c in here)
            n_pair = lambda p: sum(c["pair"] == p for c in here)
            n_cons = lambda x: sum(x in (c["consonant"], c["foil"]) for c in here)

            def cost(u):
                p, t = u; a, b = p.split("-")
                return (6 * n_pair(p) + 4 * n_type(pair_type(a, b)) + 3 * n_group(GROUP_OF[t]) + 2 * max(n_cons(a), n_cons(b))
                        + use_pair.get(p, 0) + max(use_cons.get(a, 0), use_cons.get(b, 0)) / 4 + use_talker.get(t, 0) / 6
                        + rng.random() * 0.5)
            u = min(pool, key=cost); pool.remove(u)                     # a unit is heard once at 0 dB
            add(*u)
    for i, k in zip(gone, rng.permutation(len(chosen))):
        c = chosen[k]; cons, foil, t, p = c["consonant"], c["foil"], c["talker"], c["pair"]
        rows[i - 1] = dict(row=i, is_same=0, consonant=cons, foil=foil, talker=t, group=GROUP_OF[t], pair=p, pair_type=pair_type(cons, foil),
                           voicing_only=c["voicing_only"], x_acoustic=round(X0[(p, t)], 4), x_acoustic_pair=round(XMEAN0[p], 4),
                           bin=bin_of(X0[(p, t)]), mn_dprime=audio_x(dp[(cons, foil)]), feature_distance=feature_distance(cons, foil),
                           snd1=sound_file(t, cons), snd2=sound_file(t, foil),
                           seed1=str(sft_noise.seed_for("calib_pilot_v2", i, 1)), seed2=str(sft_noise.seed_for("calib_pilot_v2", i, 2)), snr_db=SNR_LOW)

    # 3. same trials at 0 dB: spread over (consonant, voice group) cells first, then consonants, then groups
    same = [r for r in rows if r["is_same"]]
    n_cell, n_c, n_g, taken = {}, {}, {}, []
    for _ in range(n_same_zero):
        r = min((r for r in same if r["row"] not in taken),
                key=lambda r: (n_cell.get((r["consonant"], r["group"]), 0), n_c.get(r["consonant"], 0), n_g.get(r["group"], 0), rng.random()))
        taken.append(r["row"])
        for d, key in ((n_cell, (r["consonant"], r["group"])), (n_c, r["consonant"]), (n_g, r["group"])):
            d[key] = d.get(key, 0) + 1
    for i in taken:
        rows[i - 1].update(seed1=str(sft_noise.seed_for("calib_pilot_v2", i, 1)), seed2=str(sft_noise.seed_for("calib_pilot_v2", i, 2)), snr_db=SNR_LOW)
    return rows


def snr_of(r):
    return r.get("snr_db", SNR)


def design_table(rows):
    """One line per SNR x bin: how many trials, how the pair types / voice groups / consonants are spread."""
    out = []
    types = ["SS", "SF", "SN", "FF", "FN", "NN"]
    for snr in sorted({snr_of(r) for r in rows}, reverse=True):
        for bi in range(len(EDGES) - 1):
            sub = [r for r in rows if not r["is_same"] and r["bin"] == bi and snr_of(r) == snr]
            if not sub:
                continue
            cons = [c for r in sub for c in (r["consonant"], r["foil"])]
            top = sorted({c: cons.count(c) for c in set(cons)}.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            xs = [r["x_acoustic"] for r in sub]
            out.append(dict(snr_db=f"{snr:g}", bin=f"{EDGES[bi]:.1f}-{EDGES[bi + 1]:.1f}", x_min=round(min(xs), 2), x_max=round(max(xs), 2), n=len(sub),
                            voicing_only=sum(r["voicing_only"] for r in sub), distinct_pairs=len({r["pair"] for r in sub}),
                            **{ty: sum(r["pair_type"] == ty for r in sub) for ty in types}, **{g: sum(r["group"] == g for r in sub) for g in GROUPS},
                            most_frequent=" ".join(f"{c}:{n}" for c, n in top), share_with_sh=round(np.mean([("sh" in (r["consonant"], r["foil"])) for r in sub]), 2),
                            share_with_b_or_v=round(np.mean([bool({"b", "v"} & {r["consonant"], r["foil"]}) for r in sub]), 2)))
    return out


def design_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "figure.dpi": 200, "axes.unicode_minus": False})
    types = [("SS", "塞音-塞音", "#1f618d"), ("SF", "塞音-擦音", "#5dade2"), ("SN", "塞音-鼻音", "#a9cce3"),
             ("FF", "擦音-擦音", "#b9770e"), ("FN", "擦音-鼻音", "#f0b27a"), ("NN", "鼻音-鼻音", "#7d3c98")]
    tab = design_table(rows); snrs = sorted({float(r["snr_db"]) for r in tab}, reverse=True)
    labels = [f"{EDGES[i]:.1f}-{EDGES[i + 1]:.1f}" for i in range(len(EDGES) - 1)]
    w = 0.7 / len(snrs)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, snr in enumerate(snrs):                       # one bar per SNR in every bin: +6 dB plain, 0 dB hatched
        by = {r["bin"]: r for r in tab if float(r["snr_db"]) == snr}
        xs = np.arange(len(labels)) + (k - (len(snrs) - 1) / 2) * w; bottom = np.zeros(len(labels))
        for ty, lab, col in types:
            v = np.array([by[b][ty] if b in by else 0 for b in labels])
            ax.bar(xs, v, w * 0.92, bottom=bottom, color=col, label=lab if k == 0 else None, hatch=None if k == 0 else "///", edgecolor="white", linewidth=0.4)
            bottom += v
        for x_, b in zip(xs, labels):
            if b in by:
                r = by[b]
                ax.text(x_, r["n"] + 0.4, f"{r['n']}" + (f"\n清濁 {r['voicing_only']}" if r["voicing_only"] else ""), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("頻譜圖上的差別 x (dB，該說話者，該題的噪音量)" + ("；每格左 = +6 dB，右（斜線）= 0 dB" if len(snrs) > 1 else ""))
    ax.set_ylabel("「不同」題數"); ax.set_ylim(0, max(r["n"] for r in tab) + 8); ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    n_by = {s: sum(r["n"] for r in tab if float(r["snr_db"]) == s) for s in snrs}
    ax.set_title(f"校準前測的 {N_DIFF} 題「不同」（" + "、".join(f"{s:+g} dB {n} 題" for s, n in n_by.items()) + "）：每 1 dB 一格的題數與子音類別組成", loc="left", fontsize=10.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def file_name(r, slot):
    """Unchanged rows keep the version-1 name; 0 dB rows say so, so that bundles of the two versions cannot be mixed up."""
    return f"r{r['row']:03d}_s{slot}" + ("" if snr_of(r) == SNR else f"_snr{snr_of(r):g}") + ".wav"


def build_plan(version):
    """The rows of version 1 or 2, without sounds (nothing is written)."""
    X, XMEAN = load_acoustic(SNR)
    rows = make_plan(np.random.default_rng(PLAN_SEED), X, XMEAN, load_dprime())
    if version == 2:
        rows = make_plan_v2(rows, *load_acoustic(SNR_LOW), load_dprime())
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the online calibration pilot (plan, sounds, design table).")
    ap.add_argument("--version", type=int, choices=(1, 2), default=2, help="1 = the +6 dB plan CAL01 ran (writes conditions_v1.json); 2 = +6 and 0 dB (default)")
    ap.add_argument("--stim", default=None, help="version 1 only: also write its sounds into this folder")
    a = ap.parse_args()
    v1 = a.version == 1
    if v1 and a.stim is None:
        import tempfile
        stim = tempfile.mkdtemp()                                   # the headphone-check generator always writes its tones
    else:
        stim = a.stim if v1 else STIM
    os.makedirs(stim, exist_ok=True)
    rows = build_plan(a.version)
    for r in rows:
        for slot in (1, 2):
            fn = file_name(r, slot)
            if not (v1 and a.stim is None):
                sft_noise.rebuild(r[f"snd{slot}"], snr_of(r), int(r[f"seed{slot}"]), os.path.join(stim, fn))
            r[f"file{slot}"] = "stim/" + fn
    snr_pilot.STIM = stim                                           # write the headphone tones next to our syllables
    hp = snr_pilot.headphone_check(np.random.default_rng(PLAN_SEED + 1))
    d = [r for r in rows if not r["is_same"]]
    if v1:
        cond = dict(design="calib_pilot", snr_db=SNR, n_different=N_DIFF, n_same=N_SAME, bin_edges=[float(e) for e in EDGES], plan_seed=PLAN_SEED,
                    groups=GROUPS, trials=rows, headphone_check=hp, built="2026-11-25")
        path = os.path.join(OUT, "conditions_v1.json")
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        new = json.dumps(cond, indent=1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"version 1: {len(d)} different + {len(rows) - len(d)} same at {SNR:+g} dB -> {os.path.relpath(path, ROOT)}"
              + ("" if old is None else " (identical to the file that was there)" if old == new else " (DIFFERENT from the file that was there)"))
        return
    snrs = sorted({snr_of(r) for r in rows})
    by_snr = {f"{s:g}": dict(different=sum(snr_of(r) == s for r in d), same=sum(snr_of(r) == s and r["is_same"] for r in rows)) for s in snrs}
    cond = dict(design="calib_pilot", design_version=2, snrs=snrs, n_different=N_DIFF, n_same=N_SAME, n_by_snr=by_snr, bin_edges=[float(e) for e in EDGES],
                plan_seed=PLAN_SEED_V2, base_plan_seed=PLAN_SEED, groups=GROUPS, trials=rows, headphone_check=hp, built="2026-11-26")
    with open(os.path.join(OUT, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=1)
    wanted = {os.path.basename(r[f"file{slot}"]) for r in rows for slot in (1, 2)} | {f for t in hp for f in t["files"]}
    for f in os.listdir(stim):                                      # sounds of rows that changed since the last build
        if f.endswith(".wav") and f not in wanted:
            os.remove(os.path.join(stim, f))
    tab = design_table(rows)
    with open(os.path.join(OUT, "design.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)
    design_figure(rows, os.path.join(OUT, "design.png"))
    print(f"version 2: {len(d)} different + {len(rows) - len(d)} same; " + "; ".join(f"{float(s):+g} dB: {v['different']} different + {v['same']} same" for s, v in by_snr.items()))
    for s in snrs:
        xs = np.array([r["x_acoustic"] for r in d if snr_of(r) == s])
        print(f"  {s:+g} dB: x range {xs.min():.2f}-{xs.max():.2f} dB; {len({r['pair'] for r in d if snr_of(r) == s})} distinct pairs; "
              f"groups {dict((g, sum(r['group'] == g for r in d if snr_of(r) == s)) for g in GROUPS)}")
    for r in tab:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    for a_, b_ in VOICING:
        for s in snrs:
            q = [r for r in d if r["pair"] == pair_key(a_, b_) and snr_of(r) == s]
            if q:
                print(f"  voicing {a_}-{b_} at {s:+g} dB: {len(q)} trials, x {min(r['x_acoustic'] for r in q):.2f}-{max(r['x_acoustic'] for r in q):.2f} dB, "
                      f"groups {' '.join(sorted(r['group'] for r in q))}, M&N d' {q[0]['mn_dprime']:.2f}")
    sm = [r for r in rows if r["is_same"]]
    for s in snrs:
        q = [r for r in sm if snr_of(r) == s]
        print(f"  same at {s:+g} dB: {len(q)} trials, {len({r['consonant'] for r in q})} consonants, groups {dict((g, sum(r['group'] == g for r in q)) for g in GROUPS)}")
    mb = sum(os.path.getsize(os.path.join(stim, x)) for x in os.listdir(stim)) / 1e6
    print(f"{2 * len(rows)} syllable files + {len(hp) * 3} headphone tones, {mb:.1f} MB in {stim}")


if __name__ == "__main__":
    main()
