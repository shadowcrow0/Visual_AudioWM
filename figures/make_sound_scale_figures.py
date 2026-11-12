"""Supplementary figures / tables for the sound-scale section (review/論文規格書_聲音軸.md §G).

    python figures/make_sound_scale_figures.py

Writes into figures/:
  figS1_dprime_vs_luce.png     28 single-feature pairs: mdsdt GRT d' (-6 dB) against Luce -ln eta; the 0.911 line; outliers labelled
  figS2_separability_dAIC.png  14 sub-designs x 2 axes at -6 dB: dAIC = AIC(PS model) - AIC(full); <= 2 keep, > 10 reject
  tableS1_dprime_matrix.md     16 x 16 lower triangle of d' (stimuli/consonant_dprime.csv)
  tableS2_levels_to_pairs.md   for each LEVELS_AUDIO value and target consonant, the foil the experiment snaps to and its d'
Static PNG at 2x; one series per panel so no legend; text in ink, not in series colour.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import consonant_distance as cd                                   # noqa: E402
from adaptive_vawm_EN import LEVELS_AUDIO, load_dprime            # noqa: E402

D = os.path.join(ROOT, "review", "data", "grt_subdesigns")
IPA = {"theta": "θ", "eth": "ð", "sh": "ʃ", "zh": "ʒ"}
ipa = lambda c: IPA.get(c, c)   # noqa: E731
INK, INK2, GRID = "#1f1f1f", "#5f5f5f", "#e6e6e6"
MARK = "#4a6fa5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_s1():
    rows = [r for r in csv.DictReader(open(os.path.join(D, "pair_dprime_mdsdt.csv"), encoding="utf-8")) if r["snr"] == "-6"]
    agg = {}
    for r in rows:
        agg.setdefault(frozenset((r["c1"], r["c2"])), []).append(float(r["dprime"]))
    L = cd.luce_distance()
    pts = []
    for k, v in agg.items():
        a, b = sorted(k, key=lambda c: cd.INDEX[c])
        pts.append((L[cd.INDEX[a], cd.INDEX[b]], float(np.mean(v)), f"{ipa(a)}–{ipa(b)}"))
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    slope, icpt = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    resid = y - (icpt + slope * x)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    xx = np.linspace(0, 5.6, 2)
    ax.plot(xx, icpt + slope * xx, color=INK2, linewidth=1.2, zorder=1)
    ax.scatter(x, y, s=28, facecolor=MARK, edgecolor="white", linewidth=1.0, zorder=2)
    for xi, yi, lab, res in zip(x, y, [p[2] for p in pts], resid):
        if abs(res) > 0.35:
            ax.annotate(lab, (xi, yi), xytext=(5, 3 if res > 0 else -9), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Luce distance  −ln η  (Miller & Nicely, mean of −12/−6/0 dB)")
    ax.set_ylabel("GRT d′ from mdsdt 2×2 fits  (−6 dB)")
    ax.set_xlim(0, 5.6); ax.set_ylim(0, 5.2)
    ax.text(0.03, 0.97, f"d′ = {icpt:.2f} + {slope:.3f}·(−ln η)\nn = {len(x)} single-feature pairs, r = {r:.2f}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figS1_dprime_vs_luce.png")); plt.close(fig)
    return slope, icpt, r, len(x)


def fig_s2():
    S = list(csv.DictReader(open(os.path.join(D, "mdsdt_summary.csv"), encoding="utf-8")))
    S = [r for r in S if r["snr"] == "-6"]
    order = ["voicing", "nasality", "manner", "place"]
    S.sort(key=lambda r: (order.index(r["axis_f"]), order.index(r["axis_g"]), r["consonants"], r["sub"]))
    labels, vals = [], []
    for r in S:
        cons = " ".join(ipa(c) for c in r["consonants"].split())
        sub = "" if r["sub"] == "all" else f" [{r['sub']}]"
        labels.append(f"{r['axis_f']} × {r['axis_g']}   {cons}{sub}")
        f = lambda k: float(r[k]) if r[k] not in ("", "nan") else np.nan   # noqa: E731
        vals.append([f("d_PS_x"), f("d_PS_y")])
    M = np.array(vals)
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(labels) + 1.2), dpi=200)
    norm = TwoSlopeNorm(vmin=-5, vcenter=2, vmax=25)
    show = np.clip(M, -5, 25)
    im = ax.imshow(np.ma.masked_invalid(show), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["axis f separable\nacross g?  (PS_x)", "axis g separable\nacross f?  (PS_y)"])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(2):
            v = M[i, j]
            txt = "fit failed" if np.isnan(v) else f"{v:+.1f}"
            dark_cell = (not np.isnan(v)) and (v < -3 or v > 14)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if dark_cell else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-5, 2, 10, 25])
    cb.set_label("ΔAIC = AIC(PS model) − AIC(full)\n≤ 2 keep separability · > 10 reject", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Perceptual separability in Miller & Nicely (1955), −6 dB", fontsize=9, loc="right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figS2_separability_dAIC.png")); plt.close(fig)
    return len(labels)


def table_s1():
    C = cd.CONSONANTS
    M = cd.matrix("luce") * 0.911
    with open(os.path.join(HERE, "tableS1_dprime_matrix.md"), "w", encoding="utf-8") as f:
        f.write("Table S1. d′ between consonant pairs (0.911 × Luce −ln η from Miller & Nicely 1955, −12/−6/0 dB; lower triangle).\n\n")
        f.write("| | " + " | ".join(ipa(c) for c in C[:-1]) + " |\n|" + "---|" * len(C) + "\n")
        for i in range(1, len(C)):
            f.write(f"| **{ipa(C[i])}** | " + " | ".join(f"{M[i, j]:.2f}" for j in range(i)) + " |" * (len(C) - 1 - i) + " |\n")


def table_s2():
    tab = load_dprime()
    C = cd.CONSONANTS
    with open(os.path.join(HERE, "tableS2_levels_to_pairs.md"), "w", encoding="utf-8") as f:
        f.write("Table S2. The foil each target consonant is paired with at each calibration level (nearest available d′; value shown).\n\n")
        f.write("| target | " + " | ".join(f"level {x}" for x in LEVELS_AUDIO) + " |\n|" + "---|" * (len(LEVELS_AUDIO) + 1) + "\n")
        for t in C:
            cells = []
            for x in LEVELS_AUDIO:
                foil, d = min(tab[t], key=lambda z: abs(z[1] - x))
                cells.append(f"{ipa(foil)} ({d:.2f})")
            f.write(f"| {ipa(t)} | " + " | ".join(cells) + " |\n")


if __name__ == "__main__":
    s, b, r, n = fig_s1(); print(f"figS1: d' = {b:.3f} + {s:.3f} luce, r={r:.3f}, n={n}")
    print("figS2 rows:", fig_s2())
    table_s1(); table_s2(); print("tables written")
