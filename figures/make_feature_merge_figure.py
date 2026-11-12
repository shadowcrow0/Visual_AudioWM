"""Fig. M1: why Miller & Nicely's five features become four axes (affrication × duration is not factorial → one 3-level manner axis),
and where the 16 consonants then sit (voicing × manner × place, nasals apart). Static PNG, text in ink, no colour encoding.

    python figures/make_feature_merge_figure.py      → figures/figM1_feature_merge.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, FILL, EMPTY = "#1f1f1f", "#5f5f5f", "#bdbdbd", "#eef2f7", "#f7f7f7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

fig = plt.figure(figsize=(12.5, 5.0), dpi=200)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.7], wspace=0.3)


def cell(ax, x, y, w, h, text, sub="", face=FILL, hatch=None, fs=10):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=GRID, linewidth=1, hatch=hatch))
    ax.text(x + w / 2, y + h / 2 + (0.08 if sub else 0), text, ha="center", va="center", fontsize=fs, color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center", fontsize=7, color=INK2)


# ---- A: affrication × duration, M&N's two binary features
ax = fig.add_subplot(gs[0]); ax.set_xlim(-1.1, 2.1); ax.set_ylim(-0.8, 2.75); ax.axis("off")
ax.text(-1.1, 2.62, "A  M&N: affrication × duration", fontsize=9.5, weight="bold", ha="left")
ax.text(0.5, 2.15, "duration 0\n(short)", ha="center", fontsize=8, color=INK2)
ax.text(1.5, 2.15, "duration 1\n(long)", ha="center", fontsize=8, color=INK2)
ax.text(-0.08, 1.5, "affrication 0\n(stop / nasal)", ha="right", va="center", fontsize=8, color=INK2)
ax.text(-0.08, 0.5, "affrication 1\n(fricative)", ha="right", va="center", fontsize=8, color=INK2)
cell(ax, 0, 1, 1, 1, "p t k\nb d g\nm n", fs=9.5)
cell(ax, 1, 1, 1, 1, "empty", face=EMPTY, hatch="///", fs=8)
cell(ax, 0, 0, 1, 1, "f θ\nv ð", fs=9.5)
cell(ax, 1, 0, 1, 1, "s ʃ\nz ʒ", fs=9.5)
ax.text(0.5, -0.25, "3 of 4 cells filled → not a full factorial.\nNo consonant is long without being fricated:\nduration exists only inside affrication = 1.",
        ha="center", va="top", fontsize=7.5, color=INK2)

# ---- B: the merged ordinal manner axis
ax = fig.add_subplot(gs[1]); ax.set_xlim(-0.75, 2.05); ax.set_ylim(-0.8, 2.75); ax.axis("off")
ax.text(-0.75, 2.62, "B  → one ordinal axis: manner", fontsize=9.5, weight="bold", ha="left")
for i, (lab, cons, sub) in enumerate([("manner 0", "p t k  b d g  m n", "closure (stop / nasal)"),
                                       ("manner 1", "f θ  v ð", "short frication"),
                                       ("manner 2", "s ʃ  z ʒ", "long, intense frication")]):
    y = 1.6 - i * 0.75
    cell(ax, 0, y, 1.5, 0.6, cons, sub, fs=8.5)
    ax.text(-0.06, y + 0.3, lab, ha="right", va="center", fontsize=8, color=INK2)
ax.add_patch(FancyArrowPatch((0.75, 1.58), (0.75, 0.42), arrowstyle="-|>", mutation_scale=10, color=INK2, linewidth=0.8, zorder=0))
ax.text(1.58, 1.0, "more\nfrication,\nlonger", fontsize=7, color=INK2, va="center")
ax.text(0.75, -0.25, "(0,0)→0  (1,0)→1  (1,1)→2\nthe missing (0,1) never\nhas to be coded", ha="center", va="top", fontsize=7.5, color=INK2)

# ---- C: the resulting layout of the 16 consonants
ax = fig.add_subplot(gs[2]); ax.set_xlim(-1.6, 6.4); ax.set_ylim(-1.5, 3.6); ax.axis("off")
ax.text(-1.6, 3.45, "C  the four axes that remain: voicing × manner × place, nasality apart", fontsize=9.5, weight="bold", ha="left")
places = ["front", "middle", "back"]
layout = {  # (voicing, manner): [front, middle, back]
    (0, 0): ["p", "t", "k"], (0, 1): ["f", "θ", None], (0, 2): [None, "s", "ʃ"],
    (1, 0): ["b", "d", "g"], (1, 1): ["v", "ð", None], (1, 2): [None, "z", "ʒ"]}
for v in (0, 1):
    x0 = v * 3.2
    ax.text(x0 + 1.5, 3.05, "voiceless" if v == 0 else "voiced", ha="center", fontsize=8.5, color=INK)
    for pi, p in enumerate(places):
        ax.text(x0 + pi + 0.5, 2.78, p, ha="center", fontsize=7, color=INK2)
    for m in (0, 1, 2):
        y = 2.0 - m * 0.75
        if v == 0:
            ax.text(-0.1, y + 0.3, f"manner {m}", ha="right", va="center", fontsize=8, color=INK2)
        for pi in range(3):
            c = layout[(v, m)][pi]
            if c is None:
                cell(ax, x0 + pi, y, 1, 0.6, "", face=EMPTY, hatch="///")
            else:
                cell(ax, x0 + pi, y, 1, 0.6, c, fs=11)
# nasals
y = -0.55
ax.text(-0.1, y + 0.3, "nasal\n(manner 0, voiced)", ha="right", va="center", fontsize=7.5, color=INK2)
cell(ax, 3.2, y, 1, 0.6, "m", fs=11); cell(ax, 4.2, y, 1, 0.6, "n", fs=11); cell(ax, 5.2, y, 1, 0.6, "", face=EMPTY, hatch="///")
ax.text(2.4, -1.05, "hatched = no English consonant.  A GRT sub-design is a rectangle of filled cells with one consonant per cell:\n"
        "voicing × place within manner 0 (2×3); voicing × manner at middle place (2×3); nasality × place = b m / d n (2×2) …",
        ha="center", va="top", fontsize=7.2, color=INK2)

fig.savefig(os.path.join(HERE, "figM1_feature_merge.png"), bbox_inches="tight")
print("wrote figM1_feature_merge.png")
