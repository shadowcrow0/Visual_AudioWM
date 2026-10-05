"""
English procedure figure, cascade layout (same layout as the GRTv3_ada flow figure).
Panel B shows a COLOUR-CALIBRATION trial: the probe keeps the studied sound and shifts only the colour.
Minimal text: labels and times on the cascade and the timeline only.
Font: APA 7 §7.26 — sans serif inside the figure, 8–14 pt. Arial (rendered here with the metric-identical Liberation Sans; the SVG names Arial first).

    python figures/make_procedure_figure_en.py   # -> figures/experiment_procedure_en.svg / .png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.unicode_minus"] = False

INK, GREY, SCREEN = "#333333", "#666666", "#7a7a7a"
C1, C2, C1_SHIFT = "#9b59b6", "#3f51b5", "#b06fc9"     # studied colours; probe = colour 1 shifted by x

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 5.4, 0.9, 2.6], hspace=0.12)

# ───────────────────────────── A. Session ─────────────────────────────
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0, 9.6, "A. Session", fontsize=13, va="top", color=INK)
stages = [
    (0,   7,  "Instructions",            "",                                              "#eeeeee"),
    (9,   15, "Part 1  Colour calibration", "72 trials · 6 ΔE levels × 8 + 24 match\nsound = studied sound · feedback", "#dde3f3"),
    (26,  5,  "Fit",                     "lnrm2\n20–40 s",                                "#e6e6e6"),
    (33,  15, "Part 2  Sound calibration", "72 trials · 6 confusion levels × 8 + 24 match\ncolour = studied colour · feedback", "#e8dcf0"),
    (50,  5,  "Fit",                     "lnrm2\n20–40 s",                                "#e6e6e6"),
    (57,  11, "Practice",                "5 sets × 9 probes\nfeedback 1 s",               "#f3e8d3"),
    (70,  22, "Main task",               "6 blocks × 24 sets × 9 probes = 1296\nno feedback · rest between blocks", "#dcefd9"),
    (94,  6,  "End",                     "",                                              "#eeeeee"),
]
for x, w, t, s, c in stages:
    ax.add_patch(FancyBboxPatch((x, 3.0), w, 5.2, boxstyle="round,pad=0,rounding_size=0.4", fc=c, ec=GREY, lw=1))
    ax.text(x + w / 2, 7.1 if s else 5.6, t, ha="center", va="center", fontsize=10.5, color=INK)
    if s:
        ax.text(x + w / 2, 4.8, s, ha="center", va="center", fontsize=8, color=GREY)
for i in range(len(stages) - 1):
    x0 = stages[i][0] + stages[i][1]; x1 = stages[i + 1][0]
    ax.add_patch(FancyArrowPatch((x0 + 0.2, 5.6), (x1 - 0.2, 5.6), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
    if stages[i + 1][2] in ("Practice", "Main task") or stages[i][2].startswith("Part"):
        ax.text((x0 + x1) / 2, 6.4, "rest", ha="center", va="bottom", fontsize=8, color=GREY)

# ───────────────────────────── B. Trial (colour-calibration example) ─────────────────────────────
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(0, 99, "B. Trial (colour-calibration example: the probe keeps the studied sound and changes only the colour)",
        fontsize=13, va="top", color=INK)

W, H = 9.2, 13.0
DX, DY = 10.6, 6.0


def speaker(ax, x, y, s=1.0):
    ax.add_patch(Polygon([(x, y - 0.5 * s), (x + 0.5 * s, y - 0.5 * s), (x + 1.2 * s, y - 1.1 * s), (x + 1.2 * s, y + 1.1 * s),
                          (x + 0.5 * s, y + 0.5 * s), (x, y + 0.5 * s)], closed=True, fc="white", ec="white"))
    for r in (0.9, 1.5):
        ax.add_patch(Arc((x + 1.3 * s, y), r * s, 1.6 * r * s, theta1=-45, theta2=45, color="white", lw=1.0))


def patch(ax, x, y, colour, s=1.0):
    ax.add_patch(Rectangle((x - 1.1 * s, y - 1.6 * s), 2.2 * s, 3.2 * s, fc=colour, ec="#bbbbbb", lw=0.5))


def screen(ax, i, title, t, note, content, dashed=False):
    x = 1 + i * DX; y = 86 - i * DY
    ax.add_patch(Rectangle((x + 0.4, y - H - 0.4), W, H, fc="#bdbdbd", ec="none"))
    ax.add_patch(Rectangle((x, y - H), W, H, fc=SCREEN, ec=INK, lw=1.2, ls="--" if dashed else "-"))
    cx, cy = x + W / 2, y - H / 2
    for kind, args in content:
        if kind == "fix":
            ax.text(cx, cy, "+", color="white", ha="center", va="center", fontsize=14)
        elif kind == "patchL":
            patch(ax, x + 0.28 * W, cy, args)
        elif kind == "patchR":
            patch(ax, x + 0.72 * W, cy, args)
        elif kind == "patchC":
            patch(ax, cx - 0.6, cy + 0.6, args)
        elif kind == "spk":
            speaker(ax, cx + 0.2, cy)
        elif kind == "spkR":
            speaker(ax, cx + 1.0, cy + 0.6)
        elif kind == "text":
            ax.text(cx, cy, args, color="white", ha="center", va="center", fontsize=9.5)
        elif kind == "yn":
            ax.text(cx, cy + 2.2, "Same?", color="white", ha="center", va="center", fontsize=9)
            ax.text(cx, cy - 1.6, "[y] same    [n] different", color="white", ha="center", va="center", fontsize=8)
    ax.text(cx, y - H - 1.6, title, ha="center", va="top", fontsize=10.5, color=INK)
    ax.text(cx, y - H - 4.6, t, ha="center", va="top", fontsize=8.5, color=GREY)
    if note:
        ax.text(cx, y - H - 7.4, note, ha="center", va="top", fontsize=8, color=GREY, linespacing=1.25)
    return x, y


screens = [
    ("Fixation",  "0 – 0.3 s",   "",                              [("fix", None)], False),
    ("Colour 1",  "0.3 – 1.3 s", "left, 1 s",                     [("patchL", C1)], False),
    ("Colour 2",  "1.3 – 2.3 s", "right, 1 s",                    [("patchR", C2)], False),
    ("Sound 1",   "2.3 – 3.3 s", "consonant 1, blank screen",     [("spk", None)], False),
    ("Sound 2",   "3.3 – 4.3 s", "consonant 2, blank screen",     [("spk", None)], False),
    ("Fixation",  "0 – 0.3 s",   "",                              [("fix", None)], False),
    ("Probe",     "0.3 – 1.3 s", "colour 1 shifted by x\n+ sound 1 (unchanged)", [("patchC", C1_SHIFT), ("spkR", None)], False),
    ("Response",  "0.5 – 2.5 s", "y = same, n = different\n(calibration: up to 3 s)", [("yn", None)], False),
    ("Feedback",  "0.8 s",       "calibration and practice only", [("text", "Correct")], True),
]
pos = [screen(ax, i, *sc) for i, sc in enumerate(screens)]

x1_, y1_ = pos[1]; x4_, y4_ = pos[4]
ax.plot([x1_, x4_ + W], [y1_ + 1.5, y1_ + 1.5], color=INK, lw=1)
ax.plot([x1_, x1_], [y1_ + 1.5, y1_ + 0.5], color=INK, lw=1); ax.plot([x4_ + W, x4_ + W], [y1_ + 1.5, y1_ + 0.5], color=INK, lw=1)
ax.text((x1_ + x4_ + W) / 2, y1_ + 2.3, "Study, 4.3 s", ha="center", va="bottom", fontsize=9.5, color=INK)
x5, y5 = pos[5]; x8, y8 = pos[8]
ax.plot([x5, x8 + W], [y5 + 1.5, y5 + 1.5], color=INK, lw=1)
ax.plot([x5, x5], [y5 + 1.5, y5 + 0.5], color=INK, lw=1); ax.plot([x8 + W, x8 + W], [y5 + 1.5, y5 + 0.5], color=INK, lw=1)
ax.text((x5 + x8 + W) / 2, y5 + 2.3, "Probe, 3.3 s  (× 1 in calibration, × 9 in the main task)", ha="center", va="bottom", fontsize=9.5, color=INK)
x2_, y2_ = pos[2]
ax.add_patch(FancyArrowPatch((x2_, y2_ - H - 15.5), (x8 + W + 1, y8 - H - 15.5), arrowstyle="-|>", mutation_scale=14, color="#999999", lw=1))
ax.text(x8 + W + 1.5, y8 - H - 16.3, "time", ha="left", va="top", fontsize=9, color=GREY, style="italic")
ax.add_patch(FancyArrowPatch((x8 + W + 0.5, y8 - H / 2), (x8 + W + 3.5, y8 - H / 2), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
ax.text(x8 + W + 4, y8 - H / 2, "next trial", ha="left", va="center", fontsize=9.5, color=INK)

# ───────────────────────────── C. Timeline ─────────────────────────────
ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0, 9.6, "C. Timeline", fontsize=13, va="top", color=INK)
segs = [("Fix.", 0.3, "#f2f2f2"), ("Study  (4 × 1 s: colour 1, colour 2, sound 1, sound 2)", 4.0, "#dde3f3"),
        ("Fix.", 0.3, "#f2f2f2"), ("Probe", 1.0, "#e8dcf0"), ("Response  (from 0.5 s)", 1.5, "#f3e8d3"),
        ("Feedback", 0.8, "#dcefd9")]
scale = 70 / sum(s for _, s, _ in segs)
x = 0
ticks = [0.0]
for lab, dur, c in segs:
    w = dur * scale
    ax.add_patch(Rectangle((x, 3.5), w, 3.5, fc=c, ec=GREY, lw=1, ls="--" if lab == "Feedback" else "-"))
    ax.text(x + w / 2, 5.25, lab, ha="center", va="center", fontsize=8.5 if w > 6 else 7.5, color=INK)
    x += w
    ticks.append(ticks[-1] + dur)
labels = ["0", "0.3", "4.3", "4.6", "5.6", "7.1", "7.9 s"]
for i, tk in enumerate(ticks):
    dx = 1.2 if labels[i] in ("0.3", "4.6") else 0.0
    ax.text(tk * scale + dx, 3.0, labels[i], ha="center", va="top", fontsize=8, color=GREY)

# ───────────────────────────── D. Adaptive calibration (Part 1, colour, lnrm2) ─────────────────────────────
import numpy as np
gsD = gs[3].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.22)
ax = fig.add_subplot(gsD[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(0, 112, "D. Adaptive calibration — Part 1, colour (method of constant stimuli + lnrm2)", fontsize=13, va="top", color=INK, clip_on=False)
# study screen (two colours) → probe (colour ± x, same sound) → response → lnrm2 → H / L
bw, bh, by = 22, 34, 40
def mini_screen(x, label, sub, content):
    ax.add_patch(Rectangle((x + 0.8, by - 0.8), bw, bh, fc="#bdbdbd", ec="none"))
    ax.add_patch(Rectangle((x, by), bw, bh, fc=SCREEN, ec=INK, lw=1.1))
    cx, cy = x + bw / 2, by + bh / 2
    for kind, args in content:
        if kind == "patchL": patch(ax, x + 0.3 * bw, cy, args, s=2.2)
        elif kind == "patchR": patch(ax, x + 0.7 * bw, cy, args, s=2.2)
        elif kind == "patchC": patch(ax, cx - 2.5, cy + 1.5, args, s=2.2)
        elif kind == "spk": speaker(ax, cx + 1.5, cy + 1.5, s=2.2)
        elif kind == "yn":
            ax.text(cx, cy + 6, "Same?", color="white", ha="center", va="center", fontsize=8.5)
            ax.text(cx, cy - 5, "[y]   [n]", color="white", ha="center", va="center", fontsize=8.5)
    ax.text(cx, by - 5, label, ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(cx, by - 14, sub, ha="center", va="top", fontsize=8, color=GREY)
mini_screen(0,  "Study",    "two colours + two sounds",           [("patchL", C1), ("patchR", C2)])
mini_screen(27, "Probe",    "colour 1 ± x,\nsound 1 unchanged",   [("patchC", C1_SHIFT), ("spk", None)])
mini_screen(54, "Response", "y / n, RT",                          [("yn", None)])
for x0 in (bw + 0.5, 27 + bw + 0.5):
    ax.add_patch(FancyArrowPatch((x0, by + bh / 2), (x0 + 4, by + bh / 2), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1))
# loop ×72 above
ax.plot([54 + bw / 2, 54 + bw / 2, bw / 2, bw / 2], [by + bh + 2, by + bh + 12, by + bh + 12, by + bh + 2], color=INK, lw=1)
ax.add_patch(FancyArrowPatch((bw / 2, by + bh + 8), (bw / 2, by + bh + 2.5), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1))
ax.text(38, by + bh + 13.5, "× 72 trials: 6 ΔE levels × 8 + 24 same, shuffled", ha="center", va="bottom", fontsize=8.5, color=INK)
# fit box
ax.add_patch(FancyBboxPatch((82, by + 4), 17, bh - 8, boxstyle="round,pad=0,rounding_size=2", fc="#fdf2cc", ec=INK, lw=1))
ax.text(90.5, by + bh / 2 + 6, "lnrm2", ha="center", va="center", fontsize=10, color=INK)
ax.text(90.5, by + bh / 2 - 5, "fit accuracy + RT,\ninvert for H / L", ha="center", va="center", fontsize=8, color=GREY)
ax.add_patch(FancyArrowPatch((54 + bw + 0.5, by + bh / 2), (81.5, by + bh / 2), arrowstyle="-|>", mutation_scale=10, color=INK, lw=1))
ax.text(90.5, by - 5, "after the 72nd trial", ha="center", va="top", fontsize=8, color=GREY)

# D-middle: the 72-trial sequence of ΔE levels (schematic)
rng = np.random.default_rng(3)
levels = [2, 6, 12, 20, 30, 45]
plan = [v for v in levels for _ in range(8)] + [0] * 24
rng.shuffle(plan)
plan = np.array(plan, float)
ax = fig.add_subplot(gsD[1])
tr = np.arange(1, 73)
is_same = plan == 0
ax.scatter(tr[~is_same], plan[~is_same], s=14, color="#4b3f8f", zorder=3)
ax.scatter(tr[is_same], plan[is_same], s=14, facecolors="white", edgecolors="#4b3f8f", zorder=3)
for v in levels:
    ax.axhline(v, color="#cccccc", lw=0.6, zorder=1)
ax.set_xlim(0, 73); ax.set_ylim(-3, 52)
ax.set_xticks([1, 36, 72]); ax.set_yticks(levels)
ax.set_xlabel("trial", fontsize=9); ax.set_ylabel("probe ΔE (colour 1 ± x)", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_title("Levels presented across the 72 trials (schematic)", fontsize=9.5, color=INK)
ax.text(72, 48, "○  same probe (ΔE = 0)", ha="right", va="top", fontsize=8, color=GREY)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# D-right: fitted drift separation 2·d(ΔE) and the inversion to H / L
alpha, alpha2, h_targ, l_targ = 1.0, -0.12, 2.0, 0.5          # schematic observer, units of ΔE / 10
x = np.linspace(0, 45, 300); u = x / 10
sep = 2 * (alpha * u + alpha2 * u ** 2)
def invert(t): return 10 * (-alpha / alpha2 - np.sqrt((alpha / alpha2) ** 2 + 2 / alpha2 * t)) / 2
xH, xL = invert(h_targ), invert(l_targ)
ax = fig.add_subplot(gsD[2])
ax.plot(x, sep, color="#4b3f8f", lw=1.6)
for t, xv, lab in ((h_targ, xH, "H"), (l_targ, xL, "L")):
    ax.plot([0, xv], [t, t], color="#7b68ee", lw=1, ls="--")
    ax.plot([xv, xv], [0, t], color="#7b68ee", lw=1, ls="--")
    ax.text(xv + 0.6, 0.12, f"ΔE_{lab}", ha="left", va="bottom", fontsize=8.5, color=INK)
    ax.text(45, t + 0.08, f"target {lab} = {t}", ha="right", va="bottom", fontsize=8, color=INK)
for v in levels:
    ax.plot(v, 2 * (alpha * v / 10 + alpha2 * (v / 10) ** 2), "o", ms=4, color="#4b3f8f")
ax.set_xlim(0, 45); ax.set_ylim(0, 4.6)
ax.set_xticks([0, 10, 20, 30, 40]); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel("probe ΔE", fontsize=9); ax.set_ylabel("drift separation  2·d(ΔE)", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_title("Fitted lnrm2 curve and inversion to H / L (schematic)", fontsize=9.5, color=INK)
ax.text(1, 4.45, "ΔE_H and ΔE_L are then used in practice and the main task", ha="left", va="top", fontsize=8, color=GREY)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

svg = os.path.join(HERE, "experiment_procedure_en.svg")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "experiment_procedure_en.png"), dpi=170, bbox_inches="tight")
# name Arial first in the SVG so it renders in the real font where it is installed
txt = open(svg, encoding="utf-8").read()
txt = re.sub(r"font-family:\s*'?Liberation Sans'?", "font-family: 'Arial', 'Liberation Sans', sans-serif", txt)
open(svg, "w", encoding="utf-8").write(txt)
print("saved")
