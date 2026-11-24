"""評估文件用的兩張圖：三把尺在兩位聽者上的排序力；12 位說話者的音量包絡相關。

    python analyze/evaluation_figs.py

讀：analyze/out/acoustic_scale_criteria.json（acoustic_scale_criteria.py）、analyze/out/stimuli_voice_check.json（stimuli_voice_check.py）
出：analyze/out/evaluation_three_rulers_two_listeners.png、evaluation_voice_groups.png
對應：review/figures/fig33、fig34；review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md §2、§3。
"""
import json
import os

import numpy as np

import _common as C

plt = C.setup_plot()
out = C.out_dir()
R = json.load(open(os.path.join(out, "acoustic_scale_criteria.json")))
RULERS = (("mn", "Miller & Nicely 的 d'\n（現在的尺）", "#95a5a6"), ("features", "差幾個發音特徵", "#d35400"), ("acoustic", "頻譜圖上的差別\n（企劃書主線的軸）", "#2471a3"))
NOTE = {"WEB01": "新聽者，英語母語；算法定案後才看到的資料", "yyc1116": "非英語母語；發展這把尺時用的資料"}

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ids = [p for p in ("yyc1116", "WEB01") if p in R]
for i, pid in enumerate(ids):
    for k, (key, lab, col) in enumerate(RULERS):
        v, ci = R[pid][f"auc_{key}"], R[pid][f"auc_{key}_ci95"]
        x = i + (k - 1) * 0.27
        ax.bar(x, v - 0.3, 0.24, bottom=0.3, color=col, label=lab.replace("\n", "") if i == 0 else None)
        ax.errorbar(x, v, [[v - ci[0]], [ci[1] - v]], color="k", lw=1, capsize=3)
        ax.text(x, ci[1] + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.text(0.5, 0.505, "0.5 = 排不出來", fontsize=8.5, color="#555", va="bottom", ha="center")
ax.axhline(0.65, color="#2471a3", lw=0.8, ls=":")
ax.text(0.5, 0.655, "企劃書判準 C1\n的門檻 .65", fontsize=8.5, color="#2471a3", va="bottom", ha="center")
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f"{p}（{R[p]['n']} 題）\n{NOTE.get(p, '')}" for p in ids], fontsize=9)
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("排序力（答對的題分數高於答錯的題的機率）")
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
ax.set_title("三把尺排不排得出聽者的對錯（誤差線 = 95% 拔靴區間）", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(out, "evaluation_three_rulers_two_listeners.png"))
plt.close(fig)

V = json.load(open(os.path.join(out, "stimuli_voice_check.json")))
E = np.array(V["envelope_corr"])
TK = [f"T{i:02d}" for i in range(1, 13)]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(E, vmin=0.75, vmax=1.0, cmap="Blues")
for i in range(12):
    for j in range(12):
        ax.text(j, i, f"{E[i, j]:.2f}"[1:] if E[i, j] < 0.995 else "1.0", ha="center", va="center", fontsize=7, color="white" if E[i, j] > 0.93 else "k")
for a, b in ((0, 6), (6, 9), (9, 12)):
    ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec="#c0392b", lw=2))
ax.set_xticks(range(12))
ax.set_xticklabels([f"{t}\n{V['pitch_hz'][t]['median']:.0f}" for t in TK], fontsize=8)
ax.set_yticks(range(12))
ax.set_yticklabels(TK, fontsize=8)
ax.set_xlabel("說話者編號；下面一列 = 母音的音高 (Hz)")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="音量包絡的相關（16 個子音平均）")
w, b = V["envelope_corr_within"], V["envelope_corr_between"]
ax.set_title(f"12 個說話者編號分成三組：組內相關 {w[0]:.2f}–{w[1]:.2f}，組間 {b[0]:.2f}–{b[1]:.2f}", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(out, "evaluation_voice_groups.png"))
plt.close(fig)
print("寫出：evaluation_three_rulers_two_listeners.png、evaluation_voice_groups.png")
