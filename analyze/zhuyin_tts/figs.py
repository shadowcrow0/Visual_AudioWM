"""Step 4. Draw the comparison figures (waveform, spectrogram with boxes, band centre-of-gravity trace).

    python figs.py --out out          (reads out/measures.json and the wav files, writes out/figs/*.png)

Needs a font with Chinese and Bopomofo glyphs: WenQuanYi Zen Hei (Arch: wqy-zenhei) or Noto Sans CJK TC.
"""
import os, sys, json, argparse, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import spectrogram, lfilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import landmarks, band_centroid
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ARGS = ap.parse_args()
OUT = os.path.join(ARGS.out, "figs"); os.makedirs(OUT, exist_ok=True)
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)      # only one of the listed fonts needs to exist
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"], "figure.dpi": 200, "axes.unicode_minus": False})
RED, BLUE, ORA, GREY = "#c0392b", "#2471a3", "#d35400", "#7f8c8d"
ZH = {"bo": "ㄅㄛ", "po": "ㄆㄛ", "duo": "ㄉㄨㄛ", "tuo": "ㄊㄨㄛ", "guo": "ㄍㄨㄛ", "pu": "ㄆㄨ", "fu": "ㄈㄨ", "du": "ㄉㄨ", "gu": "ㄍㄨ"}
ART0 = {"b": "雙唇閉著", "p": "雙唇閉著", "d": "舌尖頂住上齒齦", "t": "舌尖頂住上齒齦", "g": "舌根頂住軟顎", "f": "下唇靠上齒"}
M = [r for r in json.load(open(os.path.join(ARGS.out, "measures.json"))) if 0.10 <= r["dur"] <= 0.32]      # outside this range the syllable was not segmented correctly
for r in M: r["syl"], r["char"] = r["label"].split("_")
G = lambda *syl: [r for r in M if r["syl"] in syl]
U_REF = float(np.mean([r["cog"][1] for r in G("pu", "fu", "du", "gu")]))          # where the "ㄨ only" syllables sit
def rep(syl, char=None):
    """the token closest to its group's median trajectory (so the figure is not a cherry-picked one)"""
    v = [r for r in G(syl) if char is None or r["char"] == char] or G(syl)
    if not v: sys.exit(f"no usable clip for {syl}: synthesize more seeds")
    med = np.median([r["cog"] for r in v], 0)
    return min(v, key=lambda r: np.abs(np.array(r["cog"]) - med).sum())
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def n1(left, right, fname, title, boxes, rime_txt):
    fig = plt.figure(figsize=(9.4, 7)); gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 0.42, 3, 1.5], hspace=0.09, wspace=0.06); T0, T1 = -110, 300
    for j, r in enumerate((left, right)):
        x, sr, a, b = landmarks(os.path.join(ARGS.out, r["file"])); t = (np.arange(len(x)) - a)/sr*1000; end = (b - a)/sr*1000
        a0 = fig.add_subplot(gs[0, j]); a0.plot(t, x, "k", lw=.5); a0.set_xlim(T0, T1); a0.set_ylim(-1.05, 1.05); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"{ZH[r['syl']]}（{r['char']}）", loc="left", fontsize=12)
        s = fig.add_subplot(gs[1, j]); s.set_xlim(T0, T1); s.set_ylim(0, 1); s.axis("off")
        for (x0, x1, txt, col) in ((T0, 0, ART0[r["syl"][0]], BLUE), (0, max(end, 200), rime_txt[j], ORA)):
            s.add_patch(plt.Rectangle((x0, .08), x1 - x0, .84, fc=col, alpha=.16, ec=col, lw=1)); s.text((x0 + x1)/2, .5, txt, ha="center", va="center", fontsize=8.5, color=col)
        a1 = fig.add_subplot(gs[2, j]); nper = int(.006*sr)
        f, tt, P = spectrogram(lfilter([1, -0.97], [1], x), fs=sr, window="hann", nperseg=nper, noverlap=nper - int(.001*sr), nfft=4096); D = 10*np.log10(P + 1e-12); D -= D[:, (tt > a/sr) & (tt < b/sr)].max()
        a1.pcolormesh(tt*1000 - a/sr*1000, f/1000, D, vmin=-50, vmax=0, cmap="Greys", shading="auto"); a1.set_ylim(0, 3); a1.set_xlim(T0, T1); a1.set_xticklabels([]); a1.axvline(0, color="#888", lw=.8, ls=":")
        for (x0, x1, y0, y1, col, txt, ty) in boxes:
            x1_ = end if x1 is None else x1; a1.add_patch(plt.Rectangle((x0, y0), x1_ - x0, y1 - y0, fill=False, ec=col, lw=1.8))
            if j == 0 and txt: a1.text(T0 + 5, ty, txt, color=col, fontsize=9, va="top", bbox=dict(fc="white", ec="none", alpha=.9, pad=1.5))
        a2 = fig.add_subplot(gs[3, j]); ts = np.arange(10, end - 5, 5); c = [band_centroid(x, sr, a/sr + u/1000) for u in ts]
        a2.plot(ts, c, color=RED, lw=1.8); a2.axhline(U_REF, color=GREY, lw=1, ls="--"); a2.set_xlim(T0, T1); a2.set_ylim(280, 680); a2.axvline(0, color="#888", lw=.8, ls=":"); a2.set_xlabel("時間 (ms，0 = 嘴巴打開、聲帶開始振動)"); clean(a2)
        a2.text(ts[0] - 6, c[0], f"{c[0]:.0f}", color=RED, fontsize=9, ha="right", va="center"); a2.text(ts[-1] + 6, c[-1], f"{c[-1]:.0f} Hz", color=RED, fontsize=9, ha="left", va="center")
        if j == 0:
            a1.set_ylabel("頻率 (kHz)"); a2.set_ylabel("0.2-1.5 kHz\n能量重心 (Hz)"); a2.text(T1 - 4, U_REF - 8, f"虛線：只有 ㄨ 的字停在這裡（{U_REF:.0f} Hz）", color=GREY, fontsize=8.5, va="top", ha="right")
        else: a1.set_yticklabels([]); a2.set_yticklabels([])
    fig.suptitle(title, fontsize=12.5, y=0.955); fig.savefig(f"{OUT}/{fname}", bbox_inches="tight"); plt.close(fig)
    print(fname, [(r["label"], r["seed"], [round(v) for v in r["cog"]], round(1000*r["dur"])) for r in (left, right)])
SAME = (0, None, 0.15, 1.3, BLUE, "藍框：最下面那一大塊黑色（兩條黑帶疊在一起），\n位置和往上移的走向兩邊一樣", 2.95)
n1(rep("bo", "撥"), rep("duo"), "figT1_bo_vs_duo.png", "ㄅㄛ 對 ㄉㄨㄛ（合成語音，臺灣口音）：聲母之後的部分一樣", [SAME], ["雙唇打開 → 嘴巴張大", "舌尖放下 → 嘴巴張大"])
n1(rep("bo", "撥"), rep("guo", "鍋"), "figT2_bo_vs_guo.png", "ㄅㄛ 對 ㄍㄨㄛ（合成語音，臺灣口音）：聲母之後的部分一樣", [SAME], ["雙唇打開 → 嘴巴張大", "舌根放下 → 嘴巴張大"])
n1(rep("duo"), rep("du"), "figT3_duo_vs_du.png", "對照：ㄉㄨㄛ 對 ㄉㄨ（合成語音）。沒有 ㄛ 的時候，重心不會往上",
   [(0, None, 0.15, 1.3, RED, "紅框：ㄉㄨㄛ 的黑帶往上移，ㄉㄨ 的停在低處", 2.95)], ["舌尖放下 → 嘴巴張大", "舌尖放下，嘴唇維持圓而小"])
# ---- N2-style summary over all verified tokens
from matplotlib.patches import Patch
GR = [("唇音 + ㄛ\n(ㄅㄛ ㄆㄛ)", G("bo", "po"), BLUE), ("其他聲母 + ㄨㄛ\n(ㄉㄨㄛ ㄊㄨㄛ ㄍㄨㄛ)", G("duo", "tuo", "guo"), ORA), ("只有 ㄨ\n(ㄆㄨ ㄈㄨ ㄉㄨ ㄍㄨ)", G("pu", "fu", "du", "gu"), GREY)]
fig, ax = plt.subplots(figsize=(8, 4.6)); rng = np.random.default_rng(0)
for i, lab in enumerate(("開頭 (15%)", "中間 (50%)", "結尾 (85%)")):
    for k, (nm, v, col) in enumerate(GR):
        c = np.array([r["cog"][i] for r in v]); xx = i + (k - 1)*.27
        ax.bar(xx, c.mean() - 250, .24, bottom=250, color=col, alpha=.8); ax.errorbar(xx, c.mean(), c.std(ddof=1), color="k", lw=1, capsize=3)
        ax.plot(xx + rng.uniform(-.08, .08, len(c)), c, "o", ms=2.2, color="k", alpha=.35); ax.text(xx, 256, f"{c.mean():.0f}", ha="center", va="bottom", fontsize=9, color="white")
ax.set_xticks(range(3)); ax.set_xticklabels(["聲母之後的開頭 (15%)", "中間 (50%)", "結尾 (85%)"]); ax.set_ylim(250, 700); ax.set_ylabel("0.2-1.5 kHz 能量重心 (Hz)"); clean(ax)
ax.legend(handles=[Patch(fc=c, alpha=.8, label=f"{n.replace(chr(10), ' ')}，{len(v)} 個音檔") for n, v, c in GR], frameon=False, fontsize=9, loc="upper left")
ax.set_title("合成語音（臺灣口音，單一聲音）：唇音 + ㄛ 和 其他聲母 + ㄨㄛ 走同一條路，只有 ㄨ 的不往上\n（長條 = 平均 ± SD，點 = 每個音檔）", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/figT4_groups.png"); plt.close(fig)
for nm, v, _ in GR:
    c = np.array([r["cog"] for r in v]); F1 = np.array([r["F1"] for r in v]); F2 = np.array([r["F2"] for r in v])
    print(nm.replace("\n", " "), len(v), "COG", np.round(c.mean(0)), "±", np.round(c.std(0, ddof=1)), "| F1", np.round(np.nanmean(F1, 0)), "| F2", np.round(np.nanmean(F2, 0)), "| dur %.0f±%.0f" % (1000*np.mean([r["dur"] for r in v]), 1000*np.std([r["dur"] for r in v], ddof=1)))
a = np.array([r["cog"] for r in GR[0][1]]); b = np.array([r["cog"] for r in GR[1][1]]); u = np.array([r["cog"] for r in GR[2][1]])
pooled = lambda p, q: np.sqrt((p.var(0, ddof=1) + q.var(0, ddof=1))/2)
print("labial+o minus other+uo:", np.round(a.mean(0) - b.mean(0)), "d =", np.round((a.mean(0) - b.mean(0))/pooled(a, b), 2)); print("other+uo minus u:", np.round(b.mean(0) - u.mean(0)), "d =", np.round((b.mean(0) - u.mean(0))/pooled(b, u), 2))
print("rise (end - start): labial+o %.0f±%.0f, other+uo %.0f±%.0f, u %.0f±%.0f" % ((a[:, 2] - a[:, 0]).mean(), (a[:, 2] - a[:, 0]).std(ddof=1), (b[:, 2] - b[:, 0]).mean(), (b[:, 2] - b[:, 0]).std(ddof=1), (u[:, 2] - u[:, 0]).mean(), (u[:, 2] - u[:, 0]).std(ddof=1)))
print("kept tokens", len(M), "U_REF", round(U_REF))
