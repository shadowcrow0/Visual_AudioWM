"""Step 7. Figures for the splice experiment and for ㄍㄜ vs ㄍㄨㄛ (same layout as figs.py: waveform,
what the mouth is doing, spectrogram with boxes, one curve).

    python figs_splice.py --out out   (needs out/measures.json and out/splice/splice_results.json; writes out/figs/figS*.png)

Two curves are used. "centre": centre of gravity of 0.2-1.5 kHz (Hz), low (about 360) while the lips are round
and nearly closed, 450-550 for ㄛ. "upper": energy in 1.0-1.6 kHz relative to 0.2-0.9 kHz (dB). The second one was
chosen from the data because the first cannot tell ㄜ from ㄨㄛ: with spread lips (ㄜ) the second dark band sits
near 1.2 kHz, with rounded lips (ㄨㄛ, ㄛ) it drops below 1 kHz and that region is empty.
"""
import os, json, argparse, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import spectrogram, lfilter
from measure import landmarks, band_centroid
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--L", type=int, default=120, help="which spliced clip to show")
A = ap.parse_args(); OUT = f"{A.out}/figs"; os.makedirs(OUT, exist_ok=True)
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)      # only one of the listed fonts needs to exist
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"], "figure.dpi": 200, "axes.unicode_minus": False})
RED, BLUE, ORA, GREY, GREEN = "#c0392b", "#2471a3", "#d35400", "#7f8c8d", "#1e8449"
M = [r for r in json.load(open(f"{A.out}/measures.json")) if 0.10 <= r["dur"] <= 0.32]
R = json.load(open(f"{A.out}/splice/splice_results.json"))
syl = lambda r: r["label"].split("_")[0]
U_REF = float(np.mean([r["cog"][1] for r in M if syl(r) in ("pu", "fu", "du", "gu")]))     # where syllables that are only ㄨ sit

def band_db(x, sr, t, lo, hi, win=0.030):
    i = int((t - win/2)*sr); seg = x[max(i, 0): i + int(win*sr)]; seg = seg*np.hanning(len(seg)); P = np.abs(np.fft.rfft(seg, 8192))**2; fr = np.fft.rfftfreq(8192, 1/sr)
    return 10*np.log10(P[(fr >= lo) & (fr < hi)].sum() + 1e-20)
centre = lambda x, sr, t: band_centroid(x, sr, t)
upper = lambda x, sr, t: band_db(x, sr, t, 1000, 1600) - band_db(x, sr, t, 200, 900)
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def n1(panels, fname, title, curve, ylabel, ylim, unit, ref=None, ref_txt=""):
    """panels: two dicts(path, title, strips=[(x0, x1, text, colour)], boxes=[(x0, x1, y0, y1, colour)], notes=[(y, text, colour)]); x1 None = end of the syllable"""
    fig = plt.figure(figsize=(9.6, 7)); gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 0.42, 3, 1.5], hspace=0.09, wspace=0.06); T0, T1 = -130, 380
    for j, p in enumerate(panels):
        x, sr, i0, i1 = landmarks(p["path"]); t = (np.arange(len(x)) - i0)/sr*1000; end = (i1 - i0)/sr*1000; E = lambda v: end if v is None else v
        a0 = fig.add_subplot(gs[0, j]); a0.plot(t, x, "k", lw=.5); a0.set_xlim(T0, T1); a0.set_ylim(-1.05, 1.05); a0.set_xticks([]); a0.set_yticks([]); a0.set_title(p["title"], loc="left", fontsize=12)
        s = fig.add_subplot(gs[1, j]); s.set_xlim(T0, T1); s.set_ylim(0, 1); s.axis("off")
        for (x0, x1, txt, col) in p["strips"]:
            s.add_patch(plt.Rectangle((x0, .08), E(x1) - x0, .84, fc=col, alpha=.16, ec=col, lw=1)); s.text((x0 + E(x1))/2, .5, txt, ha="center", va="center", fontsize=8, color=col)
        a1 = fig.add_subplot(gs[2, j]); nper = int(.006*sr)
        f, tt, P = spectrogram(lfilter([1, -0.97], [1], x), fs=sr, window="hann", nperseg=nper, noverlap=nper - int(.001*sr), nfft=4096)   # +6 dB/octave so the upper band is visible
        D = 10*np.log10(P + 1e-12); D -= D[:, (tt > i0/sr) & (tt < i1/sr)].max()
        a1.pcolormesh(tt*1000 - i0/sr*1000, f/1000, D, vmin=-50, vmax=0, cmap="Greys", shading="auto"); a1.set_ylim(0, 3); a1.set_xlim(T0, T1); a1.set_xticklabels([]); a1.axvline(0, color="#888", lw=.8, ls=":")
        for (x0, x1, y0, y1, col) in p["boxes"]: a1.add_patch(plt.Rectangle((x0, y0), E(x1) - x0, y1 - y0, fill=False, ec=col, lw=1.8))
        for (y, txt, col) in p.get("notes", []): a1.text(T0 + 5, y, txt, color=col, fontsize=9, va="top", bbox=dict(fc="white", ec="none", alpha=.9, pad=1.5))
        a2 = fig.add_subplot(gs[3, j]); ts = np.arange(10, end - 5, 5); c = [curve(x, sr, i0/sr + u/1000) for u in ts]
        a2.plot(ts, c, color=RED, lw=1.8); a2.set_xlim(T0, T1); a2.set_ylim(*ylim); a2.axvline(0, color="#888", lw=.8, ls=":"); a2.set_xlabel("時間 (ms，0 = 嘴巴打開、聲帶開始振動)"); clean(a2)
        a2.text(ts[0] - 6, c[0], f"{c[0]:.0f}", color=RED, fontsize=9, ha="right", va="center"); a2.text(ts[-1] + 6, c[-1], f"{c[-1]:.0f} {unit}", color=RED, fontsize=9, ha="left", va="center")
        if ref is not None: a2.axhline(ref, color=GREY, lw=1, ls="--")
        if j == 0:
            a1.set_ylabel("頻率 (kHz)"); a2.set_ylabel(ylabel)
            if ref is not None: a2.text(T1 - 4, ref - (ylim[1] - ylim[0])*.03, ref_txt, color=GREY, fontsize=8.5, va="top", ha="right")
        else: a1.set_yticklabels([]); a2.set_yticklabels([])
    fig.suptitle(title, fontsize=12.5, y=0.955); fig.savefig(f"{OUT}/{fname}", bbox_inches="tight"); plt.close(fig); print("wrote", fname, [os.path.basename(p["path"]) for p in panels])

# ---- A. the untouched syllable next to the same syllable with L ms of ㄨ cut in
def splice_pair(kind):
    v = [r for r in R if r["kind"] == kind and r["L"] == A.L]
    if not v: return None, None
    hit = min(v, key=lambda r: np.nan_to_num(abs(r["f0_insert"] - r["f0_b"]), nan=1e9))            # the pair whose pitches match best: fewest splice artefacts to look at
    return next(r for r in R if r["kind"] == kind and r["L"] == 0 and r["b_file"] == hit["b_file"]), hit
for kind, zh, ch, closed, fname in (("bo", "ㄅㄛ", "撥", "雙唇閉著", "figS1_bo_vs_spliced_buo.png"), ("duo", "ㄉㄨㄛ", "多", "舌尖頂住上齒齦", "figS5_duo_vs_spliced.png")):
    o, s = splice_pair(kind)
    if o is None: print("no spliced clip for", kind); continue
    LOW, HI = (0.15, 1.3), 2.95
    n1([dict(path=f"{A.out}/splice/{o['file']}", title=f"原本的 {zh}（{ch}）", strips=[(-130, 0, closed, BLUE), (0, None, "嘴巴張大", ORA)], boxes=[(0, None, *LOW, BLUE)],
             notes=[(HI, f"藍框：原本的韻母，兩邊是同一段聲音\n紅框：剪接進去的 {A.L} ms ㄨ，黑塊停在低處", "k")]),
        dict(path=f"{A.out}/splice/{s['file']}", title=f"剪接：{zh[0]} + {A.L} ms 的 ㄨ + 原本的韻母", strips=[(-130, 0, closed, BLUE), (0, A.L, "嘴唇圓而小 (ㄨ)", RED), (A.L, None, "嘴巴張大", ORA)],
             boxes=[(0, A.L, *LOW, RED), (A.L, None, *LOW, BLUE)])],
       fname, f"真的唸出 ㄨ 會多什麼：{zh} 對 剪接出來的「{zh[0]}ㄨ{zh[-1]}」（剪接，不是自然發音）", centre, "0.2-1.5 kHz\n能量重心 (Hz)", (280, 680), "Hz", U_REF, f"虛線：只有 ㄨ 的字停在這裡（{U_REF:.0f} Hz）")

# ---- A2. the centre curve for every L, averaged over clips
fig, ax = plt.subplots(figsize=(7.8, 4.4)); cols = {0: "k", 40: "#e59866", 80: ORA, 120: RED, 160: "#7b241c"}
for L in sorted({r["L"] for r in R if r["kind"] == "bo"}):
    v = [r for r in R if r["kind"] == "bo" and r["L"] == L]; n = min(len(r["cog"]) for r in v); c = np.mean([r["cog"][:n] for r in v], 0)
    ax.plot(v[0]["t_ms"][:n], c, color=cols.get(L, GREY), lw=2.2, label=f"插入 {L} ms 的 ㄨ（{len(v)} 個）" if L else f"原本的 ㄅㄛ（{len(v)} 個）")
v = [r for r in R if r["kind"] == "boctl" and r["L"] == A.L]
if v:
    n = min(len(r["cog"]) for r in v); ax.plot(v[0]["t_ms"][:n], np.mean([r["cog"][:n] for r in v], 0), color=BLUE, lw=1.8, ls="--", label=f"對照：插入 {A.L} ms 的 ㄛ（{len(v)} 個）")
ax.axhline(U_REF, color=GREY, lw=1, ls="--", label=f"只有 ㄨ 的字停在這裡（{U_REF:.0f} Hz）")
ax.set_xlabel("時間 (ms，0 = 嘴巴打開)"); ax.set_ylabel("0.2-1.5 kHz 能量重心 (Hz)"); ax.legend(frameon=False, fontsize=9, loc="lower right"); clean(ax)
ax.set_title("插入的 ㄨ 越長，重心停在低處越久，之後才往上（各條 = 該條件所有剪接音檔的平均）", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/figS2_centre_by_L.png"); plt.close(fig); print("wrote figS2_centre_by_L.png")

# ---- B. where ㄨ does distinguish words: ㄍㄜ (歌) vs ㄍㄨㄛ (鍋); and which side ㄅㄛ is on
GR = [("ㄍㄜ (歌 哥)", ("ge",), GREEN), ("ㄍㄨㄛ (鍋 郭)", ("guo",), ORA), ("ㄉㄨㄛ (多)", ("duo",), "#b9770e"), ("ㄅㄛ (撥 玻 菠)", ("bo",), BLUE)]
T = {}
for r in M:
    if any(syl(r) in g[1] for g in GR):
        x, sr, i0, i1 = landmarks(f"{A.out}/{r['file']}"); r["upper"] = [upper(x, sr, i0/sr + q*(i1 - i0)/sr) for q in (.15, .5, .85)]; T.setdefault(syl(r), []).append(r)
json.dump([dict(label=r["label"], seed=r["seed"], upper_db=r["upper"], centre_hz=r["cog"]) for v in T.values() for r in v], open(f"{A.out}/splice/vowel_side.json", "w"), ensure_ascii=False, indent=1)
def rep(s):
    v = T[s]; med = np.median([r["upper"] for r in v], 0); return min(v, key=lambda r: np.abs(np.array(r["upper"]) - med).sum())    # the clip closest to its group's median
if all(s in T for s in ("ge", "guo", "duo", "bo")):
    g, u = rep("ge"), rep("guo"); UP, LOWB = (1.0, 1.6), (0.15, 0.9)
    n1([dict(path=f"{A.out}/{g['file']}", title=f"ㄍㄜ（{g['label'][-1]}）", strips=[(-130, 0, "舌根頂住軟顎", BLUE), (0, None, "舌根放下，嘴唇不圓", GREEN)], boxes=[(0, None, *LOWB, BLUE), (0, None, *UP, RED)],
             notes=[(2.95, "藍框 0.15-0.9 kHz：兩邊都有黑塊\n紅框 1-1.6 kHz：ㄍㄜ 有第二條黑帶，ㄍㄨㄛ 這裡是空的", "k")]),
        dict(path=f"{A.out}/{u['file']}", title=f"ㄍㄨㄛ（{u['label'][-1]}）", strips=[(-130, 0, "舌根頂住軟顎", BLUE), (0, None, "舌根放下，嘴唇圓", ORA)], boxes=[(0, None, *LOWB, BLUE), (0, None, *UP, RED)])],
       "figS3_ge_vs_guo.png", "ㄨ 分得出字的地方：ㄍㄜ 對 ㄍㄨㄛ（合成語音）", upper, "1-1.6 kHz 能量\n(dB，0.2-0.9 kHz = 0)", (-60, -5), "dB")
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); rng = np.random.default_rng(0); mid = {}; print("\n1-1.6 kHz re 0.2-0.9 kHz (dB) at 15 / 50 / 85 % of the vowel")
    for k, (nm, ss, col) in enumerate(GR):
        c = np.array([r["upper"] for s in ss for r in T[s]]); print(f"  {nm:14s} n={len(c):2d}  " + "  ".join(f"{m:6.1f}±{sd:4.1f}" for m, sd in zip(c.mean(0), c.std(0, ddof=1))))
        mid[ss[0]] = c[:, 1].mean()
        for i in range(3):
            xx = i + (k - 1.5)*.21; ax.bar(xx, c[:, i].mean() + 60, .19, bottom=-60, color=col, alpha=.8); ax.errorbar(xx, c[:, i].mean(), c[:, i].std(ddof=1), color="k", lw=1, capsize=3)
            ax.plot(xx + rng.uniform(-.06, .06, len(c)), c[:, i], "o", ms=2.2, color="k", alpha=.35); ax.text(xx, -59, f"{c[:, i].mean():.0f}", ha="center", va="bottom", fontsize=8.5, color="white")
    ax.set_xticks(range(3)); ax.set_xticklabels(["聲母之後的開頭 (15%)", "中間 (50%)", "結尾 (85%)"]); ax.set_ylim(-60, 0); ax.set_ylabel("1-1.6 kHz 能量 (dB，0.2-0.9 kHz = 0)"); clean(ax)
    ax.legend(handles=[Patch(fc=c, alpha=.8, label=f"{n}，{sum(len(T[s]) for s in ss)} 個音檔") for n, ss, c in GR], frameon=False, fontsize=9, loc="upper left", ncol=2)
    side = "ㄍㄨㄛ" if abs(mid["bo"] - mid["guo"]) < abs(mid["bo"] - mid["ge"]) else "ㄍㄜ"      # the title states what this run found, not what was expected
    ax.set_title(f"1-1.6 kHz 有沒有第二條黑帶：ㄅㄛ 中段 {mid['bo']:.0f} dB，靠近 {side}（ㄍㄜ {mid['ge']:.0f}、ㄍㄨㄛ {mid['guo']:.0f}）\n（合成語音；長條 = 平均 ± SD，點 = 每個音檔）", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/figS4_which_side.png"); plt.close(fig); print("wrote figS4_which_side.png")
else: print("ㄍㄜ / ㄍㄨㄛ figures skipped: need verified ge, guo, duo and bo clips")
