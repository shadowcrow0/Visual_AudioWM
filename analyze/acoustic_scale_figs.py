"""聲學尺（四）：畫圖。

讀：analyze/out/acoustic_scale_feats.npz、_pairs.csv、_validation.json、_use.json、_timecourse.csv、_trials_<聽者>.csv
    （先依序跑 acoustic_scale_compute.py、acoustic_scale_validate.py、acoustic_scale_use.py）
出：analyze/out/acoustic_scale_noise_covers.png      一對音（預設 d 對 z，T01）乾淨 vs 加噪音：噪音蓋掉什麼、還露出什麼
    acoustic_scale_vs_mn.png                        120 對：頻譜圖差別（+6 dB）對 M&N d′
    acoustic_scale_hit_by_quartile.png              非母語聽者：頻譜圖差別四等分的命中率
    acoustic_scale_three_rulers.png                 同一批題目，三把尺的排序力
    acoustic_scale_distribution_by_snr.png          各 SNR 下 120 對的分布與難 / 易界線
    acoustic_scale_timecourse.png                   差別在時間上的位置
    acoustic_scale_frequency_profile.png            差別在頻率上的位置（語音形噪音先蓋低頻）
對應：review/figures/fig24-fig27（企劃書用的四張是 noise_covers、three_rulers、distribution_by_snr、vs_mn）。

圖上的數字都從上面那些檔讀出來，標題裡的數字也是；沒有手填的。
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

import _common as C
from acoustic_scale_compute import NSEED, SNR_KEYS, T_POST, T_PRE, band_centres, load_pairs
from acoustic_scale_validate import load_mn

LAB = {"clean": "乾淨", "12.0": "+12 dB", "6.0": "+6 dB", "0.0": "0 dB", "-6.0": "-6 dB"}
COLS = {"clean": "k", "12.0": "#5dade2", "6.0": C.BLUE, "0.0": "#d35400", "-6.0": C.RED}
MARKED = ("b-p", "g-k", "k-p", "p-t", "d-g", "f-theta", "m-n", "s-sh", "b-v", "m-sh", "s-theta")     # 散布圖上標名字的對：兩把尺分歧最大的與幾個熟面孔
pp = lambda p: "-".join(C.ipa(c) for c in p.split("-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116")
    ap.add_argument("--pair", nargs=2, default=["d", "z"], help="noise_covers 圖用的那一對")
    ap.add_argument("--talker", default="T01")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_d = C.out_dir(a.out)
    plt = C.setup_plot()
    import matplotlib.ticker
    f_ = lambda name: os.path.join(out_d, f"acoustic_scale_{name}")
    Z = np.load(f_("feats.npz"))
    CONS, TALK = list(Z["cons"]), list(Z["talkers"])
    P = load_pairs(out_d)
    V = json.load(open(f_("validation.json")))
    U = json.load(open(f_("use.json")))
    Y = pd.read_csv(f_(f"trials_{a.participant}.csv"))
    MN = load_mn()[1].mn_dprime
    CEN = band_centres()
    band_of = lambda hz: float(np.interp(hz, CEN, np.arange(32) + .5))
    T_MS = np.arange(-T_PRE, T_POST) * 10

    # ---- 一對音，乾淨 vs 加噪音
    A_, B_, TK, SI = a.pair[0], a.pair[1], a.talker, 2
    ia, ib, ti = CONS.index(A_), CONS.index(B_), TALK.index(TK)
    fig = plt.figure(figsize=(9.6, 9))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 2.4, 1.5], hspace=0.16, wspace=0.06)
    ticks = [200, 500, 1000, 2000, 4000, 7000]
    nums = {}
    for j, si in enumerate((0, SI)):
        X = Z[f"s{si}"]
        xa, xb = X[ia, ti, 0], X[ib, ti, 1]
        d = np.abs(xa - xb)
        fl2 = np.zeros(T_PRE + T_POST)
        if si:
            fl2 = np.mean([((X[c, ti, p] - X[c, ti, q]) ** 2).mean(-1) for c in (ia, ib) for p, q in itertools.combinations(range(NSEED), 2)], 0)
        ex_t = np.sqrt(np.clip((d ** 2).mean(-1) - fl2, 0, None))
        r = P[(P.pair == "-".join(sorted([A_, B_]))) & (P.snr == SNR_KEYS[si])].iloc[0]
        nums[si] = r[f"ex_{TK}"]
        vmax = max(Z["s0"][ia, ti, 0].max(), Z["s0"][ib, ti, 0].max())
        for k, (x_, nm) in enumerate(((xa, A_), (xb, B_))):
            ax = fig.add_subplot(gs[k, j])
            ax.imshow(x_.T, origin="lower", aspect="auto", cmap="Greys", vmin=vmax - 50, vmax=vmax, extent=[-300, 250, 0, 32])
            ax.set_xticks([]); ax.axvline(0, color="#888", lw=.8, ls=":")
            ax.set_yticks([band_of(t) for t in ticks]); ax.set_yticklabels([f"{t / 1000:g}" for t in ticks] if j == 0 else [])
            ax.text(-292, 30, f"/{C.ipa(nm)}/ + 母音", fontsize=10, va="top", bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
            for s_ in ("top", "right"): ax.spines[s_].set_visible(True)
            if k == 0: ax.set_title("乾淨" if si == 0 else f"加噪音（{LAB[SNR_KEYS[si]]}，實驗用的語音形噪音）", loc="left", fontsize=11.5)
            if j == 0: ax.set_ylabel("頻率 (kHz)")
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(d.T, origin="lower", aspect="auto", cmap="magma_r", vmin=0, vmax=30, extent=[-300, 250, 0, 32])
        ax.set_xticks([]); ax.axvline(0, color="#888", lw=.8, ls=":")
        ax.set_yticks([band_of(t) for t in ticks]); ax.set_yticklabels([f"{t / 1000:g}" for t in ticks] if j == 0 else [])
        for s_ in ("top", "right"): ax.spines[s_].set_visible(True)
        ax.add_patch(plt.Rectangle((-215, band_of(3300)), 195, 32 - band_of(3300) - .3, fill=False, ec=C.BLUE, lw=2))
        ax.add_patch(plt.Rectangle((-215, 0.3), 215, band_of(600), fill=False, ec=C.RED, lw=2, ls="--"))
        ax.text(-292, 22.5, "兩張圖相減（越深 = 差越多）", fontsize=9, va="top", bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
        box = dict(fc="white", ec="none", alpha=.9, pad=1.5)
        if (A_, B_) == ("d", "z"):                                   # 框上的說明只對預設的這一對成立；換別對時只畫框
            ax.text(20, 30.5, "藍框：/z/ 的高頻摩擦噪音" if j == 0 else "藍框：還露出來", color=C.BLUE, fontsize=9, va="top", bbox=box)
            ax.text(20, 4.6, "紅虛線框：/z/ 的聲帶振動" if j == 0 else "紅虛線框：被噪音蓋掉了", color=C.RED, fontsize=9, va="top", bbox=box)
        if j == 0: ax.set_ylabel("頻率 (kHz)")
        a2 = fig.add_subplot(gs[3, j])
        a2.plot(T_MS, ex_t, color=C.RED, lw=1.8); a2.set_xlim(-300, 250); a2.set_ylim(0, 24)
        a2.axvline(0, color="#888", lw=.8, ls=":"); a2.axvspan(-300, 100, color=C.RED, alpha=.06)
        a2.set_xlabel("時間 (ms，0 = 嘴巴打開進入母音)"); a2.text(-292, 22.5, f"紅底這段的差別 = {nums[si]:.1f} dB", fontsize=9.5, va="top", color=C.RED)
        if j == 0: a2.set_ylabel("每一刻的差別 (dB)")
        else: a2.set_yticklabels([])
    fig.suptitle(f"/{C.ipa(A_)}/ 對 /{C.ipa(B_)}/（說話者 {TK}）：乾淨時差 {nums[0]:.1f} dB，加 {LAB[SNR_KEYS[SI]]} 噪音後剩 {nums[SI]:.1f} dB", fontsize=12.5, y=0.925)
    fig.savefig(f_("noise_covers.png"), bbox_inches="tight"); plt.close(fig)
    print("noise_covers：", {LAB[SNR_KEYS[k]]: round(float(v), 1) for k, v in nums.items()})

    # ---- 聲學尺對 M&N 的表
    q = P[P.snr == "6.0"].set_index("pair").join(MN.rename("mn"))
    r = np.corrcoef(q.excess, q.mn)[0, 1]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.scatter(q.mn, q.excess, s=18, color=C.GREY, alpha=.7)
    for p in MARKED:
        ax.scatter(q.mn[p], q.excess[p], s=26, color=C.RED)
        ax.annotate(pp(p), (q.mn[p], q.excess[p]), xytext=(4, 3), textcoords="offset points", fontsize=9, color=C.RED)
    ax.set_xlabel("Miller & Nicely 的 d'（現在用的尺；英語母語者在白噪音裡的混淆）"); ax.set_ylabel("頻譜圖上的差別 (dB，+6 dB 語音形噪音)")
    ax.set_title(f"兩把尺不是同一把：120 對子音的相關 r = {r:.2f}（每點一對）", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(f_("vs_mn.png")); plt.close(fig)

    # ---- 非母語聽者：四等分的命中率
    g = pd.DataFrame(V["listener"]["quartiles"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, r_ in g.iterrows():
        se = np.sqrt(r_.hit * (1 - r_.hit) / r_.n)
        ax.bar(i, r_.hit, .62, color=C.BLUE, alpha=.85); ax.errorbar(i, r_.hit, 1.96 * se, color="k", lw=1, capsize=3)
        ax.text(i, .03, f"{r_.hit:.2f}\n({int(r_.k)}/{int(r_.n)})", ha="center", va="bottom", fontsize=9.5, color="white")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{r_.lo:.1f}-{r_.hi:.1f} dB" for _, r_ in g.iterrows()]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("這一題兩個音在頻譜圖上的差別（該說話者、該 SNR；四等分）"); ax.set_ylabel("命中率（答「不一樣」）")
    ax.set_title(f"非母語聽者 {a.participant}：頻譜圖上差得越多，越聽得出來（{V['listener']['n']} 題）", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f_("hit_by_quartile.png")); plt.close(fig)

    # ---- 同一批題目，三把尺
    fig, ax = plt.subplots(figsize=(7, 4.2))
    items = [("mn_dprime", "Miller & Nicely 的 d'\n（現在的尺）", C.GREY), ("features", "差幾個發音特徵", "#d35400"), ("ac_excess_talker", "頻譜圖上的差別", C.BLUE)]
    for i, (k, lab, col) in enumerate(items):
        au, ci = V["listener"][k]["auc"], V["listener"][k]["ci"]
        ax.bar(i, au - .5, .6, bottom=.5, color=col, alpha=.85); ax.errorbar(i, au, [[au - ci[0]], [ci[1] - au]], color="k", lw=1, capsize=3)
        ax.text(i, ci[1] + .012, f"{au:.2f}", ha="center", fontsize=10)
    ax.axhline(.5, color="k", lw=.8); ax.text(2.42, .492, "0.5 = 完全排不出來", fontsize=8.5, ha="right", va="top", color=C.GREY)
    ax.set_xticks(range(3)); ax.set_xticklabels([i[1] for i in items]); ax.set_ylim(.4, .9); ax.set_ylabel("排序力（答對的題比答錯的題分數高的機率）")
    ax.set_title(f"同樣 {V['listener']['n']} 題，哪把尺排得出 {a.participant} 的對錯（誤差線 = 95% 拔靴區間）", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f_("three_rulers.png")); plt.close(fig)

    # ---- 各 SNR 下 120 對的分布
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rng = np.random.default_rng(0)
    L, D_ = U["logistic"], U["distribution"]
    for i, s in enumerate(SNR_KEYS):
        e = P[P.snr == s].excess.values
        ax.plot(i + rng.uniform(-.22, .22, len(e)), e, "o", ms=3, color=C.BLUE, alpha=.45); ax.plot([i - .3, i + .3], [np.median(e)] * 2, color="k", lw=2)
    ax.axhline(L["x90"], color=C.GREEN, lw=1.2, ls="--", label=f"綠線以上：{a.participant} 命中 > .90（易）")
    ax.axhline(L["x75"], color=C.RED, lw=1.2, ls="--", label="紅線以下：命中 < .75（難）"); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{LAB[s]}\n難 {D_[s]['hard']}・中 {120 - D_[s]['hard'] - D_[s]['easy']}・易 {D_[s]['easy']}" for s in SNR_KEYS], fontsize=9)
    ax.set_ylim(0, 17.5); ax.set_ylabel("頻譜圖上的差別 (dB)")
    ax.set_title("噪音越大，120 對整批往下掉（點 = 一對，黑線 = 中位數）", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(f_("distribution_by_snr.png")); plt.close(fig)

    # ---- 差別在時間上的位置
    tc = pd.read_csv(f_("timecourse.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for s in SNR_KEYS: ax.plot(tc.t_ms, tc[s], color=COLS[s], lw=2, label=LAB[s])
    ax.axvline(0, color="#888", lw=.8, ls=":"); ax.axvspan(120, 250, color=C.GREY, alpha=.12)
    ax.text(185, ax.get_ylim()[1] * .97, "母音本體", ha="center", va="top", fontsize=9, color=C.GREY)
    ax.set_xlabel("時間 (ms，0 = 嘴巴打開進入母音)"); ax.set_ylabel("兩個音的差別 (dB，120 對平均)"); ax.legend(frameon=False, fontsize=9, title="噪音", loc="upper left")
    ax.set_title("差別集中在進入母音之前；母音本體也有，但小得多", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(f_("timecourse.png")); plt.close(fig)

    # ---- 差別在頻率上的位置
    IJ = list(itertools.combinations(range(16), 2))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    prof, W = {}, slice(0, 40)
    for si, s in enumerate(SNR_KEYS):
        X = Z[f"s{si}"]
        fl = np.zeros(32)
        if si: fl = np.mean([((X[:, :, p] - X[:, :, q])[:, :, W] ** 2).mean((0, 1, 2)) for p, q in itertools.combinations(range(NSEED), 2)], 0)
        d2 = np.mean([np.mean([((X[i, :, k] - X[j, :, (k + 1) % NSEED])[:, W] ** 2).mean((0, 1)) for k in range(NSEED)], 0) for i, j in IJ], 0)
        prof[s] = np.sqrt(np.clip(d2 - fl, 0, None)); ax.plot(CEN / 1000, prof[s], color=COLS[s], lw=2, label=LAB[s])
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([.1, .2, .5, 1, 2, 4, 7]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "4", "7"])
    ax.set_xlabel("頻率 (kHz)"); ax.set_ylabel("兩個音的差別 (dB，120 對平均)"); ax.legend(frameon=False, fontsize=9, title="噪音")
    lo, hi = CEN < 400, CEN > 3000
    keep = lambda s, k: float(np.sqrt(np.mean(prof[s][k] ** 2)) / np.sqrt(np.mean(prof["clean"][k] ** 2)))
    ax.set_title(f"語音形噪音先蓋掉低頻：+6 dB 時 0.4 kHz 以下只剩乾淨時的 {100 * keep('6.0', lo):.0f}%，3 kHz 以上剩 {100 * keep('6.0', hi):.0f}%", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f_("frequency_profile.png")); plt.close(fig)
    kept = {s: (round(keep(s, lo), 2), round(keep(s, hi), 2)) for s in SNR_KEYS[1:]}
    json.dump({"kept_below_0.4kHz_above_3kHz": kept, "r_excess6_vs_mn": float(r)}, open(f_("figs_numbers.json"), "w"), indent=1)
    print("各 SNR 剩下的比例（0.4 kHz 以下, 3 kHz 以上）：", kept); print("r(頻譜圖差別 @ +6 dB, M&N d′) =", round(float(r), 3))
    print("寫出 7 張圖到", out_d)


if __name__ == "__main__":
    main()
