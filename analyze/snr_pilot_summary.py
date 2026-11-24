"""SNR pilot：每個 SNR 的命中、誤報、50% / 90% 點。

讀：data/<id>_sound_perception_dprime_snr<dB>.csv（sound_perception_EN.py --snr 的輸出；每個 SNR 108 試 = 72 不同 + 36 相同）
算：每個 SNR 的無反應數、誤報率（相同試答「不一樣」）、前後半的誤報與命中、六個計畫層級的命中率與觀察 d'、
    以及 P(答不一樣) 對「實際 d'」的 logistic 擬合（相同試放在 d' = 0），由它解出 50% 點與 90% 點。
出：analyze/out/snr_pilot_summary_<id>.csv（每 SNR 一列）、snr_pilot_levels_<id>.csv（每 SNR × 層級一列）、
    snr_pilot_psychometric_<id>.png、snr_pilot_halves_<id>.png。
對應：review/pilot_2026-11-18_snr_pilot.md「結果」表與讀法 3、5；review/figures/fig19、fig20。

為什麼用 logistic 而不是 LNRM：這裡只要知道曲線在尺上的哪裡（選 SNR 用），不需要 RT 模型；
兩個參數的 logistic 用最大概似直接解，沒有取樣誤差，重跑數字不會動。
為什麼 x 用「實際 d'」（欄位 x）而不是計畫層級（x_planned）：每一試會把計畫層級吸附到該目標子音實際有的 foil，
同一級裡的 d' 並不相同。
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import _common as C


def summarise(d):
    same, diff = d[d.is_same == 1], d[d.is_same == 0]
    half = len(d) // 2
    h1 = lambda t: t[t.trial <= half].said_diff.mean()            # 前半
    h2 = lambda t: t[t.trial > half].said_diff.mean()             # 後半
    fa = same.said_diff.mean()
    lv = diff.groupby("x_planned").agg(n=("said_diff", "count"), hit=("said_diff", "mean"), x=("x", "mean"), rt=("rt", "median")).reset_index()
    z = lambda p, n: norm.ppf(min(max(p, 0.5 / n), 1 - 0.5 / n))            # 0 或 1 的比例用 1/(2n) 修正，d' 才算得出來
    lv["dprime_obs"] = [z(h, n) - z(fa, same.said_diff.count()) for h, n in zip(lv.hit, lv.n)]
    xx = np.concatenate([diff.x.values, np.zeros(len(same))])
    yy = np.concatenate([diff.said_diff.values, same.said_diff.values])
    ok = ~np.isnan(yy)
    xx, yy = xx[ok], yy[ok]

    def nll(p):
        q = np.clip(1 / (1 + np.exp(-(p[0] + p[1] * xx))), 1e-6, 1 - 1e-6)
        return -np.sum(yy * np.log(q) + (1 - yy) * np.log(1 - q))
    a, b = minimize(nll, [0, 1]).x
    row = dict(n=len(d), no_response=int(d.key.isna().sum()), fa=fa, fa_first=h1(same), fa_second=h2(same),
               hit=diff.said_diff.mean(), hit_first=h1(diff), hit_second=h2(diff),
               x50=-a / b, x90=(np.log(9) - a) / b, slope=b, intercept=a,
               rt_median=d.rt.median(), rt_same=same.rt.median(), rt_diff=diff.rt.median())
    return row, lv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 6, 12])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    import adaptive_vawm_EN as A
    d = C.load_perception(a.participant, a.snrs)
    rows, levels = [], []
    for snr, g in d.groupby("snr"):
        r, lv = summarise(g)
        rows.append(dict(snr=snr, **r)); levels.append(lv.assign(snr=snr))
        print(f"\n=== SNR {snr:+g} dB  n={r['n']}  無反應 {r['no_response']}  誤報 {r['fa']:.2f}（前半 {r['fa_first']:.2f}、後半 {r['fa_second']:.2f}）"
              f"  命中 前半/後半 {r['hit_first']:.2f}/{r['hit_second']:.2f}  RT 中位數 {r['rt_median']:.2f} s（相同 {r['rt_same']:.2f}、不同 {r['rt_diff']:.2f}）")
        print(lv.round(2).to_string(index=False))
        print(f"   logistic（對實際 d'）：50% 點 {r['x50']:.2f}，90% 點 {r['x90']:.2f}，斜率 {r['slope']:.2f} / d'")
    S, L = pd.DataFrame(rows), pd.concat(levels)
    S.to_csv(os.path.join(out, f"snr_pilot_summary_{a.participant}.csv"), index=False)
    L.to_csv(os.path.join(out, f"snr_pilot_levels_{a.participant}.csv"), index=False)

    plt = C.setup_plot()
    cols = dict(zip(sorted(S.snr), (C.RED, C.ORANGE, C.BLUE, C.GREEN, C.GREY)))
    g = np.linspace(0, 5, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in rows:
        lv = L[L.snr == r["snr"]]; c = cols[r["snr"]]
        ax.plot(g, 1 / (1 + np.exp(-(r["intercept"] + r["slope"] * g))), color=c, lw=2,
                label=f"{r['snr']:+g} dB：誤報 {r['fa']:.2f}，50% 點 d' {r['x50']:.1f}，90% 點 {r['x90']:.1f}")
        ax.errorbar(lv.x, lv.hit, yerr=np.sqrt(lv.hit * (1 - lv.hit) / lv.n + 1e-9), fmt="o", color=c, ms=6, capsize=3)
        ax.plot(0, r["fa"], "s", color=c, ms=7)
    for y in (0.5, 0.9):
        ax.axhline(y, color="k", ls=":", lw=.8)
    for x in A.LEVELS_AUDIO:
        ax.plot(x, -0.04, "^", color="k", ms=5, clip_on=False)
    ax.set_xlim(-0.1, 5); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("這一對在乾淨尺上的 d'（實際用的 foil；d' = 0 的方塊 = 相同試）"); ax.set_ylabel("P(回答「不一樣」)")
    ax.set_title(f"SNR pilot，聽者 {a.participant}（三角形 = LEVELS_AUDIO）", loc="left", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"snr_pilot_psychometric_{a.participant}.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w, xs = 0.18, np.arange(len(rows))
    for i, (k, lab, col, hatch) in enumerate((("fa_first", "誤報，前半", C.RED, ""), ("fa_second", "誤報，後半", C.RED, "//"),
                                              ("hit_first", "命中，前半", C.BLUE, ""), ("hit_second", "命中，後半", C.BLUE, "//"))):
        ax.bar(xs + (i - 1.5) * w, [r[k] for r in rows], w, color=col, alpha=.5 + .3 * (i % 2), hatch=hatch, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r['snr']:+g} dB" for r in rows]); ax.set_ylim(0, 1); ax.set_ylabel("回答「不一樣」的比例")
    ax.set_title(f"{a.participant}：前半與後半的誤報、命中（有沒有學會忽略噪音）", loc="left", fontsize=10); ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"snr_pilot_halves_{a.participant}.png")); plt.close(fig)
    print(f"\n寫出：{out}/snr_pilot_summary_{a.participant}.csv、snr_pilot_levels_{a.participant}.csv、兩張 png")


if __name__ == "__main__":
    main()
