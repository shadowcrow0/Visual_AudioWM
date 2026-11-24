"""聲學尺（五）：把「頻譜圖上的差別 x (dB)」換成某一位聽者「回答不一樣」的機率。

    python analyze/acoustic_scale_listener_curve.py --participant WEB01 [--compare yyc1116]

讀：analyze/out/acoustic_scale_trials_<聽者>.csv（先跑 acoustic_scale_validate.py --participant <聽者>）
    data/<聽者>_sound_perception_dprime_snr*.csv（取「相同」試的誤報率）
算：  P(答不一樣 | x) = FA + (1 - FA - λ) / (1 + exp(-(x - x50)/s))
    FA 用該 SNR「相同」試的誤報率（不擬合）；x50、s、λ 所有 SNR 共用，最大概似。
    - x90 = x50 + s·ln 9（到頂九成的點）；bootstrap 95% 區間（重抽「不同」試）
    - 每個 SNR 各自擬合一組 (x50, s, λ) 的概似比檢定：這把尺能不能讓三個 SNR 共用一條曲線
出：analyze/out/acoustic_scale_curve_<聽者>.json、_curve_<聽者>.png、（有 --compare 時）_quartiles_<聽者>_vs_<另一位>.png
對應：review/pilot_2026-11-24_WEB01_snr_pilot.md。

為什麼 FA 不放進擬合：FA 由 36 題「相同」試直接量到；放進去擬合會讓它和 x50 互相補償，曲線中點就不穩。
x 是「該題說話者」的值（同一對在不同說話者之間差很多，企劃書 §5）。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import _common as C

XCOL = "ac_excess_talker"
COLS = {0.0: "#c0392b", 6.0: "#d68910", 12.0: "#2471a3"}


def load(pid):
    f = os.path.join(C.OUT, f"acoustic_scale_trials_{pid}.csv")
    if not os.path.exists(f):
        raise SystemExit(f"找不到 {f}：先跑 python analyze/acoustic_scale_validate.py --participant {pid}")
    t = pd.read_csv(f).dropna(subset=[XCOL, "correct"])
    snrs = sorted(t.snr.unique())
    d = C.load_perception(pid, snrs)
    same = d[(d.is_same == 1) & d.key.notna()]
    fa = (1 - same.groupby("snd_snr_db").correct.mean()).to_dict()
    n_same = same.groupby("snd_snr_db").size().to_dict()
    return t, fa, n_same


def nll(p, x, y, fa):
    with np.errstate(over="ignore"):
        x50, s, lam = p[0], np.exp(p[1]), 0.3 / (1 + np.exp(-p[2]))   # λ 限制在 0-0.3
    with np.errstate(over="ignore"):                                   # 搜尋途中 s 很小時 exp 會溢位成 inf，結果仍正確（1/inf = 0）
        pr = np.clip(fa + (1 - fa - lam) / (1 + np.exp(-(x - x50) / s)), 1e-6, 1 - 1e-6)
    return -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).sum()


def fit(x, y, fa):
    best = min((minimize(nll, [a, b, -2.0], args=(x, y, fa), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
                for a in (1.0, 2.5, 4.0) for b in (-1.0, 0.0, 0.7)), key=lambda r: r.fun)
    with np.errstate(over="ignore"):
        return dict(x50=float(best.x[0]), s=float(np.exp(best.x[1])), lam=float(0.3 / (1 + np.exp(-best.x[2]))), nll=float(best.fun))


def curve(pid, n_boot, seed):
    t, fa_d, n_same = load(pid)
    x, y, snr = t[XCOL].values, t.correct.values.astype(float), t.snr.values
    fa = np.array([max(fa_d[s], 0.5 / n_same[s]) for s in snr])        # 誤報 0 時用半題，避免下限剛好是 0
    pooled = fit(x, y, fa)
    per = {float(s): fit(x[snr == s], y[snr == s], fa[snr == s]) for s in sorted(set(snr))}
    lr = 2 * (pooled["nll"] - sum(v["nll"] for v in per.values())); df = 3 * (len(per) - 1)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x)); b = fit(x[i], y[i], fa[i]); boot.append((b["x50"], b["x50"] + b["s"] * np.log(9)))
    boot = np.array(boot)
    res = dict(participant=pid, n_different=int(len(x)), hit=float(y.mean()), false_alarm={f"{k:g}": float(v) for k, v in fa_d.items()},
               x50=pooled["x50"], s=pooled["s"], lam=pooled["lam"], x90=pooled["x50"] + pooled["s"] * np.log(9),
               x50_ci95=[float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])], x90_ci95=[float(v) for v in np.percentile(boot[:, 1], [2.5, 97.5])],
               per_snr={f"{k:g}": dict(v, x90=v["x50"] + v["s"] * np.log(9), n=int((snr == k).sum())) for k, v in per.items()},
               separate_vs_pooled=dict(lr=float(lr), df=df, p=float(chi2.sf(lr, df)) if df > 0 else None),      # 單一 SNR 的場次沒有「各自一條」可比
               x_range=[float(x.min()), float(x.max())])
    return res, t, fa_d


def quartiles(t):
    q = pd.qcut(t[XCOL], 4, labels=False)
    return t.groupby(q).agg(lo=(XCOL, "min"), hi=(XCOL, "max"), hit=("correct", "mean"), n=("correct", "size"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="聲學尺（五）：某位聽者的換算曲線。")
    ap.add_argument("--participant", default="WEB01"); ap.add_argument("--compare", default=None, help="另一位聽者，畫四等分命中率的對照")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = C.out_dir(a.out); plt = C.setup_plot()
    res, t, fa_d = curve(a.participant, a.boot, a.seed)
    json.dump(res, open(os.path.join(out, f"acoustic_scale_curve_{a.participant}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{a.participant}：「不同」試 {res['n_different']} 題，命中 {res['hit']:.2f}；誤報 " + " / ".join(f"{k} dB {v:.2f}" for k, v in res["false_alarm"].items()))
    print(f"  曲線中點 x50 = {res['x50']:.2f} dB [{res['x50_ci95'][0]:.2f}, {res['x50_ci95'][1]:.2f}]；到頂九成 x90 = {res['x90']:.2f} dB [{res['x90_ci95'][0]:.2f}, {res['x90_ci95'][1]:.2f}]；s = {res['s']:.2f}；λ = {res['lam']:.3f}")
    for k, v in res["per_snr"].items(): print(f"  SNR {float(k):+g} dB 單獨擬合：x50 {v['x50']:.2f}  x90 {v['x90']:.2f}  (n = {v['n']})")
    sv = res["separate_vs_pooled"]; multi = sv["p"] is not None
    if multi: print(f"  各 SNR 各自一條 對 共用一條：概似比 {sv['lr']:.2f}，df {sv['df']}，p = {sv['p']:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.8)); xs = np.linspace(0, max(10.5, res["x_range"][1]), 300)
    for s in sorted(fa_d):
        fa = max(fa_d[s], 1e-3); ax.plot(xs, fa + (1 - fa - res["lam"]) / (1 + np.exp(-(xs - res["x50"]) / res["s"])), color=COLS.get(s, "k"), lw=2.2, label=f"{s:+g} dB（誤報 {fa_d[s]:.2f}）")
        sub = t[t.snr == s]; g = sub.groupby(pd.qcut(sub[XCOL], 4, labels=False, duplicates="drop")).agg(x=(XCOL, "mean"), p=("correct", "mean"), n=("correct", "size"))
        ax.errorbar(g.x, g.p, np.sqrt(g.p * (1 - g.p) / g.n), fmt="o", color=COLS.get(s, "k"), ms=7, capsize=2.5, lw=1)
        ax.plot(-0.25 - 0.12 * list(sorted(fa_d)).index(s), fa_d[s], "s", mfc="white", mec=COLS.get(s, "k"), ms=7)
    for v, lab, ha, dx in ((res["x50"], f"曲線中點 {res['x50']:.1f} dB", "right", -0.08), (res["x90"], f"到頂九成 {res['x90']:.1f} dB", "left", 0.08)):
        ax.axvline(v, color="#7f8c8d", ls=":", lw=1.2); ax.text(v + dx, 0.03, lab, color="#7f8c8d", fontsize=9, ha=ha)   # 兩個標籤各放一側，兩條線很近時才不會疊在一起
    ax.set_xlabel("頻譜圖上的差別 x (dB，該題的說話者)"); ax.set_ylabel("回答「不一樣」的比例"); ax.set_ylim(0, 1.03); ax.set_xlim(-0.7, xs[-1])
    ax.legend(frameon=False, fontsize=9, loc="lower right", title="三條線共用中點和斜率，只有誤報不同" if multi else None)
    head = f"{a.participant}：三個 SNR 共用一條換算曲線（各自一條 p = {sv['p']:.2f}）" if multi else f"{a.participant}：換算曲線（單一 SNR）"
    ax.set_title(head + f"\n（點 = 每個 SNR 的「不同」題依 x 分四組 ± 1 SE；空心方塊 = 「相同」題的誤報）", loc="left", fontsize=10.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"acoustic_scale_curve_{a.participant}.png")); plt.close(fig)

    q = quartiles(t); print("  x 四等分的命中率：" + " / ".join(f"{h:.2f}" for h in q.hit))
    if a.compare:
        t2, _, _ = load(a.compare); q2 = quartiles(t2)
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for k, (pid, qq, col) in enumerate(((a.compare, q2, "#7f8c8d"), (a.participant, q, "#2471a3"))):
            xx = np.arange(4) + (k - 0.5) * 0.36; ax.bar(xx, qq.hit, 0.32, color=col, label=f"{pid}（{int(qq.n.sum())} 題）")
            for u, h, n in zip(xx, qq.hit, qq.n): ax.text(u, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(["最小的四分之一", "第二", "第三", "最大的四分之一"]); ax.set_ylim(0, 1.1); ax.set_ylabel("命中率（「不同」題答對）")
        ax.set_xlabel("各自的題目依頻譜圖差別排序後分四組"); ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.set_title("頻譜圖差別越大，兩位聽者都越容易答對", loc="left", fontsize=10.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"acoustic_scale_quartiles_{a.participant}_vs_{a.compare}.png")); plt.close(fig)
    print("寫出：", out)
