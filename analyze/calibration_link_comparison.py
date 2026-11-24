"""同一份校準資料，換連結函數與有沒有判準 b0，50% 點和到頂點會差多少。需要 PyMC + nutpie。

讀：data/<id>_simple_calib_trials.csv（一個或多個）
算：每個維度擬合七個模型，全部是同一個 LNRM 賽跑似然，只有 d(x) 的形狀不同：
        quadratic、linear、ogival      —— 各做「沒有 b0」（Houpt 原版：d(0) = 0）與「有 b0」（d_eff = b0 + d(x)）
        satexp + b0                    —— 飽和指數 d = b0 + L·(1 − exp(−x/τ))（試過、沒採用）
    每個模型印：發散數、b0、P(α < 0)（斜率後驗跨零的比例）、50% 點（d_eff = 0）、90% 點（P(不一樣) = .9，
    即 d_eff = Φ⁻¹(.9)·σ/√2）、ogival 另外印到頂點（logistic = .9）。點估計 = 有解 draw 的中位數與 5-95 百分位，
    並印無解比例。
出：analyze/out/calibration_link_comparison.csv。
對應：DECISIONS 25「為什麼」的 1-3 點（沒有 b0 時斜率跨零；50% 點對連結函數不敏感；90% 正確率在 LNRM 裡常常到不了）。

為什麼這樣比：EN01 的 H = L = 2 有兩個獨立的病因 —— (a) 模型假設強度 0 時答對率 50%，same/different 作業的地板卻是誤報率；
(b) 二次式的反解是比值型的量。這支程式把 (a) 單獨拿出來看：同一個連結函數，加不加 b0，斜率的後驗差多少。
quadratic / linear / ogival 用 adaptiveSFT 的 fit_lnrm_v2（跟實驗同一個建模函式）；satexp 不在套件裡，在這裡用同一個
race_logp 另外建。DECISIONS 25 當時的探索用的先驗略有不同（ogival 的 L ~ HalfNormal(3)），所以 ogival 的 50% 點
這裡會跟定案管線（calibration_refit.py）一致，跟 DECISIONS 25 第 2 點寫的 27 / 16 差一兩個單位。
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import _common as C
import adaptive_vawm_EN as A


def draws(tr):
    return {k: tr.posterior[k].values.ravel() for k in tr.posterior.data_vars}


def fit_satexp(x, rt, c, kw):
    import pymc as pm
    from adaptivesft.models_v2 import race_logp, race_random
    with pm.Model():
        mu = pm.Normal("mu", 0, 1); varZ = pm.InverseGamma("varZ", 1, 0.1); psi = pm.Uniform("psi", 0, rt.min())
        b0 = pm.Normal("b0", 0, 2); L = pm.HalfNormal("L", 3); tau = pm.HalfNormal("tau", 5)
        pm.CustomDist("obs", mu, b0 + L * (1 - pm.math.exp(-x / tau)), varZ, psi, c, logp=race_logp, random=race_random, observed=rt)
        return pm.sample(progressbar=False, initvals=dict(mu=-0.5, varZ=0.5, psi=0.3 * rt.min(), b0=-0.5, L=1.5, tau=2.0), **kw)


def solve(link, P, t):
    """每個 draw 解 d_link(x) = t（t 可以是向量），回傳模型單位的 x；無解 = NaN。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        if link == "linear":
            return t / P["alpha"]
        if link == "quadratic":                       # 取從 0 往上第一次碰到 t 的那個根
            a, a2 = P["alpha"], P["alpha2"]; disc = a * a + 4 * a2 * t
            r1, r2 = (-a + np.sqrt(disc)) / (2 * a2), (-a - np.sqrt(disc)) / (2 * a2)
            lo = np.where((r1 > 0) & ((r1 < r2) | (r2 <= 0)), r1, r2)
            return np.where((disc >= 0) & (lo > 0), lo, np.nan)
        if link == "ogival":
            frac = t / (0.5 * P["L"])
            return np.where((frac > 0) & (frac < 1), P["midpoint"] + np.log(frac / (1 - frac)) / P["slope"], np.nan)
        if link == "satexp":
            frac = t / P["L"]
            return np.where((frac > 0) & (frac < 1), -P["tau"] * np.log(1 - frac), np.nan)


def summ(v, dim):
    ok = np.isfinite(v) & (v > 0)
    if not ok.any():
        return dict(median=np.nan, lo=np.nan, hi=np.nan, unsolved=1.0)
    u = A.from_model_units(dim, v[ok])
    return dict(median=np.median(u), lo=np.percentile(u, 5), hi=np.percentile(u, 95), unsolved=1 - ok.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calib_csv", nargs="*", default=["data/EN01_s001_2026-11-18_10h16.05_simple_calib_trials.csv", "data/yyc1110_simple_calib_trials.csv"])
    ap.add_argument("--dims", nargs="+", default=["colour"], choices=["colour", "audio"])
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--tune", type=int, default=1500); ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    A.load_full_package()
    from adaptivesft.models import make_data
    from adaptivesft.models_v2 import fit_lnrm_v2
    kw = dict(draws=a.draws, tune=a.tune, chains=a.chains, random_seed=a.seed, nuts_sampler="nutpie", target_accept=0.95)
    rows = []
    for f in a.calib_csv:
        tag = os.path.basename(f).replace("_simple_calib_trials.csv", "")
        for dim in a.dims:
            _, diff, _ = C.load_calib_trials(f)
            s = diff[diff.dim == dim]
            x = np.array([A.to_model_units(dim, v) for v in s.x]); rt = s.rt.values; c = s.correct.values.astype(int)
            print(f"\n=== {tag} / {dim}   n = {len(s)}   各級 P(不一樣)： " + " ".join(f"{k:g}:{v:.1f}" for k, v in s.groupby("x_planned").correct.mean().items()))
            for link, bias in (("quadratic", False), ("quadratic", True), ("linear", False), ("linear", True), ("ogival", False), ("ogival", True), ("satexp", True)):
                if link == "satexp":
                    tr = fit_satexp(x, rt, c, kw)
                else:
                    tr = fit_lnrm_v2(make_data(x, rt, c), link=link, bias=bias, **(dict(L="estimate") if link == "ogival" else {}), **kw)
                P = draws(tr); b0 = P.get("b0", np.zeros_like(P["mu"])); t90 = norm.ppf(0.9) * P["varZ"] / np.sqrt(2)
                x50 = summ(solve(link, P, -b0), dim) if bias else dict(median=0.0, lo=0.0, hi=0.0, unsolved=0.0)      # 沒有 b0：d(0) = 0，50% 點被模型釘在 0
                x90 = summ(solve(link, P, t90 - b0), dim)
                row = dict(file=tag, dim=dim, link=link, b0=bias, divergences=int(tr.sample_stats["diverging"].sum()), b0_median=float(np.median(b0)) if bias else np.nan,
                           p_alpha_neg=float(np.mean(P["alpha"] < 0)) if "alpha" in P else np.nan, sigma=float(np.median(P["varZ"])),
                           x50=x50["median"], x50_lo=x50["lo"], x50_hi=x50["hi"], x50_unsolved=x50["unsolved"],
                           x90=x90["median"], x90_lo=x90["lo"], x90_hi=x90["hi"], x90_unsolved=x90["unsolved"])
                line = (f"  {link:9s} b0 {'有' if bias else '無'}  發散 {row['divergences']:2d}" + (f"  b0 {row['b0_median']:5.2f}" if bias else "          ")
                        + (f"  P(α<0) {row['p_alpha_neg']:.2f}" if "alpha" in P else "             ")
                        + f"  | 50% 點 {row['x50']:5.1f} [{row['x50_lo']:5.1f}, {row['x50_hi']:5.1f}] 無解 {row['x50_unsolved']:.0%}"
                        + f"  | 90% 點 {row['x90']:5.1f} [{row['x90_lo']:5.1f}, {row['x90_hi']:5.1f}] 無解 {row['x90_unsolved']:.0%}")
                if link == "ogival":
                    pl = summ(P["midpoint"] + np.log(9) / P["slope"], dim); row.update(plateau=pl["median"], plateau_lo=pl["lo"], plateau_hi=pl["hi"])
                    line += f"  | 到頂點 {pl['median']:5.1f} [{pl['lo']:5.1f}, {pl['hi']:5.1f}]"
                rows.append(row); print(line)
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibration_link_comparison.csv"), index=False)
    print(f"\n寫出：{out}/calibration_link_comparison.csv")


if __name__ == "__main__":
    main()
