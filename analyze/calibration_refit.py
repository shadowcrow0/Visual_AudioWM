"""用現行管線（ogival + b0，L = 50% 點、H = 到頂 90%）重算校準。需要 PyMC + nutpie。

讀：data/<id>_simple_calib_trials.csv（一個或多個）
算：對每個維度呼叫 adaptive_vawm_EN.fit_points —— 跟實驗機上 SFT_WM_nochange_EN.fit_calibration 走的是同一個函式、
    同一組層級（LEVELS_COLOUR / LEVELS_AUDIO）、同一組取樣設定（FIT_KW，用 ast 從 SFT_WM_nochange_EN.py 讀出來）。
    印出 L、H、in_range、50% 點與到頂點的中位數與 90% 區間、無解比例、b0 / slope / midpoint / A / σ、發散數、警告，
    以及模型在各級預測的 P(不一樣) 對照資料。
出：analyze/out/calibration_refit_<檔名>.json（格式同實驗寫的 _simple_calib.json，多一個 false_alarm）。
對應：DECISIONS 25「結果」（EN01 顏色 L 25.2 / H 36.3，yyc1110 L 15.2 / H 26.9）；
      review/pilot_2026-11-24_EN01_s002.md §2「新版重算」欄（L 18.7 / H 41.7）。

為什麼不自己建模型：重算的目的是回答「如果當時實驗機跑的是現行版，會給什麼 H / L」，所以必須是同一段程式。
MCMC 有取樣誤差：換 seed，L / H 會動零點幾到一兩個單位（區間本來就寬）；筆記裡的數字沒有記 seed，
所以這裡重跑只會對到同一個量級，不會逐位數相同。--seed 固定之後本程式自己可重現。
舊資料是用舊層級收的（2/6/12/25/50/90），但 in_range 與 clip 用的是現行層級 —— 這正是「現行版會給什麼」的意思。
模型預測：P(不一樣) = Φ(√2 · d_eff / σ)，d_eff = b0 + A · logistic(slope · (u − midpoint))，u = 模型單位（顏色 ΔE/10）。
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import norm

import _common as C
import adaptive_vawm_EN as A


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calib_csv", nargs="*", default=["data/EN01_s001_2026-11-18_10h16.05_simple_calib_trials.csv"])
    ap.add_argument("--dims", nargs="+", default=["colour", "audio"], choices=["colour", "audio"])
    ap.add_argument("--seed", type=int, default=42, help="取樣種子（fit_lnrm_v2 的預設也是 42）")
    ap.add_argument("--draws", type=int, default=None, help="只在快速檢查時用；不給就用實驗的 FIT_KW")
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    fit_kw = dict(C.experiment_fit_kw(), random_seed=a.seed)
    for k in ("draws", "tune", "chains"):
        if getattr(a, k) is not None:
            fit_kw[k] = getattr(a, k)
    print("取樣設定：", fit_kw)
    for f in a.calib_csv:
        tag = os.path.basename(f).replace("_simple_calib_trials.csv", "")
        res = {"info": {"source": os.path.basename(f), "method": f"lnrm_{A.CALIB_LINK}{'+b0' if A.CALIB_BIAS else ''}",
                        "rule": f"L = 50% point, H = plateau {A.CALIB_PLATEAU}", "fit_kw": fit_kw}}
        for dim in a.dims:
            _, diff, same = C.load_calib_trials(f)
            s, same = diff[diff.dim == dim], same[same.dim == dim]
            levels = A.LEVELS_COLOUR if dim == "colour" else A.LEVELS_AUDIO
            r = A.fit_points(list(s.x), list(s.rt), list(s.correct), dim, levels, fit_kw)
            r["false_alarm"] = float((same.key == "n").mean())
            res[dim] = r
            e, p = r["extra"], r["extra"]["params"]
            pdiff = lambda x: norm.cdf(np.sqrt(2) * (p["b0"] + p["A"] / (1 + np.exp(-p["slope"] * (A.to_model_units(dim, x) - p["midpoint"])))) / e["varZ"])
            print(f"\n[{tag} / {dim}]  L = {r['low']:.2f}   H = {r['high']:.2f}   in_range = {r['in_range']}   誤報 = {r['false_alarm']:.2f}   n = {len(s)}")
            print(f"   50% 點   {e['x50_median']:7.2f}   90% 區間 [{e['x50_ci90'][0]:.2f}, {e['x50_ci90'][1]:.2f}]   無解 {e['x50_unsolved']:.0%}")
            print(f"   到頂點   {e['plateau_median']:7.2f}   90% 區間 [{e['plateau_ci90'][0]:.2f}, {e['plateau_ci90'][1]:.2f}]   無解 {e['plateau_unsolved']:.0%}")
            print(f"   b0 {p['b0']:.2f}   slope {p['slope']:.2f}   midpoint {A.from_model_units(dim, p['midpoint']):.1f}   天花板 A {p['A']:.2f}   σ {e['varZ']:.2f}   發散 {e['divergences']}")
            for w in r["warnings"]:
                print("   警告：", w)
            print("   模型 P(不一樣) 對資料：")
            for x, g in s.groupby("x_planned"):
                print(f"      x = {x:5g}   模型 {pdiff(g.x.mean()):.2f}   資料 {g.correct.mean():.2f}（{int(g.correct.sum())}/{len(g)}）")
            print(f"   在算出來的 L / H 上，模型的 P(不一樣) = {pdiff(r['low']):.2f} / {pdiff(r['high']):.2f}")
        dst = os.path.join(out, f"calibration_refit_{tag}.json")
        json.dump(res, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
        print(f"\n寫出：{dst}")


if __name__ == "__main__":
    main()
