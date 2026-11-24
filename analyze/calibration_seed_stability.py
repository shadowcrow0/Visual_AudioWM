"""舊校準規則（二次式、反解漂移差目標、取後驗平均）為什麼會給出負的 H：同一份資料換取樣種子看平均與中位數。需要 PyMC + nutpie。

讀：data/<id>_simple_calib_trials.csv（預設 EN01 第一場的顏色）
算：對每個種子用舊管線擬合（fit_lnrm_v2(link="quadratic")，沒有 b0），再用 adaptivesft.salience.find_salience
    （alpha2_rule="filter"，Houpt 的 R 寫法：只留 α₂ < 0 的 draw、無解的丟掉）反解 H（漂移差 = H_TARG_DEFAULT）與 L（L_TARG_DEFAULT）。
    印每個種子：α₂ < 0 的比例、有解的 draw 數、其中負根的個數、有解 draw 的平均與中位數。
出：analyze/out/calibration_seed_stability.csv、calibration_seed_stability.png。
對應：DECISIONS 25「為什麼」第 4 點（點估計改用中位數）；2026-11-18 對「-994 是怎麼來的」的說明；review/paper 的 fig10。

為什麼平均會爆：反解 x = (−α/α₂ − √((α/α₂)² + 2·目標/α₂)) / 2 是比值型的量，α₂ 的後驗貼著 0，
少數 draw 會給出 ±幾千、幾萬；平均被它們拉走（符號都可能翻），中位數不受影響。
實驗當場記在 json 裡的 H = 2 是「平均是負的 → clip 到最低級」的結果；L 再被舊規則 low = lo_lim 砍到同一格。
這裡的目標值與單位換算都從 adaptive_vawm_EN 讀（H_TARG_DEFAULT、L_TARG_DEFAULT、to_model_units），不另抄。
"""
import argparse
import os

import numpy as np
import pandas as pd

import _common as C
import adaptive_vawm_EN as A


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calib_csv", nargs="?", default="data/EN01_s001_2026-11-18_10h16.05_simple_calib_trials.csv")
    ap.add_argument("--dim", default="colour", choices=["colour", "audio"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 18)))
    ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--tune", type=int, default=1000); ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    A.load_full_package()
    from adaptivesft.models import make_data
    from adaptivesft.models_v2 import fit_lnrm_v2
    from adaptivesft.salience import find_salience
    _, diff, _ = C.load_calib_trials(a.calib_csv)
    s = diff[diff.dim == a.dim]
    data = make_data([A.to_model_units(a.dim, v) for v in s.x], list(s.rt), list(s.correct))
    rows = []
    for seed in a.seeds:
        tr = fit_lnrm_v2(data, link="quadratic", draws=a.draws, tune=a.tune, chains=a.chains, nuts_sampler="nutpie", random_seed=seed)
        res = find_salience(tr, h_targ=A.H_TARG_DEFAULT, l_targ=A.L_TARG_DEFAULT, alpha2_rule="filter")
        row = dict(seed=seed, p_alpha2_neg=float(np.mean(tr.posterior["alpha2"].values < 0)), p_alpha_neg=float(np.mean(tr.posterior["alpha"].values < 0)))
        for name, key in (("H", "high"), ("L", "low")):
            v = np.asarray(res[key]["draws"], dtype=float); v = A.from_model_units(a.dim, v[np.isfinite(v)])
            row.update({f"{name}_n": len(v), f"{name}_neg": int((v < 0).sum()), f"{name}_mean": v.mean() if len(v) else np.nan,
                        f"{name}_median": np.median(v) if len(v) else np.nan})
        rows.append(row)
        print(f"seed {seed}: P(α₂<0) {row['p_alpha2_neg']:.2f}  P(α<0) {row['p_alpha_neg']:.2f} | H 有解 {row['H_n']:4d}（負根 {row['H_neg']:3d}） 平均 {row['H_mean']:9.1f}  中位數 {row['H_median']:7.1f}"
              f" | L 有解 {row['L_n']:4d}（負根 {row['L_neg']:3d}） 平均 {row['L_mean']:8.1f}  中位數 {row['L_median']:6.1f}")
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, "calibration_seed_stability.csv"), index=False)

    plt = C.setup_plot()
    lv = A.LEVELS_COLOUR if a.dim == "colour" else A.LEVELS_AUDIO
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(T.seed, T.H_median, "o-", color=C.BLUE, label="有解 draw 的中位數"); ax.plot(T.seed, T.H_mean, "s--", color=C.RED, label="有解 draw 的平均（舊程式用的）")
    for r in T.itertuples():
        ax.annotate(f"{r.H_neg} 個負根 / {r.H_n}", (r.seed, r.H_mean), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=C.RED)
    ax.axhspan(min(lv), max(lv), color="green", alpha=.08)
    ax.set_xticks(list(T.seed))
    ax.set_yscale("symlog", linthresh=100)
    ticks = [-10000, -1000, -100, 0, 100, 1000]; ax.set_yticks(ticks); ax.set_yticklabels([str(v) for v in ticks])     # 自己給 ASCII 負號：symlog 的預設刻度會用 U+2212，中文字型沒有
    ax.set_xlabel("取樣種子（同一份資料、同一個模型）"); ax.set_ylabel(f"反解出來的 H（{'ΔE' if a.dim == 'colour' else 'd′'}，symlog；綠帶 = 可呈現的層級範圍）")
    ax.set_title(f"同樣 {len(s)} 試、{len(T)} 個種子：中位數穩定，平均會翻號", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "calibration_seed_stability.png")); plt.close(fig)
    print(f"\n寫出：{out}/calibration_seed_stability.csv、.png")


if __name__ == "__main__":
    main()
