"""聲學尺（六）：企劃書 §6 的逐人判準 C1、C2、C5，用真實資料算。

    python analyze/acoustic_scale_criteria.py --participants WEB01 yyc1116 [--fit-points]

讀：analyze/out/acoustic_scale_trials_<聽者>.csv（先跑 acoustic_scale_validate.py --participant <聽者>）
    data/<聽者>_sound_perception_dprime_snr*.csv（「相同」試的誤報）
算（定義照 review/proposal_2026-11-24_language_independent_sound_scale.md §6；門檻是那裡事先寫的，這裡不調）：
    C1 排序力   逐題 AUC（聲學尺，該題說話者的值）≥ .65；同時列出 M&N d′ 與特徵數的 AUC 當對照
    C2 單一軸   邏輯迴歸 命中 ~ 差別，再加 SNR；AIC 改善（AIC(差別) − AIC(差別 + SNR)）< 2 算通過
    C5 可校準   (a) 換算曲線（acoustic_scale_listener_curve.py 的模型）：P = .5 的點與到頂九成的點是否都在測過的 x 範圍內
                (b) --fit-points：實驗自己的校準 adaptive_vawm_EN.fit_points（LNRM、ogival + b0，用對錯與 RT），
                    x 換成聲學差別，取樣設定同實驗（_common.experiment_fit_kw）。需要 PyMC。
    C3、C4 要兩個語言族群，單一聽者算不了，這裡不算。
出：analyze/out/acoustic_scale_criteria.json；印出每人一列的表。
對應：review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md §5。

注意：
- 判準原本是對「每組 6 人」定的（至少 5 人通過）；這裡只是逐人試算，不是判定。
- fit_points 只吃「不同」試（SFT_WM_nochange_EN.py 傳進去的就是這樣），而且聲音軸沒有單位換算（AUDIO_SHIFT = 0），
  先驗是照 d′ 的 0.4–4.5 調的。聲學差別的範圍約 0.5–11 dB，所以傳進去之前除以 --scale（預設 2），結果再乘回來。
  這個縮放是這支程式的選擇，不是實驗的設定；若正式採用，要在 adaptive_vawm_EN 裡定一個聲音軸的單位換算。
- (a) 的「P = .5 的點」用該聽者三個 SNR 合併的誤報率；誤報高時這個點會掉到曲線中點以下甚至不存在。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _common as C
from acoustic_scale_listener_curve import XCOL, curve, load
from acoustic_scale_validate import auc, boot_auc, logistic

C1_MIN, C2_MAX = 0.65, 2.0        # 企劃書 §6 的門檻


def c1_c2(t):
    y = t.correct.values.astype(float)
    res = {"n": int(len(t)), "hit": float(y.mean())}
    for name, col in (("acoustic", XCOL), ("mn", "mn_dprime"), ("features", "features")):
        res[f"auc_{name}"] = auc(t[col].values, y)
        res[f"auc_{name}_ci95"] = boot_auc(t[col].values, y)
    _, aic1 = logistic(t[[XCOL]].values, y)
    if t.snr.nunique() > 1:
        b2, aic2 = logistic(t[[XCOL, "snr"]].values, y)
        res.update(aic_x=aic1, aic_x_snr=aic2, aic_gain_from_snr=aic1 - aic2, snr_coef=float(b2[2]), c2_pass=bool(aic1 - aic2 < C2_MAX))
    else:                                          # 單一 SNR 的場次（online/calib_pilot）：C2 問的是「加 SNR 有沒有幫助」，沒有第二個 SNR 就算不了
        res.update(aic_x=aic1, aic_x_snr=None, aic_gain_from_snr=None, snr_coef=None, c2_pass=None)
    res["c1_pass"] = bool(res["auc_acoustic"] >= C1_MIN)
    return res


def c5_curve(pid, t):
    """換算曲線的兩個點。P = .5 的點：FA + (1 − FA − λ)·logistic((x − x50)/s) = .5。"""
    r, _, fa_d = curve(pid, n_boot=200, seed=1)
    d = C.load_perception(pid, sorted(t.snr.unique()))
    same = d[(d.is_same == 1) & d.key.notna()]
    fa = float(1 - same.correct.mean())
    q = (0.5 - fa) / (1 - fa - r["lam"])
    x_half = r["x50"] + r["s"] * np.log(q / (1 - q)) if 0 < q < 1 else float("nan")
    lo, hi = float(t[XCOL].min()), float(t[XCOL].max())
    ok = bool(np.isfinite(x_half) and lo <= x_half < r["x90"] <= hi)
    return dict(false_alarm_pooled=fa, x_half=float(x_half), x50_midpoint=r["x50"], x90=r["x90"], x90_ci95=r["x90_ci95"],
                tested_range=[lo, hi], c5a_pass=ok)


def c5_fit_points(pid, t, scale, seed):
    import adaptive_vawm_EN as A
    d = t[t.rt.between(*C.RT_WINDOW)]
    kw = dict(C.experiment_fit_kw(), random_seed=seed)
    lv = [float(d[XCOL].min() / scale), float(d[XCOL].max() / scale)]
    r = A.fit_points(list(d[XCOL] / scale), list(d.rt), list(d.correct), "audio", lv, kw)
    e = r["extra"]
    return dict(n_used=int(len(d)), L=r["low"] * scale, H=r["high"] * scale, in_range=bool(r["in_range"]),
                x50=e["x50_median"] * scale, x50_ci90=[v * scale for v in e["x50_ci90"]], x50_unsolved=float(e["x50_unsolved"]),
                plateau=e["plateau_median"] * scale, plateau_ci90=[v * scale for v in e["plateau_ci90"]],
                plateau_unsolved=float(e["plateau_unsolved"]), divergences=int(e["divergences"]), warnings=list(r["warnings"]),
                range_db=[lv[0] * scale, lv[1] * scale], scale=scale, seed=seed, c5b_pass=bool(r["in_range"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="聲學尺（六）：企劃書 §6 的逐人判準 C1、C2、C5。")
    ap.add_argument("--participants", nargs="+", default=["WEB01", "yyc1116"])
    ap.add_argument("--fit-points", action="store_true", help="也跑實驗的校準擬合（PyMC，每人約一分鐘）")
    ap.add_argument("--by-snr", action="store_true", help="fit_points 也對每個 SNR 單獨跑（看單場的題數夠不夠）")
    ap.add_argument("--scale", type=float, default=2.0, help="傳進 fit_points 前把聲學差別除以這個數")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    res = {}
    for pid in a.participants:
        t, _, _ = load(pid)
        r = c1_c2(t)
        r["c5_curve"] = c5_curve(pid, t)
        if a.fit_points:
            r["c5_fit_points"] = c5_fit_points(pid, t, a.scale, a.seed)
            if a.by_snr:
                r["c5_fit_points_by_snr"] = {f"{s:g}": c5_fit_points(pid, t[t.snr == s], a.scale, a.seed) for s in sorted(t.snr.unique())}
        res[pid] = r
        print(f"\n{pid}（「不同」試 {r['n']} 題，命中 {r['hit']:.2f}）")
        print(f"  C1 排序力：聲學尺 {r['auc_acoustic']:.2f} [{r['auc_acoustic_ci95'][0]:.2f}, {r['auc_acoustic_ci95'][1]:.2f}]"
              f"（M&N {r['auc_mn']:.2f}，特徵數 {r['auc_features']:.2f}）→ {'通過' if r['c1_pass'] else '未過'}（門檻 ≥ {C1_MIN}）")
        if r["c2_pass"] is None:
            print("  C2 單一軸：不適用（這一場只有一個 SNR）")
        else:
            print(f"  C2 單一軸：AIC(差別) {r['aic_x']:.1f}，AIC(差別 + SNR) {r['aic_x_snr']:.1f}，加 SNR 的改善 {r['aic_gain_from_snr']:+.1f}，"
                  f"SNR 係數 {r['snr_coef']:+.3f} / dB → {'通過' if r['c2_pass'] else '未過'}（門檻 < {C2_MAX}）")
        c = r["c5_curve"]
        print(f"  C5 (a) 換算曲線：P = .5 的點 {c['x_half']:.2f} dB（合併誤報 {c['false_alarm_pooled']:.2f}；曲線中點 {c['x50_midpoint']:.2f}），"
              f"到頂九成 {c['x90']:.2f} dB，測過的範圍 {c['tested_range'][0]:.2f}–{c['tested_range'][1]:.2f} → {'在範圍內' if c['c5a_pass'] else '不在範圍內'}")
        if a.fit_points:
            f = r["c5_fit_points"]
            print(f"  C5 (b) fit_points：L {f['L']:.2f} dB，H {f['H']:.2f} dB；50% 點 {f['x50']:.2f} [{f['x50_ci90'][0]:.2f}, {f['x50_ci90'][1]:.2f}]"
                  f"（無解 {f['x50_unsolved']:.0%}），到頂點 {f['plateau']:.2f} [{f['plateau_ci90'][0]:.2f}, {f['plateau_ci90'][1]:.2f}]"
                  f"（無解 {f['plateau_unsolved']:.0%}）；in_range {f['in_range']}；發散 {f['divergences']}；n = {f['n_used']}")
            for w in f["warnings"]:
                print(f"       ⚠ {w}")
            for k, g in r.get("c5_fit_points_by_snr", {}).items():
                print(f"       只用 {float(k):+g} dB 那一場（n = {g['n_used']}）：L {g['L']:.2f}，H {g['H']:.2f}；50% 點 {g['x50']:.2f} [{g['x50_ci90'][0]:.2f}, {g['x50_ci90'][1]:.2f}]"
                      f"（無解 {g['x50_unsolved']:.0%}），到頂點 {g['plateau']:.2f} [{g['plateau_ci90'][0]:.2f}, {g['plateau_ci90'][1]:.2f}]；in_range {g['in_range']}；發散 {g['divergences']}")
    json.dump(res, open(os.path.join(out, "acoustic_scale_criteria.json"), "w"), ensure_ascii=False, indent=1)
    print("\n寫出：", os.path.join(out, "acoustic_scale_criteria.json"))
