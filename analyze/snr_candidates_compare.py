"""SNR 候選值比較：+2、+3、+4.5 dB 哪一個最接近設計要的。這是預測，不是量測：沒有人聽過這三個 SNR。

    python analyze/snr_candidates_compare.py                 # 約 10 分鐘（頻譜圖 40 秒，其餘是 bootstrap）
    python analyze/snr_candidates_compare.py --boot 200      # 快一點，區間較粗
    python analyze/snr_candidates_compare.py --figs-only     # 只用上次的 json 重畫圖

讀：stimuli/T??_<子音>3.wav（經 acoustic_scale_compute，把它的 SNR 清單換成這裡要的）
    analyze/out/acoustic_scale_trials_<聽者>.csv（先跑 acoustic_scale_validate.py --participant <聽者>）
    data/<聽者>_sound_perception_dprime_snr*.csv（「相同」試的誤報）
    stimuli/consonant_dprime.csv 與 adaptive_vawm_EN 的 LEVELS_AUDIO、foil_for（實驗實際怎麼挑對子）
算：(1) 120 對 × 12 個說話者編號 = 1440 個音檔對，在 0 / +2 / +3 / +4.5 / +6 dB 各自的「頻譜圖上的差別」x (dB)
    (2) 每位聽者的換算曲線（acoustic_scale_listener_curve 的模型與擬合，所有量過的 SNR 共用一條）
            P(答不一樣 | x) = FA + (1 − FA − λ) / (1 + exp(−(x − x50)/s))
        L = P = .5 的點，H = x90 = x50 + s·ln 9
    (3) 實驗六個層級各自用到的音檔對，x 的分布（層級在這種噪音下排不排得出難易）
        每個 SNR × 聽者：L、H 落在 1440 個音檔對的哪裡（低於中點 / 之間 / 高於 x90；L ± 0.3 dB 與 H ± 0.3 dB 內各有幾個、幾對）；
        實驗的六個聲音層級（每級 = 16 個目標子音各自用 foil_for 挑的對子 × 12 個編號）預測的命中；六級平均
    (4) 四條判準 K1–K4（常數在下面，門檻是這支程式定的，不是實驗的設定）
    (5) 沒量過的 SNR 的誤報：在量過的 SNR 之間線性內插；另用兩側量過的值各算一次看結論動多少。
        只有一個 SNR 的聽者（CAL01）：用那一個值；另借另外兩位聽者的 SNR 變化量各算一次
    (6) bootstrap：重抽「不同」試重新擬合曲線，同時重抽「相同」試的誤報（二項）；所有數字的 95% 區間、每條判準通過的比例
    (7) 內插可不可信：留一個 SNR 不用，拿其餘的曲線去預測它。偏差用兩種單位報：命中題數、與「曲線要平移幾 dB 才最貼合」
        另加一個參數讓中點隨 SNR 線性移動（x50 = x50 在 +6 dB 的值 + γ·(SNR − 6)），看 γ 的概似比區間有多寬
出：analyze/out/snr_candidates_pairs.csv（格式同 acoustic_scale_pairs.csv）、snr_candidates_compare.json、
    snr_candidates_table.csv（每個 SNR × 聽者一列）、snr_candidates_*.png 七張
對應：review/snr_2026-11-26_candidates_2_3_4.5dB.md、review/figures/fig39–fig45。

為什麼這樣算：
- 實驗的聲音軸是 M&N 的 d′ 表，但它在這種噪音下排不出這三位聽者的難易；排得出來的是頻譜圖差別。所以「在某個 SNR 會多難」
  只能經由頻譜圖差別去預測，再照實驗挑對子的規則（foil_for）加總回六個層級。
- 換算曲線假設「同一位聽者的 x50、s、λ 不隨 SNR 變」。這個假設只在 0 / +6 / +12 dB 之間檢驗過，第 (7) 項就是在量它錯多少。
- bootstrap 同時重抽誤報，所以區間比 acoustic_scale_listener_curve.py 報的（誤報固定）略寬。
- 這裡用的全是純知覺的同 / 異判斷，沒有記憶負荷。SFT 裡聲音要先記住再比，這支程式對那一段沒有任何資訊。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2

import _common as C
import adaptive_vawm_EN as A
import acoustic_scale_compute as M
from acoustic_scale_listener_curve import XCOL, fit, load, nll

SNRS = [0.0, 2.0, 3.0, 4.5, 6.0]          # 0 與 6 是量過的參考點
CANDIDATES = [2.0, 3.0, 4.5]
LISTENERS = ["CAL01", "WEB01", "yyc1116"]
NATIVE = ["CAL01", "WEB01"]               # 英語母語者（研究者確認；見各自的 pilot 筆記）
EX = [f"ex_{t}" for t in A.TALKERS]
GROUP_OF = {t: ("G1" if i < 6 else "G2" if i < 9 else "G3") for i, t in enumerate(A.TALKERS)}   # 三組近乎相同的錄音（stimuli_voice_check.py）

# ---- 判準（這支程式定的；理由在筆記 §1）
NEAR = 0.3            # dB：「在 L / H 附近」的半寬，沿用 calib_pilot_report.py 與 scratchpad HOWTO 的 ± 0.3
K_NEAR_N = 72         # K1 / K2：附近至少幾個音檔對 = 1440 的 5%（CAL01 在 +6 dB 的 L 端是 41 個、H 端 206 個；前者被判為太少）
K_NEAR_PAIRS = 20     # K1 / K2：這些音檔對至少來自幾個對子（同一例子是 12 對與 47 對）
K_OUTSIDE_N = 72      # K1：低於 L 的至少 5%；K2：高於 H 的至少 5%。L / H 不貼在現有刺激的邊緣
K_NEAR_ALT = (50, 72, 100, 150)   # 門檻的敏感度：K_NEAR_N 換成這幾個值，結論動不動。第一版用 100，見 judge() 的說明
K_MID = (0.60, 0.90)  # K3：實驗第 3、4 級（d′ 1.5、2.3）預測命中的平均要在這個範圍內
K_FA = 0.20           # K4：誤報上限
MID_LEVELS = (2, 3)   # LEVELS_AUDIO 的索引

COL = {"CAL01": C.RED, "WEB01": C.BLUE, "yyc1116": C.GREY}
MARK = {"CAL01": "o", "WEB01": "s", "yyc1116": "^"}
WHO = {"CAL01": "CAL01（母語）", "WEB01": "WEB01（母語）", "yyc1116": "yyc1116（非母語）"}
LBL = {"CAL01": ((-9, 9), "right"), "WEB01": ((8, -13), "left"), "yyc1116": ((8, 5), "left")}     # 圖上數字放在點的哪一側，三位聽者錯開
db = lambda s: f"{s:+g} dB" if s else "0 dB"


def token_x(out, recompute=False):
    """每個 SNR 的 x：{snr: 120 × 12 的陣列}，以及對子名稱。算一次約 40 秒，存成 csv，之後直接讀。"""
    f = os.path.join(out, "snr_candidates_pairs.csv")
    if recompute or not os.path.exists(f) or set(pd.read_csv(f).snr.astype(float)) != set(SNRS):
        M.SNRS = list(SNRS)                # acoustic_scale_compute 的 features() / pairs() 讀模組層級的 SNRS；演算法不動
        M.pairs(M.features()).to_csv(f, index=False)
    T = pd.read_csv(f)
    T["pair"] = ["-".join(sorted([a, b])) for a, b in zip(T.a, T.b)]
    names = sorted(T.pair.unique())
    X = {float(s): g.set_index("pair").loc[names, EX].values for s, g in T.groupby(T.snr.astype(float))}
    old = os.path.join(out, "acoustic_scale_pairs.csv")           # 量過的 SNR 要和既有的表逐格相同，否則這裡的 x 與聽者資料用的 x 不是同一把尺
    check = {}
    if os.path.exists(old):
        O = M.load_pairs(out).set_index(["snr", "pair"])
        for s in (0.0, 6.0):
            check[f"{s:g}"] = float(np.abs(O.loc[str(s)].loc[names, EX].values - X[s]).max())
    return X, names, check


def level_index(names):
    """實驗的六個聲音層級各會用到哪些音檔對：每級 16 個目標子音 × 12 個編號 = 192 個等機率的（對子, 編號）。
    挑法照 SFT_WM_nochange_EN.run_calib_trial：目標子音均勻抽、編號均勻抽、foil = foil_for(表, 目標, 層級)。"""
    table, row = A.load_dprime(), {n: i for i, n in enumerate(names)}
    idx, used = [], []
    for lev in A.LEVELS_AUDIO:
        foils = [(c, A.foil_for(table, c, lev)) for c in sorted(table)]
        idx.append(np.array([row["-".join(sorted([c, f]))] for c, (f, _) in foils]))
        used.append([dict(target=c, foil=f, dprime=d) for c, (f, d) in foils])
    return idx, used


def P(x, fa, x50, s, lam):
    with np.errstate(over="ignore"):
        return fa + (1 - fa - lam) / (1 + np.exp(-(x - x50) / s))


def x_at(p, fa, x50, s, lam):
    """曲線上 P = p 的 x；不存在（誤報已經高過 p，或 p 高過曲線的頂）回 nan。"""
    q = (p - fa) / (1 - fa - lam)
    return x50 + s * np.log(q / (1 - q)) if 0 < q < 1 else np.nan


def fa_scenarios(pid, fa_all, snr):
    """沒量過的 SNR 的誤報。fa_all = {聽者: {snr: 誤報}}。回傳 {情境名: 值}，第一個是主要的。"""
    own = fa_all[pid]; ks = sorted(own); vs = [own[k] for k in ks]
    if len(ks) > 1:
        lo = max([k for k in ks if k <= snr], default=ks[0]); hi = min([k for k in ks if k >= snr], default=ks[-1])
        return {"interp": float(np.interp(snr, ks, vs)), "lower_neighbour": own[lo], "upper_neighbour": own[hi]}
    sc = {"interp": vs[0]}                 # 只量過一個 SNR：用那個值；另借別人的「從那個 SNR 到這個 SNR 的變化」
    for o, d in fa_all.items():
        if o != pid and len(d) > 1:
            kk = sorted(d); sc[f"borrow_{o}"] = float(np.clip(vs[0] + np.interp(snr, kk, [d[k] for k in kk]) - np.interp(ks[0], kk, [d[k] for k in kk]), 0.005, 0.95))
    return sc


def metrics(par, fa, X, names, lev_idx, detail=False):
    """一位聽者、一個 SNR、一組曲線參數與誤報 → 這個 SNR 的全部數字。X：120 × 12。"""
    x50, s, lam = par["x50"], par["s"], par["lam"]
    x90, xL = x50 + s * np.log(9), x_at(0.5, fa, x50, s, lam)
    p_lev = np.array([P(X[i], fa, x50, s, lam).mean() for i in lev_idx])
    r = dict(fa=fa, x50=x50, x90=x90, xL=xL, n_below_x50=int((X < x50).sum()), n_between=int(((X >= x50) & (X <= x90)).sum()), n_above_x90=int((X > x90).sum()),
             hit_all_tokens=float(P(X, fa, x50, s, lam).mean()), hit_overall=float(p_lev.mean()), hit_mid=float(p_lev[list(MID_LEVELS)].mean()),
             hit_low3=float(p_lev[:3].mean()), hit_top2=float(p_lev[4:].mean()), hit_range=float(np.ptp(p_lev)),     # DECISIONS 24 當初的目標：d′ ≤ 1.5 的三級掉到七成、d′ ≥ 3 的兩級在天花板
             **{f"hit_L{k + 1}": float(v) for k, v in enumerate(p_lev)})
    for tag, target in (("L", xL), ("H", x90)):
        m = np.abs(X - target) <= NEAR if np.isfinite(target) else np.zeros_like(X, bool)
        r[f"n_near_{tag}"] = int(m.sum()); r[f"pairs_near_{tag}"] = int(m.any(1).sum())
        if detail and m.sum():
            ii, jj = np.where(m); tk = [A.TALKERS[j] for j in jj]
            r[f"near_{tag}_by_group"] = {g: int(sum(GROUP_OF[t] == g for t in tk)) for g in ("G1", "G2", "G3")}
            r[f"near_{tag}_with_b_or_v"] = int(sum(bool({"b", "v"} & set(names[i].split("-"))) for i in ii))
            r[f"near_{tag}_top_pairs"] = pd.Series([names[i] for i in ii]).value_counts().head(5).to_dict()
    r["n_below_L"] = int((X < xL).sum()) if np.isfinite(xL) else 0
    r.update({k: (bool(v) if k.startswith("K") or k == "all_pass" else int(v)) for k, v in judge(r).items()})
    return r


def judge(m, near_n=K_NEAR_N):
    """四條判準。m 的值可以是單一數字，也可以是 bootstrap 的陣列。
    near_n 的來歷：第一版寫 100，跑出來才發現 1440 個音檔對在分布中段每 0.6 dB 本來就只有 110 個上下，
    門檻剛好壓在那裡，K2 過不過變成跟 SNR 無關的擲銅板。改成和 K_OUTSIDE_N 同一條規則（5% = 72）；兩種門檻的結果都報（K_NEAR_ALT）。"""
    g = lambda k: np.asarray(m[k], float)
    k1 = (g("n_near_L") >= near_n) & (g("pairs_near_L") >= K_NEAR_PAIRS) & (g("n_below_L") >= K_OUTSIDE_N)
    k2 = (g("n_near_H") >= near_n) & (g("pairs_near_H") >= K_NEAR_PAIRS) & (g("n_above_x90") >= K_OUTSIDE_N)
    k3 = (g("hit_mid") >= K_MID[0]) & (g("hit_mid") <= K_MID[1])
    k4 = g("fa") <= K_FA
    n = k1.astype(int) + k2 + k3 + k4
    return dict(K1=k1, K2=k2, K3=k3, K4=k4, n_pass=n, all_pass=n == 4)


def listener(pid):
    t, fa_d, n_same = load(pid)
    return dict(x=t[XCOL].values, y=t.correct.values.astype(float), snr=t.snr.values, fa=fa_d, n_same=n_same, t=t)


def fit_listener(d, fa_d=None, rows=None):
    """共用一條曲線的擬合；fa_d 換成重抽的誤報、rows 換成重抽的題目。誤報 0 時用半題（同 acoustic_scale_listener_curve.curve）。"""
    fa_d = fa_d or d["fa"]; rows = np.arange(len(d["x"])) if rows is None else rows
    fa = np.array([max(fa_d[s], 0.5 / d["n_same"][s]) for s in d["snr"][rows]])
    return fit(d["x"][rows], d["y"][rows], fa)


def boot_draw(d, rng):
    """一次 bootstrap：重抽「不同」試（不分 SNR，同既有程式），誤報用二項重抽。回傳 (曲線參數, 誤報字典)。"""
    rows = rng.integers(0, len(d["x"]), len(d["x"]))
    fa_b = {s: rng.binomial(d["n_same"][s], d["fa"][s]) / d["n_same"][s] for s in d["fa"]}
    return fit_listener(d, fa_b, rows), {s: max(v, 0.5 / d["n_same"][s]) for s, v in fa_b.items()}


def holdout(d, train, test, n_boot, rng):
    """用 train 那幾個 SNR 的「不同」試擬合，預測 test 那個 SNR 的「不同」試。誤報兩種：test 自己量到的、由 train 的誤報內插（外側取最近的）。"""
    tr, te = np.isin(d["snr"], train), d["snr"] == test
    sub = dict(x=d["x"][tr], y=d["y"][tr], snr=d["snr"][tr], fa={s: d["fa"][s] for s in train}, n_same=d["n_same"])
    par = fit_listener(sub)
    x, y = d["x"][te], d["y"][te]
    fa_meas = max(d["fa"][test], 0.5 / d["n_same"][test]); fa_int = float(np.interp(test, sorted(train), [d["fa"][s] for s in sorted(train)]))
    res = dict(train=[float(s) for s in train], test=float(test), n=int(te.sum()), observed_hits=int(y.sum()), observed_rate=float(y.mean()),
               curve=dict(x50=par["x50"], s=par["s"], lam=par["lam"], x90=par["x50"] + par["s"] * np.log(9)), fa_measured=fa_meas, fa_interpolated=fa_int)
    boots = [boot_draw(sub, rng) for _ in range(n_boot)]
    for tag, fa in (("with_fa_measured", fa_meas), ("with_fa_interp", fa_int)):
        p = P(x, fa, par["x50"], par["s"], par["lam"])
        sim = []                                                   # 預測分布：訓練資料的 bootstrap（曲線 + 誤報）× 測試題的二項雜訊
        for pb, fb in boots:
            fab = fa if tag == "with_fa_measured" else float(np.interp(test, sorted(train), [fb[s] for s in sorted(train)]))
            sim.append(rng.binomial(1, np.clip(P(x, fab, pb["x50"], pb["s"], pb["lam"]), 0, 1)).sum())
        sim = np.array(sim)
        nll = lambda dx: -np.sum(y * np.log(np.clip(P(x + dx, fa, par["x50"], par["s"], par["lam"]), 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - P(x + dx, fa, par["x50"], par["s"], par["lam"]), 1e-6, 1 - 1e-6)))
        best = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
        grid = np.linspace(-4, 4, 1601); inside = grid[np.array([nll(g) for g in grid]) <= best.fun + 1.92]      # 概似比 95% 區間
        third = pd.qcut(x, 3, labels=False, duplicates="drop")
        res[tag] = dict(predicted_hits=float(p.sum()), predicted_rate=float(p.mean()), miss_points=float(100 * (y.mean() - p.mean())),
                        predictive_95=[float(v) for v in np.percentile(sim, [2.5, 97.5])], p_le=float((sim <= y.sum()).mean()), p_ge=float((sim >= y.sum()).mean()),
                        shift_db=float(best.x), shift_ci95=[float(inside.min()), float(inside.max())], lr_shift=float(2 * (nll(0.0) - best.fun)),
                        thirds=[dict(x_mean=float(x[third == k].mean()), n=int((third == k).sum()), observed=float(y[third == k].mean()), predicted=float(p[third == k].mean())) for k in sorted(set(third))])
    return res


def drift(d):
    """中點會不會隨 SNR 移動：x50(SNR) = x50(+6 dB) + γ·(SNR − 6)，等於把每題的 x 減掉 γ·(SNR − 6) 再用同一條曲線。
    回傳 γ（x 的 dB / SNR 的 dB）、它的概似比 95% 區間（剖面）、與 γ = 0 的概似比檢定。"""
    fa = np.array([max(d["fa"][s], 0.5 / d["n_same"][s]) for s in d["snr"]])
    f3 = lambda g: min((minimize(lambda p: nll(p, d["x"] - g * (d["snr"] - 6.0), d["y"], fa), [a, b, -2.0], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
                        for a in (1.0, 2.5) for b in (-1.0, 0.0)), key=lambda r: r.fun).fun
    grid = np.round(np.arange(-0.40, 0.401, 0.01), 2); prof = np.array([f3(g) for g in grid]); k = int(prof.argmin()); inside = grid[prof <= prof[k] + 1.92]
    lr = 2 * (f3(0.0) - prof[k])
    return dict(gamma=float(grid[k]), gamma_ci95=[float(inside.min()), float(inside.max())], lr=float(lr), p=float(chi2.sf(max(lr, 0), 1)))


def pct(v, q=(2.5, 97.5)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(a) for a in np.percentile(v, q)] if len(v) else [float("nan")] * 2


def main():
    ap = argparse.ArgumentParser(description="SNR 候選值比較：+2、+3、+4.5 dB（預測，不是量測）。")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--holdout-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261126); ap.add_argument("--recompute", action="store_true", help="重算頻譜圖差別（預設讀上次存的 csv）")
    ap.add_argument("--figs-only", action="store_true", help="不重算，讀上次的 snr_candidates_compare.json 重畫圖"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = C.out_dir(a.out); plt = C.setup_plot(); rng = np.random.default_rng(a.seed)

    X, names, check = token_x(out, a.recompute)
    lev_idx, lev_used = level_index(names)
    D = {p: listener(p) for p in LISTENERS}
    fa_all = {p: D[p]["fa"] for p in LISTENERS}
    par = {p: fit_listener(D[p]) for p in LISTENERS}
    if a.figs_only:
        figures(plt, out, X, json.load(open(os.path.join(out, "snr_candidates_compare.json"), encoding="utf-8")), D, par); print("重畫：", out); return
    n_tok = X[0.0].size
    R = dict(snrs=SNRS, candidates=CANDIDATES, n_tokens=int(n_tok), levels=A.LEVELS_AUDIO, check_vs_existing_table_max_abs_diff=check,
             criteria=dict(near_db=NEAR, K1=f"L ± {NEAR} dB 內 ≥ {K_NEAR_N} 個音檔對、≥ {K_NEAR_PAIRS} 對，且低於 L 的 ≥ {K_OUTSIDE_N}", K2=f"H ± {NEAR} dB 內 ≥ {K_NEAR_N} 個、≥ {K_NEAR_PAIRS} 對，且高於 H 的 ≥ {K_OUTSIDE_N}",
                           K3=f"第 3、4 級預測命中的平均在 {K_MID[0]}–{K_MID[1]}", K4=f"誤報 ≤ {K_FA}"),
             x_distribution={f"{s:g}": dict(min=float(X[s].min()), q10=float(np.percentile(X[s], 10)), median=float(np.median(X[s])), q90=float(np.percentile(X[s], 90)), max=float(X[s].max()), mean=float(X[s].mean())) for s in SNRS},
             x_shift_per_token={f"{lo:g}_to_{hi:g}": dict(median=float(np.median(X[hi] - X[lo])), q10=float(np.percentile(X[hi] - X[lo], 10)), q90=float(np.percentile(X[hi] - X[lo], 90))) for lo, hi in ((2.0, 3.0), (3.0, 4.5), (2.0, 4.5), (0.0, 6.0))},
             level_pairs=[sorted({"-".join(sorted([u["target"], u["foil"]])) for u in lv}) for lv in lev_used],
             level_x={f"{s:g}": [dict(level=l, median=float(np.median(X[s][i])), q10=float(np.percentile(X[s][i], 10)), q90=float(np.percentile(X[s][i], 90))) for l, i in zip(A.LEVELS_AUDIO, lev_idx)] for s in SNRS}, curves={}, point={}, fa_sensitivity={}, boot={}, holdout={})
    print(f"頻譜圖差別：{len(names)} 對 × {len(A.TALKERS)} 個編號 = {n_tok} 個音檔對；與既有的表（0 / +6 dB）最大差 {check}")
    print("  各 SNR 的 x（最小 / 10% / 中位數 / 90% / 最大）：" + "；".join(f"{s:+g} dB {v['min']:.2f} / {v['q10']:.2f} / {v['median']:.2f} / {v['q90']:.2f} / {v['max']:.2f}" for s, v in zip(SNRS, R["x_distribution"].values())))
    sh = R["x_shift_per_token"]; print(f"  同一個音檔對從 +2 到 +4.5 dB，x 增加的中位數 {sh['2_to_4.5']['median']:.2f} dB（10%–90%：{sh['2_to_4.5']['q10']:.2f}–{sh['2_to_4.5']['q90']:.2f}）；0 到 +6 dB {sh['0_to_6']['median']:.2f} dB")

    print("  實驗的六個層級各自用到的音檔對，x 的中位數（10%–90%）；層級若排得出難易，這幾個數應該由小到大、彼此不重疊：")
    for s in CANDIDATES: print(f"    {s:+g} dB  " + " | ".join(f"{v['median']:.2f}（{v['q10']:.2f}–{v['q90']:.2f}）" for v in R["level_x"][f"{s:g}"]))

    # ---- 點估計、誤報的敏感度
    rows = []
    for p in LISTENERS:
        c = par[p]; R["curves"][p] = dict(x50=c["x50"], s=c["s"], lam=c["lam"], x90=c["x50"] + c["s"] * np.log(9), false_alarm={f"{k:g}": v for k, v in D[p]["fa"].items()}, n_same={f"{k:g}": int(v) for k, v in D[p]["n_same"].items()}, n_different=int(len(D[p]["x"])))
        print(f"\n{p}：x50 {c['x50']:.2f}  s {c['s']:.2f}  λ {c['lam']:.3f}  x90 {c['x50'] + c['s'] * np.log(9):.2f}；量到的誤報 " + " / ".join(f"{k:+g} dB {v:.3f}" for k, v in D[p]["fa"].items()))
        R["point"][p], R["fa_sensitivity"][p] = {}, {}
        for s in SNRS:
            sc = fa_scenarios(p, fa_all, s)
            m = metrics(c, sc["interp"], X[s], names, lev_idx, detail=True); R["point"][p][f"{s:g}"] = m
            R["fa_sensitivity"][p][f"{s:g}"] = {k: dict(fa=v, **{q: metrics(c, v, X[s], names, lev_idx)[q] for q in ("xL", "n_near_L", "n_below_L", "hit_overall", "hit_mid", "K1", "K2", "K3", "K4", "n_pass")}) for k, v in sc.items()}
            rows.append(dict(listener=p, snr=s, measured=s in D[p]["fa"], **{k: v for k, v in m.items() if not isinstance(v, dict)}))
            print(f"  {s:+4g} dB  誤報 {m['fa']:.3f}  L {m['xL']:.2f}  H {m['x90']:.2f} | 低於中點 {m['n_below_x50']:4d}  之間 {m['n_between']:4d}  高於 x90 {m['n_above_x90']:4d} | L 附近 {m['n_near_L']:3d}（{m['pairs_near_L']:2d} 對）H 附近 {m['n_near_H']:3d}（{m['pairs_near_H']:2d} 對）"
                  f"| 六級 " + " ".join(f"{m[f'hit_L{k}']:.2f}" for k in range(1, 7)) + f"  平均 {m['hit_overall']:.2f}  中段 {m['hit_mid']:.2f}  下三級 {m['hit_low3']:.2f}  上兩級 {m['hit_top2']:.2f} | 低於 L {m['n_below_L']} | K1–K4 " + "".join("✓" if m[k] else "✗" for k in ("K1", "K2", "K3", "K4")))
        for s in CANDIDATES:
            print(f"    誤報的敏感度 {s:+g} dB：" + "；".join(f"{k} {v['fa']:.3f} → L {v['xL']:.2f}、L 附近 {v['n_near_L']}、六級平均 {v['hit_overall']:.3f}、過 {v['n_pass']} 條" for k, v in R["fa_sensitivity"][p][f"{s:g}"].items()))
    pd.DataFrame(rows).to_csv(os.path.join(out, "snr_candidates_table.csv"), index=False)

    # ---- bootstrap
    keys = ("fa", "x50", "x90", "xL", "n_below_x50", "n_between", "n_above_x90", "n_below_L", "n_near_L", "pairs_near_L", "n_near_H", "pairs_near_H", "hit_overall", "hit_mid", "hit_low3", "hit_top2", "hit_range", "hit_all_tokens",
            *[f"hit_L{k}" for k in range(1, 7)], "K1", "K2", "K3", "K4", "all_pass", "n_pass")
    B = {}
    for p in LISTENERS:
        B[p] = {s: {k: [] for k in keys} for s in SNRS}
        for _ in range(a.boot):
            pb, fb = boot_draw(D[p], rng)
            for s in SNRS:
                m = metrics(pb, fa_scenarios(p, {**fa_all, p: fb}, s)["interp"], X[s], names, lev_idx)
                for k in keys: B[p][s][k].append(m[k])
        B[p] = {s: {k: np.array(v, float) for k, v in d.items()} for s, d in B[p].items()}
        R["boot"][p] = {f"{s:g}": dict(**{k: pct(B[p][s][k]) for k in keys if not k.startswith("K") and k not in ("all_pass", "n_pass")}, xL_missing=float(np.mean(~np.isfinite(B[p][s]["xL"]))),
                                       **{f"P_{k}": float(B[p][s][k].mean()) for k in ("K1", "K2", "K3", "K4", "all_pass")}, n_pass_mean=float(B[p][s]["n_pass"].mean())) for s in SNRS}
        R["boot"][p]["pairwise"] = {f"{lo:g}_vs_{hi:g}": dict(lower_snr_passes_more=float(np.mean(B[p][lo]["n_pass"] > B[p][hi]["n_pass"])), same=float(np.mean(B[p][lo]["n_pass"] == B[p][hi]["n_pass"])),
                                                              higher_snr_passes_more=float(np.mean(B[p][lo]["n_pass"] < B[p][hi]["n_pass"]))) for lo, hi in ((2.0, 3.0), (3.0, 4.5), (2.0, 4.5))}
        R["boot"][p]["candidates_tie_share"] = float(np.mean(np.ptp(np.stack([B[p][c]["n_pass"] for c in CANDIDATES]), 0) == 0))
        print(f"\n{p} bootstrap（{a.boot} 次）：")
        for s in SNRS:
            b = R["boot"][p][f"{s:g}"]
            print(f"  {s:+4g} dB  L {np.round(b['xL'], 2).tolist()}  L 附近 {np.round(b['n_near_L']).astype(int).tolist()}  低於 L {np.round(b['n_below_L']).astype(int).tolist()}  H 附近 {np.round(b['n_near_H']).astype(int).tolist()}  六級平均 {np.round(b['hit_overall'], 2).tolist()}  中段 {np.round(b['hit_mid'], 2).tolist()}"
                  f"  過關比例 K1 {b['P_K1']:.2f} K2 {b['P_K2']:.2f} K3 {b['P_K3']:.2f} K4 {b['P_K4']:.2f} 四條都過 {b['P_all_pass']:.2f}")
        print(f"  曲線中點 x50 的區間 {np.round(R['boot'][p]['3']['x50'], 2).tolist()}、x90 {np.round(R['boot'][p]['3']['x90'], 2).tolist()}（誤報一起重抽）；三個候選值過的條數完全一樣的比例 {R['boot'][p]['candidates_tie_share']:.2f}")
        print("  兩兩比過的條數（較低的 SNR 過得多 / 一樣 / 較高的過得多）：" + "；".join(f"{k.replace('_vs_', ' 對 ')} dB {v['lower_snr_passes_more']:.2f} / {v['same']:.2f} / {v['higher_snr_passes_more']:.2f}" for k, v in R["boot"][p]["pairwise"].items()))

    # ---- 門檻的敏感度與彙總：每個候選值，兩位母語者合計過幾條；bootstrap 裡四條都過的比例
    R["threshold_sensitivity"], R["summary"] = {}, {}
    for near_n in K_NEAR_ALT:
        R["threshold_sensitivity"][str(near_n)] = {p: {f"{s:g}": dict(n_pass=int(judge(R["point"][p][f"{s:g}"], near_n)["n_pass"]), P_all_pass=float(judge(B[p][s], near_n)["all_pass"].mean())) for s in SNRS} for p in LISTENERS}
    print(f"\n彙總（兩位母語者；點估計過的條數合計，滿分 8；括號 = bootstrap 裡四條都過的比例 CAL01 / WEB01）。門檻 K_NEAR_N = {K_NEAR_N}，另列 {[n for n in K_NEAR_ALT if n != K_NEAR_N]}：")
    for s in SNRS:
        line = f"  {s:+4g} dB"
        for near_n in (K_NEAR_N, *[n for n in K_NEAR_ALT if n != K_NEAR_N]):
            q = R["threshold_sensitivity"][str(near_n)]
            line += f"  | 門檻 {near_n:3d}：{sum(q[p][f'{s:g}']['n_pass'] for p in NATIVE)}（" + " / ".join(f"{q[p][f'{s:g}']['P_all_pass']:.2f}" for p in NATIVE) + "）"
        q = R["threshold_sensitivity"][str(K_NEAR_N)]
        R["summary"][f"{s:g}"] = dict(native_n_pass=int(sum(q[p][f"{s:g}"]["n_pass"] for p in NATIVE)), native_min_P_all_pass=float(min(q[p][f"{s:g}"]["P_all_pass"] for p in NATIVE)),
                                      failed={p: [k for k in ("K1", "K2", "K3", "K4") if not R["point"][p][f"{s:g}"][k]] for p in NATIVE})
        print(line)

    # ---- 三種差異的大小：候選值之間、同一位聽者的 bootstrap、兩位母語者之間
    R["effect_sizes"] = {}
    print("\n差異的大小（候選值之間 = +4.5 減 +2 dB；區間半寬與兩人之差都在 +3 dB）：")
    for k, lab in (("xL", "L 的位置 (dB)"), ("n_below_L", "低於 L 的音檔對"), ("n_near_L", "L 附近的音檔對"), ("n_near_H", "H 附近的音檔對"), ("n_above_x90", "高於 H 的音檔對"), ("hit_overall", "六級平均命中"), ("hit_mid", "中段命中"), ("hit_top2", "上兩級命中"), ("hit_low3", "下三級命中")):
        e = dict(across_candidates={p: float(R["point"][p]["4.5"][k] - R["point"][p]["2"][k]) for p in LISTENERS},
                 across_candidates_ci95={p: pct(B[p][4.5][k] - B[p][2.0][k]) for p in LISTENERS},
                 boot_halfwidth_at_3={p: float(np.diff(pct(B[p][3.0][k]))[0] / 2) for p in LISTENERS},
                 native_gap_at_3=float(R["point"]["CAL01"]["3"][k] - R["point"]["WEB01"]["3"][k]))
        R["effect_sizes"][k] = e
        print(f"  {lab:14s} 候選值之間 " + " / ".join(f"{p} {e['across_candidates'][p]:+.3g}" for p in NATIVE) + "；bootstrap 半寬 " + " / ".join(f"{p} {e['boot_halfwidth_at_3'][p]:.3g}" for p in NATIVE) + f"；CAL01 − WEB01 {e['native_gap_at_3']:+.3g}"
              + "；候選值之差的成對區間 " + " / ".join(f"{p} {np.round(e['across_candidates_ci95'][p], 3).tolist()}" for p in NATIVE))
    slope = R["x_shift_per_token"]["2_to_4.5"]["median"] / 2.5      # x 每 1 dB SNR 移多少（+2 到 +4.5 之間）
    R["x_per_db_snr"] = slope
    R["equivalent_snr"] = dict(native_x50_gap_db=float(par["WEB01"]["x50"] - par["CAL01"]["x50"]), native_x50_gap_as_snr=float((par["WEB01"]["x50"] - par["CAL01"]["x50"]) / slope),
                               boot_halfwidth_x50={p: float(np.diff(pct(B[p][3.0]["x50"]))[0] / 2) for p in LISTENERS}, boot_halfwidth_x50_as_snr={p: float(np.diff(pct(B[p][3.0]["x50"]))[0] / 2 / slope) for p in LISTENERS},
                               candidate_span_snr=2.5, candidate_span_x=float(R["x_shift_per_token"]["2_to_4.5"]["median"]))
    q = R["equivalent_snr"]
    print(f"  換成同一個單位：x 每 1 dB SNR 移 {slope:.3f} dB。三個候選值相隔 2.5 dB SNR = x 上 {q['candidate_span_x']:.2f} dB；兩位母語者的中點差 {q['native_x50_gap_db']:.2f} dB = {q['native_x50_gap_as_snr']:.1f} dB SNR；"
          "中點的 bootstrap 半寬 " + " / ".join(f"{p} {q['boot_halfwidth_x50'][p]:.2f} dB = {q['boot_halfwidth_x50_as_snr'][p]:.1f} dB SNR" for p in NATIVE))

    # ---- 內插可不可信：留一個 SNR
    print("\n留一個 SNR 的檢驗（用其餘的曲線預測它；誤報用該 SNR 自己量到的 / 由其餘內插）：")
    for p in LISTENERS:
        if len(D[p]["fa"]) < 3: continue
        R["holdout"][p] = {}
        for train, test in (((0.0, 12.0), 6.0), ((6.0, 12.0), 0.0), ((0.0, 6.0), 12.0), ((6.0,), 0.0), ((6.0,), 12.0)):
            h = holdout(D[p], train, test, a.holdout_boot, rng); R["holdout"][p]["+".join(f"{s:g}" for s in train) + f"->{test:g}"] = h
            for tag, lab in (("with_fa_measured", "誤報用量到的"), ("with_fa_interp", "誤報用內插的")):
                g = h[tag]; print(f"  {p}  {'+'.join(f'{s:g}' for s in train):>5s} → {test:+3g} dB（{h['n']} 題）[{lab}]：實際 {h['observed_hits']}（{h['observed_rate']:.2f}），預測 {g['predicted_hits']:.1f}（{g['predicted_rate']:.2f}），差 {g['miss_points']:+.1f} 個百分點，"
                                  f"預測區間 {np.round(g['predictive_95']).astype(int).tolist()}，P(≤) {g['p_le']:.2f}；曲線要平移 {g['shift_db']:+.2f} dB {np.round(g['shift_ci95'], 2).tolist()}")
    R["drift"] = {p: drift(D[p]) for p in LISTENERS if len(D[p]["fa"]) >= 3}
    for p, g in R["drift"].items():
        print(f"  {p}：中點隨 SNR 移動的斜率 γ = {g['gamma']:+.2f} [{g['gamma_ci95'][0]:+.2f}, {g['gamma_ci95'][1]:+.2f}]（x 的 dB / SNR 的 dB；音檔對本身每 1 dB SNR 移 {slope:.2f}）；γ = 0 的概似比 {g['lr']:.2f}，p = {g['p']:.2f}")
    json.dump(R, open(os.path.join(out, "snr_candidates_compare.json"), "w"), ensure_ascii=False, indent=1, default=float)

    figures(plt, out, X, R, D, par)
    print("\n寫出：", out, "（snr_candidates_pairs.csv、snr_candidates_compare.json、snr_candidates_table.csv、snr_candidates_*.png）")


def figures(plt, out, X, R, D, par):
    xs = np.arange(len(SNRS)); off = {"CAL01": -0.09, "WEB01": 0.09, "yyc1116": 0.27}
    shade = lambda ax: [ax.axvspan(SNRS.index(c) - 0.42, SNRS.index(c) + 0.42, color="#f4f1e8", zorder=0, lw=0) for c in CANDIDATES]
    tick = lambda ax: (ax.set_xticks(xs), ax.set_xticklabels([f"{db(s)}\n（{'量過' if s in (0.0, 6.0) else '預測'}）" for s in SNRS]), ax.set_xlabel("訊噪比（底色 = 三個候選值）"))

    def by_snr(key, ylab, title, fname, hline=None, ylim=None, listeners=LISTENERS, fmt="{:.0f}"):      # fmt=None：不在每個點旁邊寫數字（數字在筆記的表裡）
        fig, ax = plt.subplots(figsize=(8.2, 4.8)); shade(ax)
        if hline: ax.axhline(hline[0], color=C.GREEN, ls="--", lw=1.2, label=hline[1])
        for p in listeners:
            v = [R["point"][p][f"{s:g}"][key] for s in SNRS]; ci = np.array([R["boot"][p][f"{s:g}"][key] for s in SNRS])
            ax.errorbar(xs + off[p], v, [np.clip(np.array(v) - ci[:, 0], 0, None), np.clip(ci[:, 1] - np.array(v), 0, None)], fmt=MARK[p] + "-", color=COL[p], ms=7, lw=1.6, capsize=3, label=WHO[p], mec="white", mew=1)
            for u, w in zip(xs + off[p], v if fmt else []): ax.annotate(fmt.format(w), (u, w), textcoords="offset points", xytext=LBL[p][0], ha=LBL[p][1], fontsize=8, color="#333333")
        tick(ax); ax.set_ylabel(ylab); ax.legend(frameon=False, fontsize=9, loc="best")
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)

    # 圖 1：每個 SNR 的音檔對分布，疊上每位聽者的 L–H 範圍
    fig, ax = plt.subplots(figsize=(8.6, 5.4)); bins = np.arange(0, 11.01, 0.25); blues = ["#b7cfe3", "#8fb4d3", "#6a9bc3", "#447fb0", "#1f5f96"]
    for k, s in enumerate(SNRS):
        h, _ = np.histogram(X[s], bins); ax.bar(bins[:-1], h / h.max() * 0.82, 0.25, bottom=k, align="edge", color=blues[k], lw=0)
        ax.text(10.95, k + 0.42, f"{db(s)}（{'量過' if s in (0.0, 6.0) else '預測'}）\n中位數 {np.median(X[s]):.1f} dB", ha="right", va="center", fontsize=8.5, color="#333333")
    for j, p in enumerate(NATIVE):
        c = par[p]; x50, x90 = c["x50"], c["x50"] + c["s"] * np.log(9)
        ax.axvspan(x50, x90, color=COL[p], alpha=0.13, lw=0); ax.axvline(x50, color=COL[p], lw=1.4); ax.axvline(x90, color=COL[p], lw=1.4, ls="--")
        ax.text((x50 + x90) / 2, len(SNRS) + 0.08 + 0.3 * j, f"{p}：五成點 {x50:.1f} dB、到頂九成 {x90:.1f} dB", color="#333333", fontsize=8.5, ha="left" if j else "center", va="bottom", bbox=dict(fc="white", ec=COL[p], lw=1, pad=2))
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(0, 11); ax.set_ylim(0, len(SNRS) + 0.75); ax.set_xlabel("頻譜圖上的差別 x (dB)：越小 = 兩個音越難分")
    ax.set_title("1440 個音檔對在每個訊噪比下的分布，和兩位母語者「聽得出一半」到「幾乎都聽得出」的範圍\n（色帶 = 該聽者曲線的中點到到頂九成；假設這個範圍不隨訊噪比變）", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "snr_candidates_distribution.png")); plt.close(fig)

    by_snr("n_near_L", f"L（答「不一樣」五成的點）± {NEAR} dB 內的音檔對數", f"L 端有多少刺激可選：1440 個音檔對裡落在 L ± {NEAR} dB 的個數（誤差線 = 重抽題目的 95% 範圍）",
           "snr_candidates_near_L.png", hline=(K_NEAR_N, f"判準 K1 的下限 {K_NEAR_N} 個（1440 的 5%）"), ylim=(0, None), listeners=NATIVE)
    by_snr("n_below_L", "比 L（答「不一樣」五成的點）還難分的音檔對數", "L 是不是貼在現有刺激的最難一端：比 L 還難分的音檔對個數（誤差線 = 重抽題目的 95% 範圍）",
           "snr_candidates_below_L.png", hline=(K_OUTSIDE_N, f"判準 K1 的下限 {K_OUTSIDE_N} 個（1440 的 5%）"), ylim=(0, None), listeners=NATIVE)
    by_snr("hit_overall", "預測的命中率（六個層級平均）", "照實驗挑對子的方式，預測「不同」題的命中率（誤差線 = 重抽題目的 95% 範圍）",
           "snr_candidates_overall_hit.png", ylim=(0.4, 1.0), fmt=None)

    # 圖 5、6：兩位母語者各自的六級
    greens = {0.0: "#9a9a9a", 2.0: "#e8b04a", 3.0: "#d17c1f", 4.5: "#a8500f", 6.0: "#4d4d4d"}
    for p in NATIVE:
        fig, ax = plt.subplots(figsize=(8.2, 4.8)); lv = np.arange(6)
        ax.axhspan(*K_MID, xmin=(1.6 + 0.4) / 5.8, xmax=(3.4 + 0.4) / 5.8, color=C.GREEN, alpha=0.10, lw=0, label=f"判準 K3：中間兩級在 {K_MID[0]:.2f}-{K_MID[1]:.2f}")
        for s in SNRS:
            v = [R["point"][p][f"{s:g}"][f"hit_L{k}"] for k in range(1, 7)]; cand = s in CANDIDATES
            ax.plot(lv, v, "o-" if cand else "o--", color=greens[s], lw=2 if cand else 1.2, ms=6 if cand else 4, label=db(s) + ("（候選，預測）" if cand else "（參考，同一條曲線算的）"), mec="white", mew=0.8)
        ci = np.array([R["boot"][p]["3"][f"hit_L{k}"] for k in range(1, 7)]); ax.fill_between(lv, ci[:, 0], ci[:, 1], color=greens[3.0], alpha=0.16, lw=0, label="+3 dB 的 95% 範圍（重抽題目）")
        if len(D[p]["fa"]) > 1:
            t = D[p]["t"]; g = t[t.snr.isin([0.0, 6.0])].groupby(["snr", "x_planned"]).correct.agg(["mean", "size"])
            for s, mk in ((0.0, "x"), (6.0, "+")):
                ax.plot(lv, [g.loc[(s, l), "mean"] for l in A.LEVELS_AUDIO], mk, color=greens[s], ms=9, mew=2, label=f"{db(s)} 實際量到的（每級 {int(g.loc[(s, A.LEVELS_AUDIO[0]), 'size'])} 題）")
        ax.set_xticks(lv); ax.set_xticklabels([f"第 {k + 1} 級\n表上 d' {l}" for k, l in enumerate(A.LEVELS_AUDIO)]); ax.set_xlim(-0.4, 5.4); ax.set_ylim(0.2, 1.03); ax.set_ylabel("「不同」題的命中率")
        ax.set_xlabel("實驗的六個聲音層級（表上越右邊應該越容易）"); ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"{p}：實驗的六個聲音層級在各訊噪比下預測的命中率", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, f"snr_candidates_levels_{p}.png")); plt.close(fig)

    # 圖 7：留 +6 dB 不用，拿 0 與 +12 dB 的曲線預測它
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); xx = np.linspace(0, 10, 300)
    for p in [q for q in LISTENERS if q in R["holdout"]]:
        h = R["holdout"][p]["0+12->6"]; c = h["curve"]; g = h["with_fa_measured"]
        ax.plot(xx, P(xx, h["fa_measured"], c["x50"], c["s"], c["lam"]), color=COL[p], lw=2, label=f"{WHO[p]}：只用 0 與 +12 dB 擬合的曲線；+6 dB 實際 {h['observed_hits']}/{h['n']}，預測 {g['predicted_hits']:.1f}")
        ax.errorbar([t["x_mean"] for t in g["thirds"]], [t["observed"] for t in g["thirds"]], [np.sqrt(t["observed"] * (1 - t["observed"]) / t["n"]) for t in g["thirds"]], fmt=MARK[p], color=COL[p], ms=8, capsize=3, lw=1, mec="white", mew=1)
    ax.plot([], [], "o", color="#555555", label="點 = +6 dB 實際的命中率（依 x 分三組，每組 24 題 ± 1 個標準誤）；沒有參與擬合")
    ax.set_xlabel("頻譜圖上的差別 x (dB)"); ax.set_ylabel("回答「不一樣」的比例"); ax.set_xlim(0, 10); ax.set_ylim(0, 1.04); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    ax.set_title("內插可不可信：不用 +6 dB 的資料，拿 0 與 +12 dB 的曲線去預測 +6 dB", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, "snr_candidates_holdout_6dB.png")); plt.close(fig)


if __name__ == "__main__":
    main()
