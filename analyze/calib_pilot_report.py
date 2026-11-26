"""線上校準前測（online/calib_pilot）一場的報告：題目設計有沒有達到目的、只差清濁的對子落在換算曲線的哪一邊、L / H 會選到什麼。

    python analyze/calib_pilot_report.py --participant CAL01 --raw data/CAL01_calib_pilot_2026-11-26_04h10.csv --others WEB01 yyc1116

讀：data/<聽者>_sound_perception_dprime_snr6.csv（online/pavlovia_to_perception_csv.py 轉出；帶 x_acoustic、bin、group、voicing_only）
    analyze/out/acoustic_scale_pairs.csv（每一對 × 每位說話者在 +6 dB 的頻譜圖差別；acoustic_scale_compute.py）
    analyze/out/acoustic_scale_curve_<另一位>.json（--others；acoustic_scale_listener_curve.py 的輸出，只用來疊圖）
    --raw：原始的 jsPsych 檔，只用來算整場用時
算：(1) 八個聲學差別區間的命中率；誤報（整體、前後半、每 81 題一段、依子音、依三組聲音）；RT
    (2) 換算曲線  P(答不一樣 | x) = FA + (1 − FA − λ) / (1 + exp(−(x − x50)/s))，x = 題目清單上的 x_acoustic
    (3) 只差清濁的 7 對：曲線只用「非清濁」題擬合，再預測清濁題（清濁題沒有參與擬合）；
        另擬合帶偏移量的模型 x_eff = x + δ·(清濁題)，δ > 0 表示清濁對比聲學尺說的容易；概似比檢定；逐對、兩個方向、三組聲音
    (4) 排序力（AUC）：全部、拿掉清濁題、加上 δ 之後
    (5) 依這條曲線在 +6 dB 會被選為 L / H 的「對子 × 說話者」及其子音類別
出：analyze/out/calib_pilot_report_<聽者>.json 與四張 png。
對應：review/pilot_2026-11-26_CAL01_calib_pilot.md。

為什麼清濁題要「不參與擬合再預測」：42 題佔最低兩個區間的近一半，若一起擬合，曲線會被它們拉過去，
看起來就像沒有偏差；用其餘 174 題定曲線，才看得出清濁題是偏難、偏易還是剛好。
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, fisher_exact

import _common as C
from acoustic_scale_listener_curve import fit, nll
from acoustic_scale_validate import auc, boot_auc

TALKERS = [f"T{i:02d}" for i in range(1, 13)]
GROUP_OF = {**{t: "G1" for t in TALKERS[:6]}, **{t: "G2" for t in TALKERS[6:9]}, **{t: "G3" for t in TALKERS[9:]}}   # analyze/stimuli_voice_check.py
CLASS = {**{c: "塞" for c in "ptkbdg"}, **{c: "擦" for c in ("f", "theta", "s", "sh", "v", "eth", "z", "zh")}, "m": "鼻", "n": "鼻"}
VOICED = {"b", "d", "g", "v", "eth", "z", "zh"}
RED, BLUE, ORA, GREY = "#c0392b", "#2471a3", "#d35400", "#7f8c8d"


def P(x, fa, x50, s, lam):
    with np.errstate(over="ignore"):
        return fa + (1 - fa - lam) / (1 + np.exp(-(np.asarray(x, float) - x50) / s))


def fit_from(x, y, fa, start):
    """只從一個起點搜（bootstrap 用：從完整資料的最佳解出發，比九個起點快九倍）。"""
    with np.errstate(over="ignore"):
        r = minimize(nll, [start["x50"], np.log(start["s"]), np.log(max(start["lam"], 1e-4) / (0.3 - min(start["lam"], 0.299)))], args=(x, y, fa),
                     method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-3, fatol=1e-5))
        return dict(x50=float(r.x[0]), s=float(np.exp(r.x[1])), lam=float(0.3 / (1 + np.exp(-r.x[2]))), nll=float(r.fun))


def fit_offset(x, y, fa, v, step=0.1, start=None):
    """x_eff = x + δ·v。δ 用格點搜（每個 δ 重新擬合曲線），避免四個參數一起搜時互相補償。"""
    best = None
    for d in np.round(np.arange(-3.0, 3.0 + step / 2, step), 2):
        f = fit(x + d * v, y, fa) if start is None else fit_from(x + d * v, y, fa, start)
        if best is None or f["nll"] < best[1]["nll"]:
            best = (float(d), f)
    return best


def pair_name(a, b):
    return "-".join(sorted([C.ipa(a), C.ipa(b)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="線上校準前測一場的報告。")
    ap.add_argument("--participant", default="CAL01"); ap.add_argument("--snr", type=float, default=6.0)
    ap.add_argument("--raw", default=None, help="原始 jsPsych 檔（算整場用時）"); ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=500); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = C.out_dir(a.out); plt = C.setup_plot(); pid = a.participant; rng = np.random.default_rng(a.seed)
    d = C.load_perception(pid, [a.snr]); d = d[d.key.notna()].copy()
    if "x_acoustic" not in d:
        raise SystemExit("這個檔沒有 x_acoustic 欄：不是 online/calib_pilot 的資料")
    same, diff = d[d.is_same == 1].copy(), d[d.is_same == 0].copy()
    fa = float(1 - same.correct.mean()); x, y, v = diff.x_acoustic.values.astype(float), diff.correct.values.astype(float), diff.voicing_only.values.astype(float)
    faf = max(fa, 0.5 / len(same)); R = dict(participant=pid, snr=a.snr, n_different=int(len(diff)), n_same=int(len(same)), hit=float(y.mean()), false_alarm=fa)

    # ---- (1) 題目設計
    bins = diff.groupby("bin").agg(n=("correct", "size"), hit=("correct", "mean"), x=("x_acoustic", "mean"), voicing_n=("voicing_only", "sum")).reset_index()
    bins["hit_voicing"] = [diff[(diff.bin == b) & (diff.voicing_only == 1)].correct.mean() for b in bins.bin]
    bins["hit_other"] = [diff[(diff.bin == b) & (diff.voicing_only == 0)].correct.mean() for b in bins.bin]
    R["bins"] = bins.replace({np.nan: None}).to_dict("records")
    half = len(d) // 2; d["block"] = (d.trial - 1) // 81 + 1
    R["false_alarm_halves"] = [float(1 - same[same.trial <= half].correct.mean()), float(1 - same[same.trial > half].correct.mean())]
    R["hit_halves"] = [float(diff[diff.trial <= half].correct.mean()), float(diff[diff.trial > half].correct.mean())]
    R["by_block"] = [dict(block=int(b), n=int(len(g)), hit=float(g[g.is_same == 0].correct.mean()), false_alarm=float(1 - g[g.is_same == 1].correct.mean()),
                          rt_median=float(g.rt.median())) for b, g in d.groupby("block")]
    R["by_group"] = [dict(group=g, n_diff=int(len(q)), hit=float(q.correct.mean()), x_mean=float(q.x_acoustic.mean()),
                          false_alarm=float(1 - same[same.group == g].correct.mean()), n_same=int((same.group == g).sum())) for g, q in diff.groupby("group")]
    R["false_alarm_by_consonant"] = {C.ipa(c): [int((1 - g.correct).sum()), int(len(g))] for c, g in same.groupby("consonant")}
    R["rt"] = dict(median=float(d.rt.median()), same=float(same.rt.median()), different=float(diff.rt.median()),
                   hit=float(diff[diff.correct == 1].rt.median()), miss=float(diff[diff.correct == 0].rt.median()))
    if a.raw:
        with open(C.path(a.raw), newline="", encoding="utf-8-sig") as f:
            csv.field_size_limit(10**9); raw = list(csv.DictReader(f))
        el = [float(r["time_elapsed"]) for r in raw]; pe = [float(r["time_elapsed"]) for r in raw if r.get("task") == "pair"]
        R["minutes"] = dict(session=max(el) / 60000, pair_part=(pe[-1] - pe[0]) / 60000)

    # ---- (2) 換算曲線，全部「不同」題
    f_all = fit(x, y, faf); x90 = f_all["x50"] + f_all["s"] * np.log(9)
    q = (0.5 - fa) / (1 - fa - f_all["lam"]); x_half = f_all["x50"] + f_all["s"] * np.log(q / (1 - q)) if 0 < q < 1 else float("nan")
    boot = []
    for _ in range(a.boot):
        i = rng.integers(0, len(x), len(x)); b = fit_from(x[i], y[i], faf, f_all)
        qq = (0.5 - fa) / (1 - fa - b["lam"]); boot.append((b["x50"], b["x50"] + b["s"] * np.log(9), b["x50"] + b["s"] * np.log(qq / (1 - qq)) if 0 < qq < 1 else np.nan))
    boot = np.array(boot); ci = lambda k: [float(u) for u in np.nanpercentile(boot[:, k], [2.5, 97.5])]
    R["curve_all"] = dict(f_all, x90=float(x90), x_half=float(x_half), x50_ci95=ci(0), x90_ci95=ci(1), x_half_ci95=ci(2), x_range=[float(x.min()), float(x.max())])

    # ---- (3) 清濁對
    nv = v == 0; f_nv = fit(x[nv], y[nv], faf); p_v = P(x[v == 1], faf, f_nv["x50"], f_nv["s"], f_nv["lam"])
    sim = (rng.random((20000, len(p_v))) < p_v).sum(1); obs = int(y[v == 1].sum())
    d_hat, f_off = fit_offset(x, y, faf, v); lr = 2 * (f_all["nll"] - f_off["nll"])
    dboot = []
    for _ in range(min(a.boot, 200)):
        i = rng.integers(0, len(x), len(x)); dboot.append(fit_offset(x[i], y[i], faf, v[i], step=0.2, start=f_off)[0])
    vd = diff[diff.voicing_only == 1].copy(); vd["p_pred"] = p_v; vd["voiced_first"] = vd.consonant.isin(VOICED)
    per_pair = []
    for (pr,), g in vd.assign(pr=[pair_name(c, f) for c, f in zip(vd.consonant, vd.foil)]).groupby(["pr"]):
        per_pair.append(dict(pair=pr, n=int(len(g)), hits=int(g.correct.sum()), hit=float(g.correct.mean()), predicted=float(g.p_pred.mean()), x_mean=float(g.x_acoustic.mean()),
                             x_min=float(g.x_acoustic.min()), x_max=float(g.x_acoustic.max()), mn_dprime=float(g["count"].mean()), cls=CLASS[g.consonant.iloc[0]],
                             voiced_first=[int(g[g.voiced_first].correct.sum()), int(g.voiced_first.sum())], voiceless_first=[int(g[~g.voiced_first].correct.sum()), int((~g.voiced_first).sum())]))
    per_pair.sort(key=lambda r: r["x_mean"])
    # 同一個區間裡的非清濁題當對照
    matched = [dict(bin=int(b), voicing=[int(g[g.voicing_only == 1].correct.sum()), int((g.voicing_only == 1).sum())],
                    other=[int(g[g.voicing_only == 0].correct.sum()), int((g.voicing_only == 0).sum())]) for b, g in diff.groupby("bin") if (g.voicing_only == 1).any()]
    vd["cls"] = vd.consonant.map(CLASS)
    def two(col, k1, k2):   # 事後比較：兩組的命中 / 題數、曲線預測的命中數、Fisher 精確檢定（雙尾）
        g1, g2 = vd[vd[col] == k1], vd[vd[col] == k2]; t = [[int(g1.correct.sum()), int((1 - g1.correct).sum())], [int(g2.correct.sum()), int((1 - g2.correct).sum())]]
        return dict(hits=[t[0][0], t[1][0]], n=[len(g1), len(g2)], predicted=[float(g1.p_pred.sum()), float(g2.p_pred.sum())], x_mean=[float(g1.x_acoustic.mean()), float(g2.x_acoustic.mean())], fisher_p=float(fisher_exact(t)[1]))
    low = diff[diff.bin <= 1]; lt = [[int(low[low.voicing_only == k].correct.sum()), int((1 - low[low.voicing_only == k].correct).sum())] for k in (1, 0)]
    mn_floor = float(vd["count"].min()); hi = diff[(diff.voicing_only == 0) & (diff["count"] >= mn_floor)]   # M&N 表上和清濁對一樣「容易」的其他題
    mt = [[obs, int((v == 1).sum()) - obs], [int(hi.correct.sum()), int((1 - hi.correct).sum())]]
    post = dict(same_table_range_voicing_vs_other=dict(mn_floor=mn_floor, hits=[mt[0][0], mt[1][0]], n=[sum(mt[0]), sum(mt[1])], x_mean=[float(vd.x_acoustic.mean()), float(hi.x_acoustic.mean())], fisher_p=float(fisher_exact(mt)[1])),
                direction_voiced_first_vs_voiceless_first=two("voiced_first", True, False), stops_vs_fricatives=two("cls", "塞", "擦"),
                lowest_two_bins_voicing_vs_other=dict(hits=[lt[0][0], lt[1][0]], n=[sum(lt[0]), sum(lt[1])], fisher_p=float(fisher_exact(lt)[1])))
    R["voicing"] = dict(post_hoc=post, n=int((v == 1).sum()), hits=obs, hit=float(obs / (v == 1).sum()), expected_from_other_trials=float(p_v.sum()),
                        p_obs_le=float((sim <= obs).mean()), p_obs_ge=float((sim >= obs).mean()), curve_without_voicing=dict(f_nv, x90=float(f_nv["x50"] + f_nv["s"] * np.log(9)), n=int(nv.sum())),
                        offset_db=d_hat, offset_ci95=[float(u) for u in np.percentile(dboot, [2.5, 97.5])], lr=float(lr), p=float(chi2.sf(max(lr, 0), 1)),
                        per_pair=per_pair, matched_by_bin=matched,
                        by_direction={k: [int(g.correct.sum()), int(len(g))] for k, g in (("voiced_then_voiceless", vd[vd.voiced_first]), ("voiceless_then_voiced", vd[~vd.voiced_first]))},
                        by_group={g: [int(q.correct.sum()), int(len(q))] for g, q in vd.groupby("group")},
                        by_class={c: [int(q.correct.sum()), int(len(q))] for c, q in vd.groupby(vd.consonant.map(CLASS))},
                        misses_total=int((1 - y).sum()), misses_voicing=int((1 - y[v == 1]).sum()))

    # ---- (4) 排序力
    R["auc"] = dict(acoustic=auc(x, y), acoustic_ci95=boot_auc(x, y), acoustic_without_voicing=auc(x[nv], y[nv]), acoustic_without_voicing_ci95=boot_auc(x[nv], y[nv]),
                    acoustic_with_offset=auc(x + d_hat * v, y), mn=auc(diff["count"].values.astype(float), y), mn_ci95=boot_auc(diff["count"].values.astype(float), y),
                    mn_without_voicing=auc(diff["count"].values.astype(float)[nv], y[nv]),
                    features=auc(diff.feature_distance.values.astype(float), y), features_ci95=boot_auc(diff.feature_distance.values.astype(float), y),
                    misses_without_voicing=int((1 - y[nv]).sum()))

    # ---- (5) L / H 會選到什麼：全部 120 對 × 12 位說話者在這個 SNR 的值
    pc_all = pd.read_csv(os.path.join(C.OUT, "acoustic_scale_pairs.csv")); pc = pc_all[pc_all.snr.astype(str).isin([f"{a.snr:g}", f"{a.snr:.1f}"])]
    cand = pd.DataFrame([dict(a=r.a, b=r.b, talker=t, x=float(getattr(r, f"ex_{t}"))) for r in pc.itertuples() for t in TALKERS])
    cand["cls"] = ["-".join(sorted([CLASS[p], CLASS[q]], key="塞擦鼻".index)) for p, q in zip(cand.a, cand.b)]
    cand["voicing"] = [(p in VOICED) != (q in VOICED) and {p, q} in [{"p", "b"}, {"t", "d"}, {"k", "g"}, {"f", "v"}, {"theta", "eth"}, {"s", "z"}, {"sh", "zh"}] for p, q in zip(cand.a, cand.b)]
    sel = {}
    for name, target in (("L_at_P_half", x_half), ("L_at_midpoint", f_all["x50"]), ("H_at_x90", x90)):
        g = cand[(cand.x - target).abs() <= 0.3]
        top = g.assign(pr=[pair_name(p, q) for p, q in zip(g.a, g.b)]).pr.value_counts().head(8)
        sel[name] = dict(target=float(target), n_items=int(len(g)), n_pairs=int(g.groupby(["a", "b"]).ngroups), by_class=g.cls.value_counts().to_dict(),
                         voicing_items=int(g.voicing.sum()), with_b_or_v=int((g.a.isin(["b", "v"]) | g.b.isin(["b", "v"])).sum()), by_group=g.talker.map(GROUP_OF).value_counts().to_dict(), top_pairs=top.to_dict())
    xp = lambda p_: f_all["x50"] + f_all["s"] * np.log(((p_ - fa) / (1 - fa - f_all["lam"])) / (1 - (p_ - fa) / (1 - fa - f_all["lam"])))   # 曲線上 P = p_ 的 x
    lo_, hi_ = xp(0.7), xp(0.9); g = cand[(cand.x >= lo_) & (cand.x <= hi_)]
    R["x_at_p"] = {f"{p_:g}": float(xp(p_)) for p_ in (0.5, 0.7, 0.8, 0.9)}; R["candidates_p70_to_p90"] = dict(n_items=int(len(g)), n_pairs=int(g.groupby(["a", "b"]).ngroups), by_group=g.talker.map(GROUP_OF).value_counts().to_dict())
    R["candidates_below_x_half_by_snr"] = {str(k): int((g_[[f"ex_{t}" for t in TALKERS]].values < x_half).sum()) for k, g_ in pc_all.groupby(pc_all.snr.astype(str))}   # 假設曲線可跨 SNR 共用（沒有驗證）
    R["selection"] = sel; R["candidates_below_midpoint"] = int((cand.x < f_all["x50"]).sum()); R["candidates_above_x90"] = int((cand.x > x90).sum()); R["candidates"] = int(len(cand))
    json.dump(R, open(os.path.join(out, f"calib_pilot_report_{pid}.json"), "w"), ensure_ascii=False, indent=1)

    # ---- 印出來
    print(f"{pid}，{a.snr:+g} dB：「不同」{len(diff)} 題命中 {y.mean():.3f}（漏 {int((1 - y).sum())}）；「相同」{len(same)} 題誤報 {fa:.3f}（前半 {R['false_alarm_halves'][0]:.2f}、後半 {R['false_alarm_halves'][1]:.2f}）")
    if a.raw: print(f"  整場 {R['minutes']['session']:.1f} 分鐘，其中配對題 {R['minutes']['pair_part']:.1f} 分鐘；RT 中位數 {R['rt']['median']:.2f} s（命中 {R['rt']['hit']:.2f}、漏 {R['rt']['miss']:.2f}）")
    print("  區間  x(dB)   n  命中   其中清濁 n/命中   其餘命中")
    for r in bins.itertuples(): print(f"   {int(r.bin)}   {r.x:5.2f}  {r.n:3d}  {r.hit:.2f}    {int(r.voicing_n):2d} / {'' if np.isnan(r.hit_voicing) else format(r.hit_voicing, '.2f'):4s}     {r.hit_other:.2f}")
    print("  每 81 題一段：" + "；".join(f"第 {b['block']} 段 命中 {b['hit']:.2f} 誤報 {b['false_alarm']:.2f} RT {b['rt_median']:.2f}" for b in R["by_block"]))
    print("  三組聲音：" + "；".join(f"{g['group']} 命中 {g['hit']:.2f}（{g['n_diff']} 題，x 平均 {g['x_mean']:.1f}）誤報 {g['false_alarm']:.2f}" for g in R["by_group"]))
    print("  誤報依子音（錯 / 題數）：" + " ".join(f"{k} {v_[0]}/{v_[1]}" for k, v_ in R["false_alarm_by_consonant"].items() if v_[0]))
    c = R["curve_all"]; print(f"  曲線（全部）：x50 {c['x50']:.2f} [{c['x50_ci95'][0]:.2f}, {c['x50_ci95'][1]:.2f}]  s {c['s']:.2f}  λ {c['lam']:.3f}  x90 {c['x90']:.2f} [{c['x90_ci95'][0]:.2f}, {c['x90_ci95'][1]:.2f}]  P=.5 的點 {c['x_half']:.2f} [{c['x_half_ci95'][0]:.2f}, {c['x_half_ci95'][1]:.2f}]")
    V = R["voicing"]; cv = V["curve_without_voicing"]
    print(f"  曲線（不含清濁題，{cv['n']} 題）：x50 {cv['x50']:.2f}  s {cv['s']:.2f}  x90 {cv['x90']:.2f}")
    print(f"  清濁題 {V['n']} 題：命中 {V['hits']}（{V['hit']:.2f}）；其餘題的曲線預測 {V['expected_from_other_trials']:.1f} 題；P(實際 <= 預測分布) = {V['p_obs_le']:.3f}、P(>=) = {V['p_obs_ge']:.3f}")
    print(f"  偏移量 δ = {V['offset_db']:+.1f} dB [{V['offset_ci95'][0]:+.1f}, {V['offset_ci95'][1]:+.1f}]（正 = 清濁對比聲學尺說的容易）；概似比 {V['lr']:.2f}，p = {V['p']:.3f}")
    print("   對子   類  n 命中  實際  預測  x 平均 (範圍)      M&N d′")
    for r in per_pair: print(f"   {r['pair']:5s} {r['cls']}  {r['n']}  {r['hits']}   {r['hit']:.2f}  {r['predicted']:.2f}  {r['x_mean']:.2f} ({r['x_min']:.2f}-{r['x_max']:.2f})   {r['mn_dprime']:.2f}")
    print(f"   方向：濁→清 {V['by_direction']['voiced_then_voiceless']}、清→濁 {V['by_direction']['voiceless_then_voiced']}；三組：{V['by_group']}；塞 / 擦：{V['by_class']}；全部 {V['misses_total']} 個漏題裡清濁題佔 {V['misses_voicing']}")
    for k, lab in (("direction_voiced_first_vs_voiceless_first", "濁→清 對 清→濁"), ("stops_vs_fricatives", "塞音 對 擦音")):
        q_ = V["post_hoc"][k]; print(f"   事後比較 {lab}：命中 {q_['hits'][0]}/{q_['n'][0]} 對 {q_['hits'][1]}/{q_['n'][1]}（曲線預測 {q_['predicted'][0]:.1f} 對 {q_['predicted'][1]:.1f}；x 平均 {q_['x_mean'][0]:.2f} 對 {q_['x_mean'][1]:.2f}）Fisher p = {q_['fisher_p']:.3f}")
    q_ = V["post_hoc"]["lowest_two_bins_voicing_vs_other"]; print(f"   最低兩個區間：清濁題 {q_['hits'][0]}/{q_['n'][0]} 對 其餘 {q_['hits'][1]}/{q_['n'][1]}，Fisher p = {q_['fisher_p']:.3f}")
    q_ = V["post_hoc"]["same_table_range_voicing_vs_other"]; print(f"   M&N d′ >= {q_['mn_floor']:.2f} 的題：清濁題 {q_['hits'][0]}/{q_['n'][0]}（x 平均 {q_['x_mean'][0]:.2f}）對 其餘 {q_['hits'][1]}/{q_['n'][1]}（x 平均 {q_['x_mean'][1]:.2f}），Fisher p = {q_['fisher_p']:.4f}")
    print("   漏掉的題（x 由小到大）：" + "；".join(f"{C.ipa(r.consonant)}→{C.ipa(r.foil)} {r.talker} {r.x_acoustic:.2f}" for r in diff[diff.correct == 0].sort_values("x_acoustic").itertuples()))
    print("   逐對的兩個方向（濁→清 / 清→濁）：" + "；".join(f"{r['pair']} {r['voiced_first'][0]}/{r['voiced_first'][1]}、{r['voiceless_first'][0]}/{r['voiceless_first'][1]}" for r in per_pair))
    print("   同一區間的對照（命中 / 題數）：" + "；".join(f"區間 {m['bin']} 清濁 {m['voicing'][0]}/{m['voicing'][1]} 其餘 {m['other'][0]}/{m['other'][1]}" for m in matched))
    A_ = R["auc"]; print(f"  排序力：頻譜圖差別 {A_['acoustic']:.2f} {np.round(A_['acoustic_ci95'], 2).tolist()}；拿掉清濁題 {A_['acoustic_without_voicing']:.2f} {np.round(A_['acoustic_without_voicing_ci95'], 2).tolist()}（剩 {A_['misses_without_voicing']} 個漏題）；"
                        f"加 δ 後 {A_['acoustic_with_offset']:.2f}；M&N {A_['mn']:.2f} {np.round(A_['mn_ci95'], 2).tolist()}（拿掉清濁題 {A_['mn_without_voicing']:.2f}）；特徵數 {A_['features']:.2f} {np.round(A_['features_ci95'], 2).tolist()}")
    for k, s_ in sel.items(): print(f"  {k}：目標 {s_['target']:.2f} dB ± 0.3，{s_['n_items']} 個音檔對（{s_['n_pairs']} 對），類別 {s_['by_class']}，清濁 {s_['voicing_items']}，含 b 或 v 的 {s_['with_b_or_v']}，三組 {s_['by_group']}，最多的 {s_['top_pairs']}")
    print("  曲線上 P = .5 / .7 / .8 / .9 的 x：" + " / ".join(f"{v_:.2f}" for v_ in R["x_at_p"].values()) + f" dB；P 在 .70-.90 之間的音檔對 {R['candidates_p70_to_p90']['n_items']} 個（{R['candidates_p70_to_p90']['n_pairs']} 對，三組 {R['candidates_p70_to_p90']['by_group']}）")
    print(f"  x 低於 P = .5 的點（{x_half:.2f} dB）的音檔對，各 SNR：{R['candidates_below_x_half_by_snr']}（曲線只在 {a.snr:+g} dB 量過）")
    print(f"  全部 {R['candidates']} 個「對子 × 說話者」裡，低於曲線中點 {R['candidates_below_midpoint']}、高於 x90 {R['candidates_above_x90']}")

    # ---- 圖 1：換算曲線，清濁題另外標
    xs = np.linspace(0, 9, 300); fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.plot(xs, P(xs, faf, f_all["x50"], f_all["s"], f_all["lam"]), color="k", lw=2.2, label="全部「不同」題擬合的曲線")
    ax.plot(xs, P(xs, faf, f_nv["x50"], f_nv["s"], f_nv["lam"]), color=BLUE, lw=1.6, ls="--", label="只用非清濁題擬合的曲線")
    g = diff[diff.voicing_only == 0].groupby("bin").agg(x=("x_acoustic", "mean"), p=("correct", "mean"), n=("correct", "size"))
    ax.errorbar(g.x, g.p, np.sqrt(g.p * (1 - g.p) / g.n), fmt="o", color=BLUE, ms=7, capsize=2.5, lw=1, label="非清濁題（每 1 dB 一組 ± 1 SE）")
    for r in per_pair:
        ax.plot(r["x_mean"], r["hit"], "^", color=RED, ms=9, zorder=4); ax.annotate(r["pair"], (r["x_mean"], r["hit"]), textcoords="offset points", xytext=(6, -11), fontsize=9, color=RED)
    ax.plot([], [], "^", color=RED, ms=9, label="只差清濁的 7 對（每對 6 題）"); ax.plot(-0.3, fa, "s", mfc="white", mec="k", ms=7)
    ax.text(-0.17, fa, f"相同題的誤報 {fa:.2f}", fontsize=8.5, color=GREY, va="center")
    for val, lab, ha, dx in ((f_all["x50"], f"曲線中點 {f_all['x50']:.1f} dB", "right", -0.08), (x90, f"到頂九成 {x90:.1f} dB", "left", 0.08)):
        ax.axvline(val, color=GREY, ls=":", lw=1.2); ax.text(val + dx, 0.015, lab, color=GREY, fontsize=9, ha=ha)
    ax.set_xlabel(f"頻譜圖上的差別 x (dB，該題的說話者，{a.snr:+g} dB 噪音)"); ax.set_ylabel("回答「不一樣」的比例"); ax.set_ylim(0, 1.04); ax.set_xlim(-0.6, 9); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"{pid}：換算曲線；紅三角是只差清濁的對子", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, f"calib_pilot_curve_{pid}.png")); plt.close(fig)

    # ---- 圖 2：清濁對逐對，實際對預測
    fig, ax = plt.subplots(figsize=(8, 4.6)); xx = np.arange(len(per_pair))
    ax.bar(xx - 0.19, [r["hit"] for r in per_pair], 0.36, color=RED, alpha=0.85, label="實際命中率（每對 6 題）")
    ax.bar(xx + 0.19, [r["predicted"] for r in per_pair], 0.36, color=BLUE, alpha=0.85, label="非清濁題的曲線在同樣的 x 預測的命中率")
    for u, r in zip(xx, per_pair):
        ax.text(u - 0.19, r["hit"] + 0.015, f"{r['hits']}/{r['n']}", ha="center", fontsize=9); ax.text(u + 0.19, r["predicted"] + 0.015, f"{r['predicted']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels([f"{r['pair']}\nx = {r['x_mean']:.1f} dB\nM&N d' {r['mn_dprime']:.1f}" for r in per_pair], fontsize=9); ax.set_ylim(0, 1.18); ax.set_ylabel("命中率")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)
    ax.set_title(f"{pid}：只差清濁的 42 題，實際 {obs} 題命中，曲線預測 {p_v.sum():.1f} 題（依 x 由小到大排）", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"calib_pilot_voicing_{pid}.png")); plt.close(fig)

    # ---- 圖 3：八個區間
    fig, ax = plt.subplots(figsize=(8, 4.4)); xx = np.arange(len(bins))
    ax.bar(xx, bins.hit, 0.62, color=BLUE, alpha=0.85)
    for u, r in zip(xx, bins.itertuples()):
        ax.text(u, r.hit + 0.015, f"{r.hit:.2f}", ha="center", fontsize=9.5)
    lab = [f"{0.5 + r.bin:.1f}-{1.5 + r.bin:.1f}\n" + (f"清濁 {int(round(r.hit_voicing * r.voicing_n))}/{int(r.voicing_n)}\n其餘 {int(round(r.hit_other * (r.n - r.voicing_n)))}/{int(r.n - r.voicing_n)}" if r.voicing_n else f"清濁 0 題\n其餘 {int(round(r.hit * r.n))}/{int(r.n)}") for r in bins.itertuples()]
    ax.axhline(fa, color=RED, ls="--", lw=1.2, label=f"相同題的誤報 {fa:.2f}（答「不一樣」的下限）"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8.5); ax.set_xlabel(f"頻譜圖上的差別 x (dB)，每格 {int(bins.n.iloc[0])} 題；下面兩行是命中 / 題數"); ax.set_ylabel("命中率（「不同」題答對）"); ax.set_ylim(0, 1.22)
    ax.set_title(f"{pid}：八個聲學差別區間的命中率", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, f"calib_pilot_bins_{pid}.png")); plt.close(fig)

    # ---- 圖 4：和其他聽者的曲線疊在一起（各自在這個 SNR 的誤報）
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, P(xs, faf, f_all["x50"], f_all["s"], f_all["lam"]), color=RED, lw=2.4, label=f"{pid}（單一 SNR，{len(diff)} 題）：中點 {f_all['x50']:.1f}、到頂 {x90:.1f} dB，誤報 {fa:.2f}")
    for o, col in zip(a.others, (BLUE, GREY, ORA)):
        fo = os.path.join(C.OUT, f"acoustic_scale_curve_{o}.json")
        if not os.path.exists(fo): print(f"  （沒有 {fo}，略過 {o}）"); continue
        r = json.load(open(fo)); fo_fa = r["false_alarm"].get(f"{a.snr:g}", np.mean(list(r["false_alarm"].values())))
        ax.plot(xs, P(xs, max(fo_fa, 1e-3), r["x50"], r["s"], r["lam"]), color=col, lw=2, label=f"{o}（三個 SNR 合併，{r['n_different']} 題）：中點 {r['x50']:.1f}、到頂 {r['x90']:.1f} dB，誤報 {fo_fa:.2f}")
    ax.set_xlabel("頻譜圖上的差別 x (dB)"); ax.set_ylabel(f"回答「不一樣」的比例（{a.snr:+g} dB 的誤報當下限）"); ax.set_ylim(0, 1.04); ax.set_xlim(0, 9); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"{1 + len(a.others)} 位聽者的換算曲線（設計不同：見圖例）", loc="left", fontsize=10.5); fig.tight_layout(); fig.savefig(os.path.join(out, f"calib_pilot_curves_compared_{pid}.png")); plt.close(fig)
    print("寫出：", out)
