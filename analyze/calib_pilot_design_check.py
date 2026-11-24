"""線上校準前測（online/calib_pilot/）的題目設計檢查：用模擬聽者看這份題目能不能把換算曲線估回來。

    python analyze/calib_pilot_design_check.py                      # 200 場模擬，印出估回來的 x50 / x90
    python analyze/calib_pilot_design_check.py --write-session /tmp/SIMCAL1_calib_pilot.csv --session-seed 1
        # 另外寫出一場「瀏覽器格式」的資料檔，可以丟給 online/pavlovia_to_perception_csv.py 走完整條分析管線

讀：online/calib_pilot/conditions.json（216 題不同 + 108 題相同，每題的 x_acoustic）
    analyze/out/acoustic_scale_trials_WEB01.csv、data/WEB01_sound_perception_dprime_snr*.csv（只用來抽 RT）
算：模擬聽者 = WEB01 的換算曲線（預設 x50 1.98、s 0.57、λ .017、誤報 .08；review/pilot_2026-11-24_WEB01_snr_pilot.md §4）
      「不同」題：P(答不一樣) = FA + (1 − FA − λ) / (1 + exp(−(x − x50)/s))；「相同」題：P(答不一樣) = FA
    每場用 acoustic_scale_listener_curve.fit 擬合（誤報用該場「相同」題的實際比例），收集 x50、x90、逐題排序力（AUC）。
    RT 只在 --write-session 時產生：從 WEB01 同樣對錯、x 相近的題目裡抽（fit_points 要用 RT）。
出：analyze/out/calib_pilot_design_check.json；印出摘要。
對應：online/calib_pilot/README.md 的「驗證」。

注意：這是模擬，而且假設聽者就是這條曲線、每題獨立。它回答的是「題目的 x 怎麼排、有幾題」夠不夠，
不是「真人做不做得出來」。斜率更平或誤報更高的聽者，區間會更寬（--x50 --s --lam --fa 可以改）。
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

import _common as C
from acoustic_scale_listener_curve import fit
from acoustic_scale_validate import auc

PLAN = os.path.join(C.ROOT, "online", "calib_pilot", "conditions.json")


def p_diff(x, x50, s, lam, fa):
    return fa + (1 - fa - lam) / (1 + np.exp(-(x - x50) / s))


def simulate(trials, rng, x50, s, lam, fa):
    """每題一個 0/1：有沒有答「不一樣」。"""
    said = []
    for t in trials:
        p = fa if t["is_same"] else p_diff(t["x_acoustic"], x50, s, lam, fa)
        said.append(int(rng.random() < p))
    return np.array(said)


def rt_pool():
    """WEB01 的 RT（秒），依「相同/不同」「對/錯」「x 的四分位」分池；模擬的 RT 從對應的池抽。"""
    t = pd.read_csv(os.path.join(C.OUT, "acoustic_scale_trials_WEB01.csv"))
    t["q"] = pd.qcut(t.ac_excess_talker, 4, labels=False)
    edges = t.groupby("q").ac_excess_talker.max().values[:-1]
    d = C.load_perception("WEB01", [0, 6, 12])
    same = d[(d.is_same == 1) & d.key.notna()]
    pool = {("same", int(c)): g.rt.values for c, g in same.groupby("correct")}
    for (q, c), g in t.groupby(["q", "correct"]):
        pool[("diff", int(q), int(c))] = g.rt.values
    return pool, edges


def write_session(path, trials, said, rng, cond, pid):
    """一場模擬資料，欄位與 online/calib_pilot/experiment.js 存出來的一樣（只放轉檔程式會讀的欄）。"""
    pool, edges = rt_pool()
    order = rng.permutation(len(trials))
    rows = []
    for k in range(1, 7):
        rows.append(dict(task="hpcheck", trial=k, correct=1))
    for k, i in enumerate(order, 1):
        t = trials[i]; s = int(said[i]); correct = int((s == 0) == bool(t["is_same"]))
        if t["is_same"]:
            cand = pool.get(("same", correct), pool[("same", 1)])
        else:
            q = int(np.searchsorted(edges, t["x_acoustic"]))
            cand = pool.get(("diff", q, correct), pool[("diff", q, 1)])
        rows.append(dict(task="pair", trial=k, key="n" if s else "y", response="n" if s else "y", rt=int(round(1000 * float(rng.choice(cand)))), correct=correct,
                         **{c: ("null" if t[c] is None else t[c]) for c in ("row", "is_same", "consonant", "foil", "talker", "group", "pair", "pair_type", "voicing_only",
                                                                           "x_acoustic", "x_acoustic_pair", "bin", "mn_dprime", "feature_distance", "snd1", "snd2", "seed1", "seed2")}))
    props = dict(participant=pid, design=cond["design"], plan_seed=cond["plan_seed"], snr_db=cond["snr_db"], order_seed="simulated",
                 native_language="(simulated listener)", english_at_home="", english_start_age="", test_maxtrials="", user_agent="analyze/calib_pilot_design_check.py")
    cols = list(props) + sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({**props, **r})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="線上校準前測的題目設計檢查（模擬聽者）。")
    ap.add_argument("--n", type=int, default=200, help="模擬幾場"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x50", type=float, default=1.98); ap.add_argument("--s", type=float, default=0.57)
    ap.add_argument("--lam", type=float, default=0.017); ap.add_argument("--fa", type=float, default=0.08)
    ap.add_argument("--write-session", default=None, help="另外寫出一場瀏覽器格式的資料檔"); ap.add_argument("--session-seed", type=int, default=1)
    ap.add_argument("--session-id", default="SIMCAL1"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = C.out_dir(a.out)
    cond = json.load(open(PLAN, encoding="utf-8")); trials = cond["trials"]
    is_same = np.array([t["is_same"] for t in trials]).astype(bool)
    x = np.array([t["x_acoustic"] if t["x_acoustic"] is not None else np.nan for t in trials])
    truth = dict(x50=a.x50, s=a.s, lam=a.lam, fa=a.fa, x90=a.x50 + a.s * np.log(9))

    rng = np.random.default_rng(a.seed); est = []
    for _ in range(a.n):
        said = simulate(trials, rng, a.x50, a.s, a.lam, a.fa)
        fa_hat = max(said[is_same].mean(), 0.5 / is_same.sum())
        f = fit(x[~is_same], said[~is_same].astype(float), np.full((~is_same).sum(), fa_hat))
        est.append((f["x50"], f["x50"] + f["s"] * np.log(9), auc(x[~is_same], said[~is_same]), said[~is_same].mean(), said[is_same].mean()))
    E = np.array(est)
    q = lambda v: [float(np.percentile(v, p)) for p in (5, 50, 95)]
    res = dict(truth=truth, n_sessions=a.n, n_different=int((~is_same).sum()), n_same=int(is_same.sum()),
               x50=q(E[:, 0]), x90=q(E[:, 1]), auc=q(E[:, 2]), hit=q(E[:, 3]), false_alarm=q(E[:, 4]),
               x50_width90=float(np.percentile(E[:, 0], 95) - np.percentile(E[:, 0], 5)), x90_width90=float(np.percentile(E[:, 1], 95) - np.percentile(E[:, 1], 5)),
               x90_within_1db=float(np.mean(np.abs(E[:, 1] - truth["x90"]) < 1.0)))
    json.dump(res, open(os.path.join(out, "calib_pilot_design_check.json"), "w"), ensure_ascii=False, indent=1)
    print(f"模擬聽者：x50 {a.x50}、s {a.s}、λ {a.lam}、誤報 {a.fa}（真值 x90 = {truth['x90']:.2f} dB）；{a.n} 場，每場 {res['n_different']} 不同 + {res['n_same']} 相同")
    print(f"  估回來的 x50：中位數 {res['x50'][1]:.2f}，中間 90% [{res['x50'][0]:.2f}, {res['x50'][2]:.2f}]（寬 {res['x50_width90']:.2f} dB）")
    print(f"  估回來的 x90：中位數 {res['x90'][1]:.2f}，中間 90% [{res['x90'][0]:.2f}, {res['x90'][2]:.2f}]（寬 {res['x90_width90']:.2f} dB）；落在真值 ±1 dB 內的比例 {res['x90_within_1db']:.2f}")
    print(f"  逐題排序力（AUC）：中位數 {res['auc'][1]:.2f} [{res['auc'][0]:.2f}, {res['auc'][2]:.2f}]；命中 {res['hit'][1]:.2f}；誤報 {res['false_alarm'][1]:.2f}")
    if a.write_session:
        r2 = np.random.default_rng(a.session_seed)
        write_session(a.write_session, trials, simulate(trials, r2, a.x50, a.s, a.lam, a.fa), r2, cond, a.session_id)
        print("  寫出一場模擬資料：", a.write_session)
    print("寫出：", os.path.join(out, "calib_pilot_design_check.json"))
