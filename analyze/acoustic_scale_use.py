"""聲學尺（三）：實際拿來用會怎樣——差別在時間上的位置、各 SNR 下 120 對的分布、會選到哪些對、對分析選擇有多敏感。

讀：analyze/out/acoustic_scale_feats.npz、acoustic_scale_pairs.csv、acoustic_scale_trials_<聽者>.csv
    （先跑 acoustic_scale_compute.py 與 acoustic_scale_validate.py）
算：- 差別在時間上的位置：每一格的平方差減掉噪音自己造成的，對 120 對與 12 位說話者平均
    - 敏感度：換視窗、不扣噪音底，聽者的逐題排序力（AUC）變多少
    - 頻譜圖差別 → 這位聽者命中率的邏輯曲線，以及由它換算的「難（命中 < .75）／易（> .90）」界線
    - 各 SNR 下 120 對的分位數、落在難 / 中 / 易的對數；樣板觀察者的中位 d′ 與正確率 < .90 的對數
    - +6、0 dB 時差別最小與最大的 10 對；M&N 表最難的 10 對在聲學尺上的名次；兩把尺最難 20 對的重疊
出：analyze/out/acoustic_scale_use.json、acoustic_scale_timecourse.csv
對應：review/proposal_2026-11-24_language_independent_sound_scale.md §1（分布表）、§3 (A)(B)。

注意：難 / 易的界線是用同一位聽者、同一批資料擬合出來的換算，只能當示意。聲學尺自己說不出絕對難度，
也說不出「該用哪個 SNR」；那要每位受試者自己的校準（企劃書的 (B)）。
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

import _common as C
from acoustic_scale_compute import NSEED, SNR_KEYS, T_POST, T_PRE, load_pairs
from acoustic_scale_validate import auc, load_mn, pair_key

IJ = list(itertools.combinations(range(16), 2))
WINDOWS = (("-300..+100 ms，扣噪音底（主分析）", slice(0, 40), True), ("-300..+100 ms，不扣", slice(0, 40), False),
           ("-200..+50 ms，扣噪音底", slice(10, 35), True), ("-300..+250 ms（整個音節），扣噪音底", slice(0, 55), True),
           ("0..+100 ms（只看過渡），扣噪音底", slice(30, 40), True), ("-300..0 ms（只看母音之前），扣噪音底", slice(0, 30), True))
TIME_BINS = ((-300, -100), (-100, 0), (0, 100), (120, 250))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_d = C.out_dir(a.out)
    Z = np.load(os.path.join(out_d, "acoustic_scale_feats.npz"))
    CONS, TALK = list(Z["cons"]), list(Z["talkers"])
    P = load_pairs(out_d)
    _, MN = load_mn()
    Y = pd.read_csv(os.path.join(out_d, f"acoustic_scale_trials_{a.participant}.csv"))
    Y["snr"] = Y.snr.astype(str)
    out = {}

    # ---- 差別在時間上的位置
    tc = {}
    for si, s in enumerate(SNR_KEYS):
        X = Z[f"s{si}"]
        fl = np.zeros(X.shape[3])
        if s != "clean":
            fl = np.mean([((X[:, :, p] - X[:, :, q]) ** 2).mean((0, 1, 3)) for p, q in itertools.combinations(range(NSEED), 2)], 0)
        d2 = np.mean([np.mean([((X[i, :, k] - X[j, :, (k + 1) % NSEED]) ** 2).mean((0, 2)) for k in range(NSEED)], 0) for i, j in IJ], 0)
        tc[s] = np.sqrt(np.clip(d2 - fl, 0, None))
    t_ms = np.arange(-T_PRE, T_POST) * 10
    pd.DataFrame({"t_ms": t_ms, **tc}).to_csv(os.path.join(out_d, "acoustic_scale_timecourse.csv"), index=False)
    print("差別在哪個時段（扣掉噪音自己的貢獻後的 rms dB；時間以母音起點為 0）：")
    out["timecourse_rms"] = {}
    for lo, hi in TIME_BINS:
        k = (t_ms >= lo) & (t_ms < hi)
        out["timecourse_rms"][f"{lo}..{hi}"] = {s: float(np.sqrt(np.mean(tc[s][k] ** 2))) for s in SNR_KEYS}
        print(f"   {lo:+5d}..{hi:+4d} ms: " + "  ".join(f"{s}: {out['timecourse_rms'][f'{lo}..{hi}'][s]:.1f}" for s in SNR_KEYS))

    # ---- 敏感度：聽者的比較會不會跟著視窗或扣不扣噪音底而變
    def pair_talker_measure(win, subtract):
        M = {}
        for si, s in enumerate(SNR_KEYS):
            X = Z[f"s{si}"]
            rms = lambda d: np.sqrt(np.mean(d[..., win, :] ** 2, axis=(-1, -2)))
            f2 = np.zeros((16, 12))
            if s != "clean" and subtract:
                f2 = np.mean([rms(X[:, :, p] - X[:, :, q]) ** 2 for p, q in itertools.combinations(range(NSEED), 2)], 0)
            for i, j in IJ:
                D = np.mean([rms(X[i, :, k] - X[j, :, (k + 1) % NSEED]) for k in range(NSEED)], 0)
                e = np.sqrt(np.clip(D ** 2 - (f2[i] + f2[j]) / 2, 0, None))
                for ti, tk in enumerate(TALK):
                    M[(pair_key(CONS[i], CONS[j]), s, tk)] = e[ti]
        return M
    print(f"\n敏感度（{a.participant} 逐題 AUC，n = {len(Y)}；第一列是主分析）：")
    out["sensitivity"] = {}
    for lab, win, sub in WINDOWS:
        M = pair_talker_measure(win, sub)
        out["sensitivity"][lab] = auc([M[(p, s, t)] for p, s, t in zip(Y.pair, Y.snr, Y.talker)], Y.correct)
        print(f"   {lab:36s} AUC {out['sensitivity'][lab]:.2f}")

    # ---- 頻譜圖差別 → 這位聽者的命中率（邏輯曲線，牛頓法）
    x, y = Y.ac_excess_talker.values, Y.correct.values.astype(float)
    b = np.zeros(2)
    Xd = np.c_[np.ones_like(x), x]
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xd @ b))
        b += np.linalg.solve(Xd.T @ (Xd * (p * (1 - p))[:, None]), Xd.T @ (y - p))
    inv = lambda q: (np.log(q / (1 - q)) - b[0]) / b[1]
    out["logistic"] = dict(b0=float(b[0]), b1=float(b[1]), x60=float(inv(.6)), x75=float(inv(.75)), x90=float(inv(.9)))
    print(f"\n{a.participant}: P(命中) = logistic({b[0]:.2f} + {b[1]:.2f} × 差別)；命中 .60 在 {inv(.6):.1f} dB、.75 在 {inv(.75):.1f}、.90 在 {inv(.9):.1f}")

    # ---- 各 SNR 下 120 對的分布
    print("\n120 對的分布（差別，dB）與落在這位聽者「難（< .75）」「易（> .90）」區的對數：")
    out["distribution"] = {}
    for s in SNR_KEYS:
        q_ = P[P.snr == s]
        e, dp = q_.excess.values, q_.dprime.values
        qs = np.percentile(e, [5, 25, 50, 75, 95])
        nh, ne = int((e < inv(.75)).sum()), int((e > inv(.9)).sum())
        out["distribution"][s] = dict(q=qs.tolist(), hard=nh, easy=ne, dprime_median=float(np.median(dp)), acc_lt_90=int((q_.acc < .9).sum()))
        print(f"   {s:>5}: 5/25/50/75/95 % = " + " ".join(f"{v:4.1f}" for v in qs) + f"   難 {nh:3d}  中 {120 - nh - ne:3d}  易 {ne:3d}"
              f"   | 樣板觀察者：中位 d′ {np.median(dp):.1f}，正確率 < .90 的對數 {(q_.acc < .9).sum()}")

    # ---- 會選到哪些對
    for s in ("6.0", "0.0"):
        q_ = P[P.snr == s].set_index("pair").join(MN).sort_values("excess")
        fmt = lambda rows: ", ".join(f"{p} {r.excess:.1f}|{r.mn_dprime:.2f}|{int(r.features)}" for p, r in rows.iterrows())
        print(f"\n{s} dB 時差別最小的 10 對（差別 dB | M&N d′ | 特徵數）：\n   " + fmt(q_.head(10)) + "\n   最大的 10 對：\n   " + fmt(q_.tail(10)))
        out[f"lowest@{s}"] = q_.head(10)[["excess", "mn_dprime", "features"]].reset_index().to_dict("records")
        out[f"highest@{s}"] = q_.tail(10)[["excess", "mn_dprime", "features"]].reset_index().to_dict("records")
    q_ = P[P.snr == "6.0"].set_index("pair").join(MN)
    q_["rank_ac"] = q_.excess.rank().astype(int)
    low_mn = q_.sort_values("mn_dprime").head(10)
    print("\nM&N 表最難的 10 對，在聲學尺（+6 dB）的名次（1 = 差別最小，共 120）：\n   "
          + ", ".join(f"{p}（M&N {r.mn_dprime:.2f} → 第 {r.rank_ac} 名，{r.excess:.1f} dB）" for p, r in low_mn.iterrows()))
    out["mn_hardest_in_acoustic_rank"] = low_mn[["mn_dprime", "excess", "rank_ac"]].reset_index().to_dict("records")
    out["overlap_bottom20"] = len(set(q_.sort_values("excess").head(20).index) & set(q_.sort_values("mn_dprime").head(20).index))
    print(f"   兩把尺各自最難的 20 對，重疊 {out['overlap_bottom20']} 對")
    voi = [pair_key(*p) for p in (("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("theta", "eth"), ("s", "z"), ("sh", "zh"))]
    out["voicing_pairs@6.0"] = q_.loc[voi, ["excess", "mn_dprime", "rank_ac"]].reset_index().to_dict("records")
    print("   只差清濁的 7 對（+6 dB）：" + ", ".join(f"{p} {q_.excess[p]:.1f} dB（第 {q_.rank_ac[p]} 名；M&N {q_.mn_dprime[p]:.2f}）" for p in voi))
    h = Y.groupby("pair").agg(hit=("correct", "mean"), n=("correct", "size"), ex=("ac_excess_talker", "mean"), mn=("mn_dprime", "mean")).sort_values("n", ascending=False).head(12)
    print(f"\n{a.participant} 聽最多次的 12 對：命中率、頻譜圖差別（該題的 SNR 與說話者平均）、M&N d′"); print(h.round(2).to_string())
    out["listener_top_pairs"] = h.reset_index().to_dict("records")
    json.dump(out, open(os.path.join(out_d, "acoustic_scale_use.json"), "w"), indent=1, default=float, ensure_ascii=False)
    print("\n寫出：acoustic_scale_use.json、acoustic_scale_timecourse.csv")


if __name__ == "__main__":
    main()
