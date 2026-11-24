"""聲學尺（五）：驗證實驗的設計檢查——建議的對子清單，以及「如果兩個族群真的用同一條曲線／不是同一條」時各項判準會長什麼樣。

讀：analyze/out/acoustic_scale_pairs.csv、acoustic_scale_use.json（先跑 compute、validate、use）
算：(1) 建議的 40 個無序對：7 個只差清濁的對全收（現有資料只測到 3 題），其餘 113 對照 +6 dB 的頻譜圖差別排序後等距取 33 對
    (2) 模擬（預設 500 次）：每組 6 人、每人 40 對 × 2 個 SNR（+6、+12）× 6 次 =「不同」試 480 題
        每人 P(答不一樣) = logistic(b0 + 個人位移 + b1 × 差別)，b0、b1 取 use.json 裡那位聽者的擬合值，個人位移 ~ N(0, 0.5)
        情境 same ：兩組同一條曲線
        情境 shift：第二組整條曲線往難的方向平移（logit -1.0）——「只差位置」，尺仍然不論母語
        情境 voice：第二組只在「只差清濁」的 7 對上變難（logit -2.0）——「排序變了」，尺不是不論母語
      報：每人逐題排序力（AUC）、組內折半與兩組之間的逐對命中率 Spearman ρ、清濁對相對於共同曲線的殘差（兩組之差）
出：analyze/out/acoustic_scale_design_pairs.csv、acoustic_scale_design_check.json
對應：review/proposal_2026-11-24_language_independent_sound_scale.md §4（判準的門檻從哪來）、§5、附錄 A。

為什麼要模擬：判準要事先定，但門檻不能憑感覺——逐對命中率有抽樣誤差，就算兩組真的一樣，ρ 也到不了 1。
這裡算的是「真的一樣時 ρ 大概多少、真的不一樣時掉到多少」，門檻訂在兩者之間。模擬的曲線來自一位聽者的事後擬合，
所以這些數字只是量級；真實資料若比模擬吵（例如說話者差異），門檻要在收資料前重跑這支程式更新，不是收完再調。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C
from acoustic_scale_compute import load_pairs
from acoustic_scale_validate import auc, load_mn, pair_key

VOICING = [pair_key(*p) for p in (("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("theta", "eth"), ("s", "z"), ("sh", "zh"))]
SNRS = ("6.0", "12.0")
N_OTHER, N_PER_GROUP, REPS = 33, 6, 6


def choose_pairs(P):
    q = P[P.snr == "6.0"].set_index("pair").sort_values("excess")
    rest = [p for p in q.index if p not in VOICING]
    idx = np.round(np.linspace(0, len(rest) - 1, N_OTHER)).astype(int)
    return VOICING + [rest[i] for i in idx]


def simulate(x, is_voi, b0, b1, scenario, rng):
    """x[pair, snr] = 頻譜圖差別。回傳兩組的命中矩陣 [人, 對, snr, 次]。"""
    out = []
    for g in range(2):
        shift = rng.normal(0, 0.5, N_PER_GROUP)[:, None, None]
        eta = b0 + shift + b1 * x[None]
        if g == 1 and scenario == "shift": eta = eta - 1.0
        if g == 1 and scenario == "voice": eta = eta - 2.0 * is_voi[None, :, None]
        p = 1 / (1 + np.exp(-eta))
        out.append((rng.random(p.shape + (REPS,)) < p[..., None]).astype(float))
    return out


def summarise(H, x, is_voi):
    """H = [組][人, 對, snr, 次]。"""
    r = {}
    xs = np.broadcast_to(x[None, :, :, None], H[0].shape)
    r["auc_person"] = [auc(xs[i].ravel(), h[i].ravel()) for h in H for i in range(N_PER_GROUP)]
    pair_hit = [h.mean((0, 2, 3)) for h in H]                                   # 每組每對（合併人、SNR、次）
    r["rho_between"] = float(spearmanr(pair_hit[0], pair_hit[1])[0])
    half = [H[0][::2].mean((0, 2, 3)), H[0][1::2].mean((0, 2, 3))]              # 第一組折半（3 人對 3 人）
    r["rho_split_half"] = float(spearmanr(half[0], half[1])[0])
    # 清濁對的殘差：每組各自用非清濁對擬合 logit(命中) ~ 差別（對 × SNR 的格平均），再看清濁對比預測低多少
    res = []
    for h in H:
        cell = np.clip(h.mean((0, 3)), .02, .98)
        lg = np.log(cell / (1 - cell))
        A = np.c_[np.ones((~is_voi).sum() * x.shape[1]), x[~is_voi].ravel()]
        coef = np.linalg.lstsq(A, lg[~is_voi].ravel(), rcond=None)[0]
        res.append(float((lg[is_voi] - (coef[0] + coef[1] * x[is_voi])).mean()))
    r["voicing_residual_diff"] = res[1] - res[0]                               # 第二組減第一組（logit）；負 = 第二組在清濁對上特別差
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nsim", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_d = C.out_dir(a.out)
    P = load_pairs(out_d)
    _, MN = load_mn()
    U = json.load(open(os.path.join(out_d, "acoustic_scale_use.json")))
    b0, b1 = U["logistic"]["b0"], U["logistic"]["b1"]
    pairs = choose_pairs(P)
    tab = pd.DataFrame({"pair": pairs, "voicing_only": [p in VOICING for p in pairs]}).set_index("pair")
    for s in ("12.0", "6.0", "0.0"):
        tab[f"excess@{s}"] = P[P.snr == s].set_index("pair").excess.reindex(pairs).round(2)
    tab = tab.join(MN)
    tab.to_csv(os.path.join(out_d, "acoustic_scale_design_pairs.csv"))
    cons = pd.Series([c for p in pairs for c in p.split("-")]).value_counts()
    print(f"(1) 建議的 {len(pairs)} 對：清濁 {len(VOICING)} 對 + 等距取 {N_OTHER} 對。+6 dB 的差別範圍 {tab['excess@6.0'].min():.1f}-{tab['excess@6.0'].max():.1f} dB；"
          f"每個子音出現 {cons.min()}-{cons.max()} 次（最少：{', '.join(cons[cons == cons.min()].index)}）")
    print(tab.to_string())

    x = np.stack([tab[f"excess@{s}"].values for s in SNRS], 1)
    is_voi = tab.voicing_only.values
    rng = np.random.default_rng(a.seed)
    res = {"pairs": pairs, "n_pairs": len(pairs), "per_person_different_trials": int(len(pairs) * len(SNRS) * REPS), "b0": b0, "b1": b1, "nsim": a.nsim, "scenarios": {}}
    print(f"\n(2) 模擬 {a.nsim} 次；每組 {N_PER_GROUP} 人，每人「不同」試 {res['per_person_different_trials']} 題。中位數 [5%, 95%]")
    for sc in ("same", "shift", "voice"):
        S = [summarise(simulate(x, is_voi, b0, b1, sc, rng), x, is_voi) for _ in range(a.nsim)]
        q = lambda v: [float(np.percentile(v, k)) for k in (50, 5, 95)]
        r = dict(auc_person=q(np.concatenate([s["auc_person"] for s in S])), rho_between=q([s["rho_between"] for s in S]),
                 rho_split_half=q([s["rho_split_half"] for s in S]), voicing_residual_diff=q([s["voicing_residual_diff"] for s in S]),
                 ratio_between_over_split=q([s["rho_between"] / s["rho_split_half"] for s in S]))
        res["scenarios"][sc] = r
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
        print(f"   {sc:5s}  每人 AUC {f(r['auc_person'])}   組間 ρ {f(r['rho_between'])}   組內折半 ρ {f(r['rho_split_half'])}   "
              f"組間÷折半 {f(r['ratio_between_over_split'])}   清濁殘差（第二組-第一組，logit）{f(r['voicing_residual_diff'])}")
    json.dump(res, open(os.path.join(out_d, "acoustic_scale_design_check.json"), "w"), indent=1, ensure_ascii=False)
    print("\n寫出：acoustic_scale_design_pairs.csv、acoustic_scale_design_check.json")


if __name__ == "__main__":
    main()
