"""聲學尺（二）：它是不是一把有用的尺？對三樣東西檢查。

讀：analyze/out/acoustic_scale_pairs.csv（先跑 acoustic_scale_compute.py）
    stimuli/consonant_dprime.csv（現在用的尺：Miller & Nicely 的 Luce 距離換成 d′）與 consonant_distance.luce_distance 的各 SNR 矩陣
    data/<非母語聽者>_sound_perception_dprime_snr<dB>.csv（預設 yyc1116，0 / 6 / 12 dB）
    data/EN01_s00*_simple_calib_trials.csv 的 audio 維度（英語母語，乾淨條件）
算：(1) 聲學尺對 M&N：120 個無序對的 Pearson r 與 Spearman ρ（表、各 SNR 的 M&N 矩陣、特徵數）
    (2) 對非母語聽者的「不同」試，逐題比三把尺：
        排序力 = 答對的題分數高於答錯的題的機率（AUC；0.5 = 排不出來），加拔靴 95% 區間（2000 次，seed 0）
        邏輯迴歸 AIC：命中 ~ SNR + 尺；頻譜圖差別四等分的命中率
    (3) 對 EN01：校準檔只記了探測音與那一對的 d′，沒記學習的那個音，所以用「探測音的子音 + d′ 值」反推是哪一對，
        反推出唯一一對的才用
出：analyze/out/acoustic_scale_validation.json、acoustic_scale_trials_<聽者>.csv（每題一列，附三把尺的分數）
對應：review/proposal_2026-11-24_language_independent_sound_scale.md §1 的表。

為什麼用 AUC 而不是只看相關：每一對的題數差很多（1 到 16 題），逐對命中率的 r 會被只有一兩題的對子主導；
AUC 逐題算，不必先把題目合併成對。兩者都報。
注意：這些比較都是事後的，只有一位非母語聽者；尺的視窗與頻帶是先定的，沒有對著這份資料調（敏感度見 acoustic_scale_use.py）。
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import _common as C
import consonant_distance as cd
from acoustic_scale_compute import SNR_KEYS, load_pairs

pair_key = lambda a, b: "-".join(sorted([a, b]))
RULERS = (("mn_dprime", "M&N d′（現在的尺）"), ("features", "差幾個發音特徵"), ("ac_excess", "頻譜圖差別（整對，12 位平均）"),
          ("ac_excess_talker", "頻譜圖差別（該題的說話者）"), ("ac_dprime", "樣板觀察者 d′"))


def load_mn():
    """現在的尺，合成無序對：mn_dprime = 兩個方向的平均，features = 差幾個特徵。回傳 (有序表, 無序表)。"""
    T = pd.read_csv(os.path.join(C.ROOT, "stimuli", "consonant_dprime.csv"))
    T["pair"] = [pair_key(a, b) for a, b in zip(T.target, T.foil)]
    return T, T.groupby("pair").agg(mn_dprime=("dprime", "mean"), features=("feature_distance", "first"))


def auc(score, y):
    """答對的題分數高於答錯的題的機率（平手算一半）。"""
    score, y = np.asarray(score), np.asarray(y)
    s1, s0 = score[y == 1], score[y == 0]
    if len(s1) == 0 or len(s0) == 0:
        return np.nan
    return float((s1[:, None] > s0[None, :]).mean() + 0.5 * (s1[:, None] == s0[None, :]).mean())


def boot_auc(score, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    score, y = np.asarray(score), np.asarray(y)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = auc(score[i], y[i])
        if not np.isnan(a):
            v.append(a)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def logistic(X, y, iters=60):
    """最大概似的邏輯迴歸（牛頓法）。回傳 (係數, AIC)。"""
    X = np.c_[np.ones(len(y)), X]
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        b += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(b)), X.T @ (y - p))
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    return b, float(2 * len(b) - 2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def listener_trials(participant, snrs, P):
    """非母語聽者的「不同」試，每題附上三把尺的分數。檔案照檔名排序讀（拔靴的抽樣順序因此固定）。"""
    files = sorted(os.path.join(C.DATA, f"{participant}_sound_perception_dprime_snr{s:g}.csv") for s in snrs)
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[d.key.notna()]                                           # 沒按鍵的不進分母（_common 的共用規則）
    df = d[d.is_same == 0].copy()
    df["pair"] = [pair_key(a, b) for a, b in zip(df.consonant, df.foil)]
    df["snr"] = df.snd_snr_db.astype(float).astype(str)
    Pi = P.set_index(["pair", "snr"])
    df["ac_excess"] = [Pi.loc[(p, s), "excess"] for p, s in zip(df.pair, df.snr)]
    df["ac_dprime"] = [Pi.loc[(p, s), "dprime"] for p, s in zip(df.pair, df.snr)]
    df["ac_excess_talker"] = [Pi.loc[(p, s), f"ex_{t}"] for p, s, t in zip(df.pair, df.snr, df.talker)]   # 同一對、這位說話者的兩個音檔
    df["mn_dprime"] = df["count"]                                  # sound_perception_EN.py 把該題實際的 d′ 記在 count 欄
    df["features"] = df.feature_distance
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116", help="非母語聽者的 SNR pilot")
    ap.add_argument("--snrs", type=float, nargs="*", default=[0, 6, 12])
    ap.add_argument("--native", default="data/EN01_s00*_simple_calib_trials.csv", help="母語聽者的校準 trials（glob）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_d = C.out_dir(a.out)
    P = load_pairs(out_d)
    T, MN = load_mn()
    out = {"n_pairs": int(MN.shape[0])}

    # ---- 1. 對 Miller & Nicely
    print("1. 對 Miller & Nicely（120 個無序對）。r = Pearson，rho = Spearman")
    out["mn"] = {}
    for snr in SNR_KEYS:
        q = P[P.snr == snr].set_index("pair").join(MN)
        line = f"  我們的 {snr:>5}："
        for m in ("excess", "dprime"):
            r, rho = pearsonr(q[m], q.mn_dprime)[0], spearmanr(q[m], q.mn_dprime)[0]
            out["mn"][f"{m}@{snr}_vs_table"] = [r, rho]
            line += f"  {m} 對 M&N 表 r={r:.2f} rho={rho:.2f} |"
        rf = spearmanr(q.excess, q.features)[0]
        out["mn"][f"excess@{snr}_vs_features"] = rf
        print(line + f" excess 對特徵數 rho={rf:.2f}")
    print("   同名目 SNR（M&N 是白噪音，我們是語音形噪音：同樣的 dB 不是同樣的遮蔽）：")
    iu = np.triu_indices(16, 1)
    names = [pair_key(cd.CONSONANTS[i], cd.CONSONANTS[j]) for i, j in zip(*iu)]
    for mn_snr in (12, 6, 0, -6, -12):
        L = pd.Series(cd.luce_distance(snrs=(mn_snr,))[iu], index=names)
        line = f"   M&N {mn_snr:+3d} dB："
        for snr in SNR_KEYS[1:]:
            q = P[P.snr == snr].set_index("pair")
            rho = spearmanr(q.excess, L.reindex(q.index))[0]
            out["mn"][f"excess@{snr}_vs_mn@{mn_snr}"] = rho
            line += f"  我們 {snr:>4}: rho={rho:.2f}"
        print(line)

    # ---- 2. 非母語聽者，逐題
    df = listener_trials(a.participant, a.snrs, P)
    df.to_csv(os.path.join(out_d, f"acoustic_scale_trials_{a.participant}.csv"), index=False)
    y = df.correct.values.astype(float)
    print(f"\n2. {a.participant}「不同」試：n = {len(df)}，命中 {int(y.sum())}，漏 {int((1 - y).sum())}")
    out["listener"] = {"participant": a.participant, "n": int(len(df)), "misses": int((1 - y).sum())}
    snr_num = df.snd_snr_db.astype(float).values
    one_snr = len(set(snr_num)) == 1                               # 單一 SNR 的場次（online/calib_pilot）：SNR 是常數，不能放進迴歸
    for m, lab in RULERS:
        au, ci, r = auc(df[m], y), boot_auc(df[m], y), pearsonr(df[m], y)[0]
        by = {s: auc(g[m], g.correct) for s, g in df.groupby("snr")}
        if one_snr:
            b, aic = logistic(df[[m]].values, y)
            b = np.array([b[0], np.nan, b[1]])
        else:
            b, aic = logistic(np.c_[snr_num, df[m].values], y)
        out["listener"][m] = dict(auc=au, ci=ci, r=r, by_snr=by, aic_snr_plus_ruler=aic, coef_snr=float(b[1]), coef_ruler=float(b[2]))
        print(f"   {lab:26s} AUC {au:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  r={r:+.2f}  AIC({'尺' if one_snr else 'SNR + 尺'}) {aic:6.1f}  SNR 係數 {b[1]:+.3f}   各 SNR " + " ".join(f"{s}:{v:.2f}" for s, v in by.items()))
    out["listener"]["aic_snr_only"] = logistic(np.empty((len(y), 0)) if one_snr else snr_num[:, None], y)[1]
    print(f"   {'只有截距' if one_snr else '只放 SNR'} 的 AIC {out['listener']['aic_snr_only']:.1f}")
    g = df.groupby("pair").agg(hit=("correct", "mean"), n=("correct", "size"), mn=("mn_dprime", "mean"), ex=("ac_excess", "mean"),
                               ext=("ac_excess_talker", "mean"), dp=("ac_dprime", "mean"), ft=("features", "mean"))
    for nmin in (1, 3):
        q = g[g.n >= nmin]
        out["listener"][f"pair_level_n>={nmin}"] = dict(n_pairs=int(len(q)), **{c: float(pearsonr(q[c], q.hit)[0]) for c in ("mn", "ft", "ex", "ext", "dp")})
        print(f"   逐對（題數 >= {nmin} 的 {len(q)} 個無序對）r(命中, ·)：M&N {pearsonr(q.mn, q.hit)[0]:+.2f}  特徵數 {pearsonr(q.ft, q.hit)[0]:+.2f}  "
              f"頻譜圖差別 {pearsonr(q.ex, q.hit)[0]:+.2f}  （該說話者）{pearsonr(q.ext, q.hit)[0]:+.2f}  樣板 d′ {pearsonr(q.dp, q.hit)[0]:+.2f}")
    df["bin"] = pd.qcut(df.ac_excess_talker, 4, labels=False)
    qt = df.groupby("bin").agg(lo=("ac_excess_talker", "min"), hi=("ac_excess_talker", "max"), hit=("correct", "mean"), k=("correct", "sum"), n=("correct", "size"))
    out["listener"]["quartiles"] = qt.reset_index().to_dict("records")
    print("   頻譜圖差別（該說話者）四等分的命中率："); print(qt.round(2).to_string())
    # 只差清濁的對子測到幾題：這是現有資料的洞，企劃書 §5 要補
    voi = {"p": "b", "t": "d", "k": "g", "f": "v", "theta": "eth", "s": "z", "sh": "zh"}
    vp = {pair_key(k_, v_) for k_, v_ in voi.items()}
    vv = df[df.pair.isin(vp)]
    out["listener"]["voicing_only"] = dict(n=int(len(vv)), hits=int(vv.correct.sum()), pairs=sorted(vv.pair.unique()))
    print(f"   只差清濁的對子：{len(vv)} 題，命中 {int(vv.correct.sum())}（{', '.join(sorted(vv.pair.unique())) or '無'}）")

    # ---- 3. 母語聽者 EN01，乾淨條件的校準試
    rows = []
    for f in sorted(glob.glob(C.path(a.native))):
        c = pd.read_csv(f)
        c = c[(c.dim == "audio") & (c.is_match == 0) & c.key.notna()]
        for _, r in c.iterrows():
            cons = os.path.basename(r.probe_snd).split("_")[1][:-5]
            hit = T[((T.target == cons) | (T.foil == cons)) & ((T.dprime - r.x).abs() < 2e-3)]
            if hit.pair.nunique() == 1:
                rows.append(dict(pair=hit.pair.iloc[0], correct=r.correct, mn=r.x, session=os.path.basename(f)[:9]))
        out.setdefault("native", {}).setdefault("n_audio_different_trials", 0)
        out["native"]["n_audio_different_trials"] += int(len(c))
    E = pd.DataFrame(rows)
    q = P[P.snr == "clean"].set_index("pair")
    E["excess"], E["dprime"] = q.excess.reindex(E.pair).values, q.dprime.reindex(E.pair).values
    out["native"].update(n=int(len(E)), misses=int((1 - E.correct).sum()))
    print(f"\n3. 母語聽者（乾淨）：「不同」試 {out['native']['n_audio_different_trials']} 題，反推出唯一一對的 {len(E)} 題，漏 {int((1 - E.correct).sum())}")
    for m in ("mn", "excess", "dprime"):
        au, ci = auc(E[m], E.correct), boot_auc(E[m], E.correct)
        out["native"][m] = dict(auc=au, ci=ci)
        print(f"   {m:8s} AUC {au:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    print("   漏掉的對：", E[E.correct == 0][["pair", "mn", "excess"]].round(2).to_dict("records"))
    json.dump(out, open(os.path.join(out_d, "acoustic_scale_validation.json"), "w"), indent=1, default=float, ensure_ascii=False)
    print("\n寫出：acoustic_scale_validation.json、acoustic_scale_trials_%s.csv" % a.participant)


if __name__ == "__main__":
    main()
