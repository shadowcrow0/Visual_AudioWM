"""SNR pilot：以子音對為單位，看 d' 尺對這位聽者有沒有排序力。

讀：data/<id>_sound_perception_dprime_snr<dB>.csv（預設三個 SNR 合併）
算：(1) 每個有序對（目標–foil）的命中率、試次數、d'、特徵數；(2) 命中率對 d' 的 Pearson r（以對為單位，不加權）；
    (3) 依 Miller-Nicely 特徵數（差 1 / 2 / 3 / 4 個）的命中率；(4) 依說話者的命中率；(5) 命中最低的對子。
出：analyze/out/snr_pilot_pairs_<id>.csv、snr_pilot_pairs_<id>.png。
對應：review/pilot_2026-11-18_snr_pilot.md 讀法 1、2、4；review/figures/fig21。

為什麼以「對」為單位算 r：尺的主張是「d' 大的對比較好分」。如果尺有效，不管噪音多少，對與對之間的命中率
應該跟著 d' 走；r 接近 0 表示掉的是特定的對，不是特定的 d'。
無反應的試次不算進命中率的分母（said_diff 是 NaN）。
"""
import argparse
import os

import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 6, 12])
    ap.add_argument("--min-n", type=int, default=3, help="圖上標名字與「最低的對」只看試次數 >= 這個值的對")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    d = C.load_perception(a.participant, a.snrs)
    diff = d[d.is_same == 0]
    pp = diff.groupby(["consonant", "foil"]).agg(hit=("said_diff", "mean"), k=("said_diff", "sum"), n=("said_diff", "count"),
                                                 dprime=("x", "mean"), features=("feature_distance", "first")).reset_index()
    pp = pp[pp.n > 0]
    r = np.corrcoef(pp.dprime, pp.hit)[0, 1]
    print(f"{a.participant}，SNR {a.snrs}：{len(pp)} 個有序對，{int(pp.n.sum())} 個有反應的「不同」試次")
    print(f"命中率對 d' 的相關（以對為單位）：r = {r:.2f}")
    print("\n依特徵數：")
    print(diff.groupby("feature_distance").said_diff.agg(hit="mean", k="sum", n="count").round(2).to_string())
    print("\n依計畫層級：")
    print(diff.groupby("x_planned").said_diff.agg(hit="mean", k="sum", n="count").round(2).to_string())
    print("\n依說話者（命中率，「不同」試次）：")
    print(diff.groupby("talker").said_diff.agg(hit="mean", n="count").round(2).sort_values("hit").T.to_string())
    print(f"\n命中最低的對（n >= {a.min_n}）：")
    low = pp[pp.n >= a.min_n].sort_values(["hit", "n"]).head(12)
    for x in low.itertuples():
        print(f"   {C.ipa(x.consonant)}-{C.ipa(x.foil):3s} {int(x.k)}/{x.n}  d' {x.dprime:.2f}  特徵 {int(x.features)}")
    pp.sort_values("hit").to_csv(os.path.join(out, f"snr_pilot_pairs_{a.participant}.csv"), index=False)

    plt = C.setup_plot()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sc = ax.scatter(pp.dprime, pp.hit, s=18 * pp.n, c=pp.features, cmap="viridis", vmin=1, vmax=4, alpha=.8, edgecolor="k", linewidth=.4)
    for x in pp[(pp.hit <= 0.34) & (pp.n >= a.min_n)].itertuples():
        ax.annotate(f"{C.ipa(x.consonant)}-{C.ipa(x.foil)}", (x.dprime, x.hit), textcoords="offset points", xytext=(6, -3), fontsize=8)
    fig.colorbar(sc, ax=ax).set_label("相差的 Miller-Nicely 特徵數")
    ax.set_xlabel("這一對在乾淨尺上的 d'"); ax.set_ylabel("命中率（各 SNR 合併）"); ax.set_ylim(-0.05, 1.08); ax.set_xlim(0, 5)
    ax.set_title(f"{a.participant}：每一對的命中率（{len(pp)} 對，面積 = 試次數），r(命中, d') = {r:.2f}", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"snr_pilot_pairs_{a.participant}.png")); plt.close(fig)
    print(f"\n寫出：{out}/snr_pilot_pairs_{a.participant}.csv、.png")


if __name__ == "__main__":
    main()
