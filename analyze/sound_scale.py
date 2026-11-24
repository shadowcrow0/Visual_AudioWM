"""聲音的 d' 尺長什麼樣、受試者的校準落在尺的哪裡、固定 SNR 會把整把尺推到哪裡。

讀：stimuli/consonant_dprime.csv（240 個有序對的 d'；make_sound_pairs.py 產生）、
    data/<id>_simple_calib_trials.csv（一個或多個；取聲音維度）、
    Miller & Nicely (1955) 的逐 SNR 混淆矩陣（經 consonant_distance.luce_distance 讀）
算：(1) d' 表的分布：有幾對低於 --floor（預設 0.8）；
    (2) 每位受試者聲音校準各級答「不一樣」的比例（疊在尺上）；
    (3) 用 M&N 各 SNR 的矩陣各自重算 Luce 距離 × 0.911，看整把尺的中位數與低於 floor 的比例怎麼變。
出：analyze/out/sound_scale_pairs.csv、sound_scale_by_snr.csv、sound_scale_vs_calibration.png、sound_scale_by_snr.png。
對應：DECISIONS 24「為什麼 (1)(2)」（240 對只有 14 對 < 0.8；噪音把整把尺左移）；review/paper 的 fig17、fig18。

為什麼要看這個：EN01 與 yyc1110 的聲音校準六級幾乎全對，問題不是層級放錯，而是尺的左邊沒有東西 ——
可用的對子裡 d' 很小的太少，斜面在尺外。
(3) 的限制：0.911 這個係數只在 -6 dB 估過（review/GRT擬合MN_B路線_D2.md §8），拿去乘別的 SNR 是近似；
而且實驗用的是語音形噪音，M&N 是白噪音，同一個 dB 數遮蔽量不同。所以 (3) 只說明方向，不能拿來選 SNR。
"""
import argparse
import os

import numpy as np
import pandas as pd

import _common as C
import adaptive_vawm_EN as A
import consonant_distance as cd
from make_sound_pairs import LUCE_TO_DPRIME as SCALE          # Luce 距離 → d' 的係數（0.911）；唯一定義在 make_sound_pairs.py


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calib_csv", nargs="*", default=["data/EN01_s001_2026-11-18_10h16.05_simple_calib_trials.csv", "data/yyc1110_simple_calib_trials.csv"])
    ap.add_argument("--floor", type=float, default=0.8)
    ap.add_argument("--snrs", type=int, nargs="+", default=[0, -6, -12], help="M&N 有的 SNR：12 6 0 -6 -12 -18")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    table = A.load_dprime()
    dp = pd.DataFrame([(t, f, d) for t, fl in table.items() for f, d in fl], columns=["target", "foil", "dprime"])
    n_low = int((dp.dprime < a.floor).sum())
    print(f"(1) d' 表：{len(dp)} 個有序對，範圍 {dp.dprime.min():.2f}-{dp.dprime.max():.2f}，中位數 {dp.dprime.median():.2f}；"
          f"d' < {a.floor} 的有 {n_low} 對（{n_low / len(dp):.0%}）")
    print("    最小的幾對：" + "、".join(f"{C.ipa(r.target)}-{C.ipa(r.foil)} {r.dprime:.2f}" for r in dp.sort_values("dprime").head(8).itertuples()))
    dp.sort_values("dprime").to_csv(os.path.join(out, "sound_scale_pairs.csv"), index=False)

    print("\n(2) 聲音校準各級 P(不一樣)")
    calib = {}
    for f in a.calib_csv:
        tag = os.path.basename(f).replace("_simple_calib_trials.csv", "")
        _, diff, same = C.load_calib_trials(f)
        s = diff[diff.dim == "audio"]
        calib[tag] = s.groupby("x_planned").agg(n=("correct", "size"), k=("correct", "sum"), x=("x", "mean")).reset_index()
        print(f"   {tag}：" + "  ".join(f"{r.x_planned:g}→{int(r.k)}/{int(r.n)}" for r in calib[tag].itertuples())
              + f"   誤報 {(same[same.dim == 'audio'].key == 'n').mean():.2f}")

    print("\n(3) 用 M&N 各 SNR 的矩陣重算（Luce 距離 × 0.911）")
    by_snr = {}
    for snr in a.snrs:
        M = cd.luce_distance(snrs=(snr,)); by_snr[snr] = SCALE * M[~np.eye(len(M), dtype=bool)]
        print(f"   {snr:+d} dB：中位數 {np.median(by_snr[snr]):.2f}，< {a.floor} 的比例 {(by_snr[snr] < a.floor).mean():.0%}")
    pd.DataFrame([dict(snr=k, median=np.median(v), below_floor=(v < a.floor).mean()) for k, v in by_snr.items()]).to_csv(os.path.join(out, "sound_scale_by_snr.csv"), index=False)

    plt = C.setup_plot()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(dp.dprime, bins=np.arange(0, 7.5, 0.25), color="#bdc3c7", label=f"d' 表的 {len(dp)} 個有序對")
    ax.axvline(a.floor, color="k", ls="--", lw=1); ax.text(a.floor + 0.05, ax.get_ylim()[1] * 0.9, f"d' < {a.floor}：{n_low} 對", fontsize=8)
    for x in A.LEVELS_AUDIO:
        ax.plot(x, -0.6, "^", color="k", ms=7, clip_on=False)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    for (tag, lv), col in zip(calib.items(), (C.RED, C.BLUE, C.GREEN, C.ORANGE)):
        p = lv.k / lv.n
        ax2.errorbar(lv.x, p, yerr=np.sqrt(p * (1 - p) / lv.n + 1e-9), fmt="o", color=col, ms=6, capsize=3, lw=1, label=f"{tag}：校準各級 P(不一樣)")      # 不連線：同一級實際呈現的 d' 平均不一定照層級順序
    ax2.set_ylim(-0.05, 1.25); ax2.set_ylabel("P(回答「不一樣」)")
    ax.set_xlim(0, 7.2); ax.set_xlabel("這一對的 d'（三角形 = LEVELS_AUDIO）"); ax.set_ylabel("對數")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    ax.set_title("聲音的尺：0.8 以下幾乎沒有對子，受試者從最低級就在天花板", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sound_scale_vs_calibration.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (snr, v), col in zip(by_snr.items(), (C.GREY, C.BLUE, C.RED, C.GREEN, C.ORANGE, "k")):
        ax.hist(v, bins=np.arange(0, 8.5, 0.25), histtype="step", lw=2, color=col, label=f"M&N {snr:+d} dB 的矩陣：中位數 {np.median(v):.1f}，{(v < a.floor).mean():.0%} 的對低於 {a.floor}")
    ax.axvline(a.floor, color="k", ls="--", lw=1); ax.set_xlabel("0.911 × Luce 距離（用該 SNR 的 M&N 矩陣算）"); ax.set_ylabel("對數（共 240）")
    ax.set_title("SNR 越低，整把尺越往左（0.911 只在 -6 dB 估過，其他 SNR 是近似）", loc="left", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "sound_scale_by_snr.png")); plt.close(fig)
    print(f"\n寫出：{out}/sound_scale_pairs.csv、sound_scale_by_snr.csv、sound_scale_vs_calibration.png、sound_scale_by_snr.png")


if __name__ == "__main__":
    main()
