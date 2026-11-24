"""SNR pilot：華語有 / 沒有相近類別的子音，誤報與命中有沒有差。

讀：data/<id>_sound_perception_dprime_snr<dB>.csv
算：把 16 個子音分成「華語有相近的音」（p t k b d g f s ʃ m n）與「華語沒有」（θ ð v z ʒ），然後
    (1)「相同」試次（同一個音檔播兩次，只有噪音不同）的誤報率，分組 × SNR，合併後做 Fisher exact；
    (2)「不同」試次的命中率：含華語沒有的子音的對 vs 兩個都在華語裡的對；
    (3) 只差清濁的擦音對（f-v、s-z、θ-ð、ʃ-ʒ）的命中；
    (4) 逐子音的誤報、以及「弱 / 強 × 有 / 沒有類別」的拆法（用來分辨「沒有類別」與「音太弱被蓋掉」兩種解釋）。
出：analyze/out/native_category_false_alarms_<id>.csv、.png。
對應：2026-11-24 對話中回報的數字（誤報 18/72 對 17/35、Fisher p = .027；命中 86/109 對 77/101）。

程式字面算的只有上面這些比例與檢定。「沒有類別就只能比原始聲音」是解讀，不是這支程式能證明的：
分組是事後定的、只有一位聽者、p 值未校正；而且 θ ð v 同時也是最弱的三個音。
哪些子音算「華語沒有」是音韻學上的判斷（華語的塞音、擦音全是清音，沒有齒間音），寫死在 ABSENT 裡，不是從資料推的。
"""
import argparse
import os

import pandas as pd
from scipy.stats import fisher_exact

import _common as C

ABSENT = {"theta", "eth", "v", "z", "zh"}
VOICELESS_OF = {"v": "f", "z": "s", "eth": "theta", "zh": "sh"}             # 只差清濁的擦音對
WEAK = {"theta", "eth", "v", "f"}                                            # 摩擦噪音最弱的四個（consonant_acoustics.py：1.5-7 kHz 約 -32 到 -37 dB）


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="yyc1116")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 6, 12])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    d = C.load_perception(a.participant, a.snrs)
    d = d[d.said_diff.notna()]                                               # 無反應不算
    same, diff = d[d.is_same == 1].copy(), d[d.is_same == 0].copy()
    assert (same.snd1 == same.snd2).all(), "相同試應該是同一個音檔播兩次"

    same["absent"] = same.consonant.isin(ABSENT)
    g = same.groupby(["snr", "absent"]).said_diff.agg(fa="mean", k="sum", n="count")
    t = same.groupby("absent").said_diff.agg(fa="mean", k="sum", n="count")
    odds, p = fisher_exact([[t.loc[True, "k"], t.loc[True, "n"] - t.loc[True, "k"]], [t.loc[False, "k"], t.loc[False, "n"] - t.loc[False, "k"]]])
    print("(1)「相同」試次的誤報率（同一個音檔卻答不一樣）")
    print(g.round(2).to_string())
    print(f"   合併：華語有 {int(t.loc[False, 'k'])}/{int(t.loc[False, 'n'])} = {t.loc[False, 'fa']:.2f}；"
          f"華語沒有 {int(t.loc[True, 'k'])}/{int(t.loc[True, 'n'])} = {t.loc[True, 'fa']:.2f}；Fisher exact 勝算比 {odds:.2f}，p = {p:.3f}（雙尾，未校正）")

    diff["absent"] = diff.consonant.isin(ABSENT) | diff.foil.isin(ABSENT)
    h = diff.groupby("absent").said_diff.agg(hit="mean", k="sum", n="count")
    print("\n(2)「不同」試次的命中率")
    print(f"   含華語沒有的子音 {int(h.loc[True, 'k'])}/{int(h.loc[True, 'n'])} = {h.loc[True, 'hit']:.2f}；"
          f"兩個都在華語裡 {int(h.loc[False, 'k'])}/{int(h.loc[False, 'n'])} = {h.loc[False, 'hit']:.2f}")
    voi = diff[[VOICELESS_OF.get(c) == f or VOICELESS_OF.get(f) == c for c, f in zip(diff.consonant, diff.foil)]]
    print(f"\n(3) 只差清濁的擦音對：{int(voi.said_diff.sum())}/{len(voi)} 命中；出現的對："
          + "、".join(f"{C.ipa(c)}-{C.ipa(f)}" for c, f in zip(voi.consonant, voi.foil)))

    by_c = same.groupby("consonant").said_diff.agg(fa="mean", k="sum", n="count").sort_values("fa")
    by_c.index = [C.ipa(c) for c in by_c.index]
    print("\n(4) 逐子音的誤報"); print(by_c.round(2).T.to_string())
    lab = lambda c: ("沒有類別" if c in ABSENT else "有類別") + ("・弱" if c in WEAK else "・其他")
    print(same.groupby(same.consonant.map(lab)).said_diff.agg(fa="mean", k="sum", n="count").round(2).to_string())
    pd.concat([g.reset_index().assign(table="fa_by_snr"), t.reset_index().assign(table="fa_pooled"),
               h.reset_index().rename(columns={"hit": "fa"}).assign(table="hit_pooled")]).to_csv(
        os.path.join(out, f"native_category_false_alarms_{a.participant}.csv"), index=False)

    plt = C.setup_plot()
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    snrs = sorted(same.snr.unique())
    for i, s in enumerate(snrs):
        for k, ab in enumerate((False, True)):
            r = g.loc[(s, ab)]; xx = i + (k - .5) * .36
            ax.bar(xx, r.fa, .32, color="white" if ab else C.BLUE, ec=C.RED if ab else C.BLUE, lw=1.6, hatch="///" if ab else None)
            ax.text(xx, r.fa + .02, f"{r.fa:.2f}\n({int(r.k)}/{int(r.n)})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(snrs))); ax.set_xticklabels([f"SNR {s:+g} dB" for s in snrs]); ax.set_ylim(0, 1)
    ax.set_ylabel("誤報率（同一個音檔卻回答「不一樣」）")
    ax.legend(handles=[Patch(fc=C.BLUE, ec=C.BLUE, label="華語有相近的音 (p t k b d g f s ʃ m n)"),
                       Patch(fc="white", ec=C.RED, hatch="///", label="華語沒有的音 (θ ð v z ʒ)")], frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"聽者 {a.participant}：「相同」試次的誤報率", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"native_category_false_alarms_{a.participant}.png")); plt.close(fig)
    print(f"\n寫出：{out}/native_category_false_alarms_{a.participant}.csv、.png")


if __name__ == "__main__":
    main()
