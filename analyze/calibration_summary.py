"""校準階段：每一級答「不一樣」的比例、誤報率，以及實驗當場算出來的 H / L（不重新擬合，不需要 PyMC）。

讀：data/<id>_simple_calib_trials.csv（每試一列）與同名的 _simple_calib.json（實驗當場的擬合結果，若存在）
算：每個維度（colour / audio）
      各計畫層級的試次數、答「不一樣」的比例（= 不同試的正確率）、RT 中位數、實際呈現的 x 平均；
      誤報率（相同試按了 n）；json 裡記的 method、H、L、in_range、warnings。
    只取 other == 'T' 且 RT 在 0.15-3.0 s 的不同試（擬合用的就是這些；見 _common.load_calib_trials）。
出：analyze/out/calibration_levels.csv（每檔 × 維度 × 層級一列）、calibration_levels_colour.png、calibration_levels_audio.png。
對應：review/pilot_2026-11-24_EN01_s002.md §2 的「舊版算出 / 資料」兩欄；DECISIONS 23、25 引用的各級比例；review/paper 的 fig13。

為什麼要有不擬合的這一支：先看原始比例，才知道擬合結果合不合理（EN01 s001 的 H = L = 2 就是原始比例一看就不對）。
"""
import argparse
import glob
import json
import os

import pandas as pd

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calib_csv", nargs="*", help="預設：data/ 下所有 *_simple_calib_trials.csv（機器人的除外）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    files = a.calib_csv or sorted(f for f in glob.glob(os.path.join(C.DATA, "*_simple_calib_trials.csv")) if not os.path.basename(f).startswith("robot"))
    rows = []
    for f in files:
        tag = os.path.basename(f).replace("_simple_calib_trials.csv", "")
        js = C.path(f).replace("_simple_calib_trials.csv", "_simple_calib.json")
        rec = json.load(open(js, encoding="utf-8")) if os.path.isfile(js) else {}
        print(f"\n=== {tag}" + (f"   實驗當場：method = {rec.get('info', {}).get('method')}" if rec else "   （沒有 json）"))
        for dim in ("colour", "audio"):
            _, diff, same = C.load_calib_trials(f)
            diff, same = diff[diff.dim == dim], same[same.dim == dim]
            fa = (same.key == "n").mean()
            lv = diff.groupby("x_planned").agg(n=("correct", "size"), k=("correct", "sum"), p_different=("correct", "mean"), x=("x", "mean"), rt_median=("rt", "median")).reset_index()
            r = rec.get(dim, {})
            print(f"[{dim}] 擬合用 {len(diff)} 試；誤報 {fa:.2f}（{int((same.key == 'n').sum())}/{len(same)}）"
                  + (f"；當場 L = {r['low']:.2f}、H = {r['high']:.2f}、in_range = {r['in_range']}、method = {r.get('extra', {}).get('method', '?')}" if r else ""))
            print("   層級      " + "  ".join(f"{v:5g}" for v in lv.x_planned))
            print("   P(不一樣) " + "  ".join(f"{v:5.2f}" for v in lv.p_different))
            print("   k/n       " + "  ".join(f"{int(k):2d}/{int(n):<2d}" for k, n in zip(lv.k, lv.n)))
            for w in r.get("warnings", []):
                print("   當場警告：", w)
            rows.append(lv.assign(file=tag, dim=dim, false_alarm=fa, recorded_low=r.get("low"), recorded_high=r.get("high")))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(os.path.join(out, "calibration_levels.csv"), index=False)

    plt = C.setup_plot()
    cols = [C.BLUE, C.RED, C.GREEN, C.ORANGE, C.GREY, "k"]
    for dim, xlabel in (("colour", "顏色差 ΔE"), ("audio", "子音對的 d'")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for (tag, g), c in zip(T[T.dim == dim].groupby("file", sort=False), cols * 3):
            if dim == "audio" and g.x_planned.min() < 0:          # 2026-11-03 以前的聲音軸是 -log10(混淆次數 + 1)，不是 d'（DECISIONS 21）；不能畫在同一條軸上
                print(f"   （{tag} 的聲音層級是舊軸，不畫進 calibration_levels_audio.png）"); continue
            ax.plot(g.x, g.p_different, "o-", color=c, ms=6, label=f"{tag}（誤報 {g.false_alarm.iloc[0]:.2f}）")
        ax.axhline(0.5, color="k", ls=":", lw=.8); ax.set_ylim(-0.05, 1.05); ax.set_xlabel(xlabel + "（實際呈現值的層級平均）"); ax.set_ylabel("P(回答「不一樣」)")
        ax.set_title(f"校準：{'顏色' if dim == 'colour' else '聲音'}各級答「不一樣」的比例（每級約 5 試）", loc="left", fontsize=10); ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"calibration_levels_{dim}.png")); plt.close(fig)
    print(f"\n寫出：{out}/calibration_levels.csv、calibration_levels_colour.png、calibration_levels_audio.png")


if __name__ == "__main__":
    main()
