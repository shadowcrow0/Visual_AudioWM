"""主實驗（SFT，same/different）：各條件正確率與 RT、AA 誤報、存活函數序（選擇性影響）、SIC。

讀：data/<id>_simple_main.csv（SFT_WM_nochange_EN.py 寫的，每個探測一列；block 0 = 練習，不算）
算：(1) 九個條件（AA、單變 HA LA AH AL、雙變 HH HL LH LL）的試次數、正確率、RT 中位數（正確試 / 所有有反應的試次各一欄）；
    (2) AA 的誤報率，逐 block；
    (3) 存活函數序：每個通道把雙變格合併成 H 與 L 兩組（聲音 = cond_name 第一個字母，顏色 = 第二個），
        用單尾 KS 兩個方向各檢一次；單變條件（HA vs LA、AH vs AL）同樣檢一次；
    (4) Houpt 的八個 dominance 檢定（sft::siDominance 的移植）與 KS-SIC 檢定（D+、D-）。
出：analyze/out/main_task_<檔名>_conditions.csv、_tests.csv、_survivor_sound.png、_survivor_colour.png。
對應：review/pilot_2026-11-24_EN01_s002.md §3、DECISIONS 24 的「為什麼 (1)」；review/figures/fig16、fig22、fig23。

只取正確且 RT 在 0.15-3.0 s 的試次做 RT 分析（SFT 的慣例：存活函數是「答對的 RT」的）。

⚠ KS 的方向（這支程式照數學定義標，不照 sft 套件的列名）：
      "H 較快"  D = max(F_H − F_L)   —— 選擇性影響預期的方向；H 的 CDF 在上面 ⇔ S_H(t) < S_L(t)
      "L 較快"  D = max(F_L − F_H)   —— 違反的方向
  兩個 p 都是 exp(−2·n·D²)（R ks.test(exact=FALSE)）。選擇性影響成立 = 「H 較快」顯著、「L 較快」不顯著。
  dominance 表的列名 "S.hh > S.hl" 是 sft 套件自己的寫法，它對應 ks.test(HH, HL, "greater") = F_HH 在上 = HH 較快；
  所以前四列顯著、後四列不顯著才是通過（adaptivesft.sic.classify 就是這樣判的）。
  2026-11-18 / 11-24 的筆記把 "less" 那個 p 標成「預期方向」，標反了；見 analyze/README.md「與筆記對不上的地方」。
"""
import argparse
import os

import numpy as np
import pandas as pd

import _common as C

CONDS = ("AA", "HA", "LA", "AH", "AL", "HH", "HL", "LH", "LL")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("main_csv", nargs="?", default="data/EN01_s002_2026-11-24_08h49.32_simple_main.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    sic = C.load_sic()
    tag = os.path.basename(a.main_csv).replace("_simple_main.csv", "")
    allrows, m = C.load_main(a.main_csv)
    ok = m[(m.correct == 1) & m.rt.between(*C.RT_WINDOW)]
    print(f"{tag}：正式 block {len(allrows)} 列，有反應 {len(m)}，正確且 RT 在 {C.RT_WINDOW} 內 {len(ok)}")

    tab = pd.DataFrame({"n": m.groupby("cond_name").size(), "accuracy": m.groupby("cond_name").correct.mean(),
                        "rt_median_correct": ok.groupby("cond_name").rt.median(), "n_correct_rt": ok.groupby("cond_name").size(),
                        "rt_median_all": m.groupby("cond_name").rt.median()}).reindex(CONDS)      # 筆記裡的 RT 中位數是這一欄（所有有反應的試次）
    print("\n(1) 各條件"); print(tab.round(3).T.to_string())
    aa = m[m.cond_name == "AA"]
    print(f"\n(2) AA 誤報率 {1 - aa.correct.mean():.2f}；逐 block：" + " ".join(f"{b}:{1 - g.correct.mean():.2f}" for b, g in aa.groupby("block")))
    tab.to_csv(os.path.join(out, f"main_task_{tag}_conditions.csv"))

    cell = {c: ok[ok.cond_name == c].rt.values for c in CONDS}
    groups = {"sound": (np.concatenate([cell["HH"], cell["HL"]]), np.concatenate([cell["LH"], cell["LL"]]), cell["HA"], cell["LA"], "HA", "LA"),
              "colour": (np.concatenate([cell["HH"], cell["LH"]]), np.concatenate([cell["HL"], cell["LL"]]), cell["AH"], cell["AL"], "AH", "AL")}
    tests = []
    print("\n(3) 存活函數序（正確 RT）。預期：「H 較快」顯著、「L 較快」不顯著")
    for ch, (H, L, h1, l1, nh, nl) in groups.items():
        for lab, x, y in ((f"{ch}：雙變格合併", H, L), (f"{ch}：單變 {nh} vs {nl}", h1, l1)):
            Dg, pg = sic._ks_one_sided(x, y, "greater"); Dl, pl = sic._ks_one_sided(x, y, "less")
            tests.append(dict(test=lab, n_H=len(x), n_L=len(y), median_H=np.median(x), median_L=np.median(y), D_H_faster=Dg, p_H_faster=pg, D_L_faster=Dl, p_L_faster=pl))
            print(f"   {lab:22s} n {len(x):3d}/{len(y):3d}  中位數 H {np.median(x):.3f} L {np.median(y):.3f}  |  H 較快 D={Dg:.3f} p={pg:.3f}  |  L 較快 D={Dl:.3f} p={pl:.3f}")
    acc = m[m.cond_name.isin(["HH", "HL", "LH", "LL"])]
    print("   雙變格正確率：聲音 H {:.3f} / L {:.3f}；顏色 H {:.3f} / L {:.3f}".format(
        acc[acc.cond_name.str[0] == "H"].correct.mean(), acc[acc.cond_name.str[0] == "L"].correct.mean(),
        acc[acc.cond_name.str[1] == "H"].correct.mean(), acc[acc.cond_name.str[1] == "L"].correct.mean()))

    print("\n(4) Houpt dominance（列名是 sft 的寫法；前四列 = 較快那一格的 CDF 在上，通過時應顯著；後四列應不顯著）")
    for name, D, p in sic.si_dominance(cell["HH"], cell["HL"], cell["LH"], cell["LL"]):
        tests.append(dict(test="dominance " + name, D_H_faster=D, p_H_faster=p)); print(f"   {name:14s} D={D:.3f}  p={p:.3f}")
    st = sic.sic_test(cell["HH"], cell["HL"], cell["LH"], cell["LL"])
    print(f"   KS-SIC：D+ {st['positive'][0]:.3f} p={st['positive'][1]:.3f}；D- {st['negative'][0]:.3f} p={st['negative'][1]:.3f}"
          "   （選擇性影響不成立時 SIC 不能解讀）")
    tests += [dict(test="SIC D+", D_H_faster=st["positive"][0], p_H_faster=st["positive"][1]), dict(test="SIC D-", D_H_faster=st["negative"][0], p_H_faster=st["negative"][1])]
    pd.DataFrame(tests).to_csv(os.path.join(out, f"main_task_{tag}_tests.csv"), index=False)

    plt = C.setup_plot()
    t = np.linspace(0.3, 2.0, 400); S = lambda v: np.array([np.mean(v > tt) for tt in t])
    for (ch, (H, L, h1, l1, nh, nl)), zh in zip(groups.items(), ("聲音", "顏色")):
        r = next(x for x in tests if x["test"] == f"{ch}：雙變格合併"); r1 = next(x for x in tests if x["test"].startswith(f"{ch}：單變"))
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(t, S(H), color=C.RED, lw=2, label=f"{zh} H（雙變格），n = {len(H)}，中位數 {np.median(H):.2f} s")
        ax.plot(t, S(L), color=C.BLUE, lw=2, label=f"{zh} L（雙變格），n = {len(L)}，中位數 {np.median(L):.2f} s")
        ax.plot(t, S(h1), color=C.RED, lw=1.2, ls="--", label=f"{nh}（只有{zh}變，H），n = {len(h1)}")
        ax.plot(t, S(l1), color=C.BLUE, lw=1.2, ls="--", label=f"{nl}（只有{zh}變，L），n = {len(l1)}")
        ax.text(0.98, 0.62, f"單尾 KS，雙變格\n  H 較快（預期）：p = {r['p_H_faster']:.2f}\n  L 較快（違反）：p = {r['p_L_faster']:.2f}\n"
                            f"單變 {nh} vs {nl}\n  H 較快：p = {r1['p_H_faster']:.2f}\n  L 較快：p = {r1['p_L_faster']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
        ax.set_xlabel("t (s)"); ax.set_ylabel("S(t) = P(正確 RT > t)"); ax.legend(fontsize=8, loc="lower left")
        ax.set_title(f"{tag}：{zh}通道 H 與 L 的存活函數（正確 RT）", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"main_task_{tag}_survivor_{ch}.png")); plt.close(fig)
    print(f"\n寫出：{out}/main_task_{tag}_conditions.csv、_tests.csv、_survivor_sound.png、_survivor_colour.png")


if __name__ == "__main__":
    main()
