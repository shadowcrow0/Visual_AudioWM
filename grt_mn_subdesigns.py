"""B 路線 D1：把 Miller & Nicely (1955) 的 16×16 混淆矩陣切成「兩個特徵的全因子子設計」，
給 mdsdt::fit.grt（R）用，並在 Python 先算一份簡單 SDT 摘要。

企劃：review/企劃_GRT擬合MillerNicely.md §2、§3 A4、§4b。

四條軸（A4）：affrication 與 duration 併成三階 manner
    voicing   0 無聲 / 1 有聲
    nasality  0 口音 / 1 鼻音
    manner    0 塞音與鼻音 (affrication 0) / 1 短擦音 f θ v ð (affrication 1, duration 0) / 2 噝音 s ʃ z ʒ (duration 1)
    place     0 前 / 1 中 / 2 後

子設計 = 固定另外兩條軸的值後，軸 f、g 的水準子集 L_f × L_g，要求**每個組合都有一個子音**（全因子），
並且取極大者（不能再加一個水準）。每條軸至少兩個水準。

輸出（review/data/grt_subdesigns/）：
    <design>_<snr>.csv       長格式：stim_f, stim_g, resp_f, resp_g, count, stim_cons, resp_cons, stim
                             → R: xtabs(count ~ resp_f + resp_g + stim, ...)
    index.csv                每個 (design, snr) 一列：軸、固定脈絡、子音、大小、檔名、保留比例
    python_sdt_summary.txt   逐軸的邊際 d′（整體、分另一軸水準）+ 邊際反應不變性 χ² 檢定
    python_sdt_summary.csv   同上，機器可讀

用法：python grt_mn_subdesigns.py
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import chi2_contingency, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adaptive_vawm_EN import MN_FEATURES  # noqa: E402  (voicing, nasality, affrication, duration, place)

TABLES_JSON = os.path.join(HERE, "review", "data", "miller_nicely_1955_tables_I-VI.json")
OUT_DIR = os.path.join(HERE, "review", "data", "grt_subdesigns")

CONSONANTS = ["p", "t", "k", "f", "theta", "s", "sh", "b", "d", "g", "v", "eth", "z", "zh", "m", "n"]
IPA = {"p": "p", "t": "t", "k": "k", "f": "f", "theta": "θ", "s": "s", "sh": "ʃ", "b": "b", "d": "d", "g": "g",
       "v": "v", "eth": "ð", "z": "z", "zh": "ʒ", "m": "m", "n": "n"}

AXES = ("voicing", "nasality", "manner", "place")
LEVEL_NAMES = {"voicing": {0: "voiceless", 1: "voiced"},
               "nasality": {0: "oral", 1: "nasal"},
               "manner": {0: "stop/nasal", 1: "short-fric", 2: "sibilant"},
               "place": {0: "front", 1: "middle", 2: "back"}}
MAIN_SNRS = ("-12", "-6", "0", "6")


# ------------------------------------------------------------------------------------------------ 特徵

def a4_features():
    """{consonant: (voicing, nasality, manner, place)}，由 MN_FEATURES 的五特徵折出 manner。"""
    out = {}
    for c, (v, n, aff, dur, pl) in MN_FEATURES.items():
        manner = 0 if aff == 0 else (1 if dur == 0 else 2)
        out[c] = (v, n, manner, pl)
    return out


FEATURES = a4_features()
AXIS_INDEX = {a: i for i, a in enumerate(AXES)}


# ------------------------------------------------------------------------------------------------ 資料

def load_tables(path=TABLES_JSON):
    """{snr_str: 16x16 int array}，列 = 說的、欄 = 聽到的，CONSONANTS 順序。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert list(raw["order"]) == CONSONANTS
    out = {}
    for key, table in raw["tables"].items():
        m = np.zeros((16, 16), dtype=int)
        for i, a in enumerate(CONSONANTS):
            for j, b in enumerate(CONSONANTS):
                m[i, j] = int(table[a].get(b, 0))
        out[key] = m
    return out


# ------------------------------------------------------------------------------------------------ 子設計枚舉

class SubDesign:
    """軸 f × 軸 g 的全因子子設計；fixed = {其他軸: 值}；grid[(lf, lg)] = 子音。"""

    def __init__(self, axis_f, axis_g, fixed, levels_f, levels_g, grid):
        self.axis_f, self.axis_g = axis_f, axis_g
        self.fixed = dict(fixed)
        self.levels_f, self.levels_g = tuple(levels_f), tuple(levels_g)
        self.grid = dict(grid)
        # 子音順序：f 跑最快（f0g0, f1g0, ..., f0g1, ...）— R 腳本假設同一順序
        self.consonants = [grid[(lf, lg)] for lg in self.levels_g for lf in self.levels_f]
        fixed_tag = "_".join(f"{a}{self.fixed[a]}" for a in AXES if a in self.fixed)
        # 同一對軸、同一固定脈絡可能有兩個極大子設計（manner × place：{p f t θ} 與 {t s k ʃ}），所以把水準也寫進名字
        level_tag = f"{axis_f}{''.join(map(str, self.levels_f))}_{axis_g}{''.join(map(str, self.levels_g))}"
        self.name = f"{axis_f}_x_{axis_g}__{fixed_tag}__{level_tag}"

    @property
    def size(self):
        return f"{len(self.levels_f)}x{len(self.levels_g)}"

    @property
    def context(self):
        return ", ".join(f"{a}={LEVEL_NAMES[a][self.fixed[a]]}" for a in AXES if a in self.fixed)

    def level_of(self, cons):
        """子音在 (f, g) 兩軸上的水準。"""
        feat = FEATURES[cons]
        return feat[AXIS_INDEX[self.axis_f]], feat[AXIS_INDEX[self.axis_g]]

    def __repr__(self):
        return f"SubDesign({self.name}, {self.size}, {self.consonants})"


def _is_full(cells, lf_set, lg_set):
    return all((lf, lg) in cells for lf in lf_set for lg in lg_set)


def enumerate_subdesigns(features=FEATURES, axes=AXES, min_levels=2):
    """所有極大全因子子設計：每對軸 (f, g)、每組其他軸的固定值。"""
    designs = []
    for axis_f, axis_g in itertools.combinations(axes, 2):
        others = [a for a in axes if a not in (axis_f, axis_g)]
        other_values = [sorted({features[c][AXIS_INDEX[a]] for c in features}) for a in others]
        for fixed_vals in itertools.product(*other_values):
            fixed = dict(zip(others, fixed_vals))
            cells = {}
            for c, feat in features.items():
                if all(feat[AXIS_INDEX[a]] == v for a, v in fixed.items()):
                    key = (feat[AXIS_INDEX[axis_f]], feat[AXIS_INDEX[axis_g]])
                    assert key not in cells, "two consonants in one cell — axes do not identify the 16 consonants"
                    cells[key] = c
            lf_all = sorted({k[0] for k in cells})
            lg_all = sorted({k[1] for k in cells})
            full = []
            for nf in range(min_levels, len(lf_all) + 1):
                for lf_set in itertools.combinations(lf_all, nf):
                    for ng in range(min_levels, len(lg_all) + 1):
                        for lg_set in itertools.combinations(lg_all, ng):
                            if _is_full(cells, lf_set, lg_set):
                                full.append((set(lf_set), set(lg_set)))
            for lf_set, lg_set in full:
                maximal = not any((lf_set <= a and lg_set <= b) and (lf_set, lg_set) != (a, b) for a, b in full)
                if maximal:
                    grid = {(lf, lg): cells[(lf, lg)] for lf in lf_set for lg in lg_set}
                    designs.append(SubDesign(axis_f, axis_g, fixed, sorted(lf_set), sorted(lg_set), grid))
    return designs


# ------------------------------------------------------------------------------------------------ 子矩陣與長格式

def submatrix(table, design):
    """(子矩陣 counts [n×n, 刺激 × 反應，design.consonants 順序], 每個刺激保留比例)。"""
    idx = [CONSONANTS.index(c) for c in design.consonants]
    sub = table[np.ix_(idx, idx)]
    row_full = table[idx].sum(axis=1)
    retained = sub.sum(axis=1) / np.where(row_full == 0, 1, row_full)
    return sub, retained


def long_rows(sub, design):
    """長格式列：stim_f, stim_g, resp_f, resp_g, count, stim_cons, resp_cons, stim。"""
    rows = []
    for i, sc in enumerate(design.consonants):
        sf, sg = design.level_of(sc)
        for j, rc in enumerate(design.consonants):
            rf, rg = design.level_of(rc)
            rows.append(dict(stim_f=sf, stim_g=sg, resp_f=rf, resp_g=rg, count=int(sub[i, j]),
                             stim_cons=sc, resp_cons=rc, stim=f"f{sf}g{sg}"))
    return rows


LONG_COLUMNS = ("stim_f", "stim_g", "resp_f", "resp_g", "count", "stim_cons", "resp_cons", "stim")


def write_long_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_long_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("stim_f", "stim_g", "resp_f", "resp_g", "count"):
            r[k] = int(r[k])
    return rows


def long_to_matrix(rows, design):
    """長格式 → 子矩陣（design.consonants 順序）。"""
    n = len(design.consonants)
    m = np.zeros((n, n), dtype=int)
    pos = {c: i for i, c in enumerate(design.consonants)}
    for r in rows:
        m[pos[r["stim_cons"]], pos[r["resp_cons"]]] += r["count"]
    return m


# ------------------------------------------------------------------------------------------------ 簡單 SDT

def collapse(sub, design, axis):
    """把子矩陣沿另一軸折疊：回傳 (table[刺激水準 × 反應水準], levels)。"""
    which = 0 if axis == design.axis_f else 1
    levels = design.levels_f if which == 0 else design.levels_g
    pos = {lv: i for i, lv in enumerate(levels)}
    t = np.zeros((len(levels), len(levels)), dtype=float)
    for i, sc in enumerate(design.consonants):
        for j, rc in enumerate(design.consonants):
            t[pos[design.level_of(sc)[which]], pos[design.level_of(rc)[which]]] += sub[i, j]
    return t, levels


def adjacent_dprimes(table):
    """序數水準表 [刺激 × 反應]（同一順序）→ 相鄰水準 d′ 列表。

    第 k 條判準切在水準 k 與 k+1 之間：
        H  = P(反應 > k | 刺激 = k+1),  FA = P(反應 > k | 刺激 = k)
        d′ = z(H) − z(FA)，log-linear 校正（每格 +0.5、列和 +1）。
    兩水準時就是一般的 2×2 d′。
    """
    n = table.shape[0]
    out = []
    for k in range(n - 1):
        above_hi = table[k + 1, k + 1:].sum()
        tot_hi = table[k + 1].sum()
        above_lo = table[k, k + 1:].sum()
        tot_lo = table[k].sum()
        h = (above_hi + 0.5) / (tot_hi + 1.0)
        fa = (above_lo + 0.5) / (tot_lo + 1.0)
        out.append(float(norm.ppf(h) - norm.ppf(fa)))
    return out


def marginal_dprimes(sub, design, axis):
    """軸 axis 的邊際 d′：'overall' 及另一軸每個水準各一組（相鄰水準 d′ 列表）。"""
    other = design.axis_g if axis == design.axis_f else design.axis_f
    which_other = 1 if axis == design.axis_f else 0
    other_levels = design.levels_g if which_other == 1 else design.levels_f
    result = {}
    t, _ = collapse(sub, design, axis)
    result["overall"] = adjacent_dprimes(t)
    for lv in other_levels:
        keep = [i for i, c in enumerate(design.consonants) if design.level_of(c)[which_other] == lv]
        masked = np.zeros_like(sub)
        masked[keep, :] = sub[keep, :]
        t, _ = collapse(masked, design, axis)
        result[f"{other}={lv}"] = adjacent_dprimes(t)
    return result


def _safe_chi2(table):
    """chi2_contingency，先丟掉全零的列／欄；退化（<2 列或 <2 欄）回 nan。"""
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        # 退化：某個反應水準從沒出現（例：0 dB 無聲刺激從未被聽成有聲）— 無變異可檢定，回 nan 而不是 0
        return float("nan"), 0, float("nan")
    chi2, p, dof, _ = chi2_contingency(t, correction=False)
    return float(chi2), int(dof), float(p)


def mri_tests(sub, design, axis):
    """邊際反應不變性：固定刺激在 axis 的水準，軸 axis 的反應分布是否依另一軸的刺激水準而變。

    回傳 {axis=lv: (chi2, df, p)} 與 'pooled'（χ²、df 相加）。
    """
    which = 0 if axis == design.axis_f else 1
    which_other = 1 - which
    levels = design.levels_f if which == 0 else design.levels_g
    other_levels = design.levels_g if which_other == 1 else design.levels_f
    pos = {lv: i for i, lv in enumerate(levels)}
    opos = {lv: i for i, lv in enumerate(other_levels)}
    out = {}
    tot_chi2, tot_df = 0.0, 0
    for lv in levels:
        t = np.zeros((len(other_levels), len(levels)))
        for i, sc in enumerate(design.consonants):
            s = design.level_of(sc)
            if s[which] != lv:
                continue
            for j, rc in enumerate(design.consonants):
                t[opos[s[which_other]], pos[design.level_of(rc)[which]]] += sub[i, j]
        chi2, df, p = _safe_chi2(t)
        out[f"{axis}={lv}"] = (chi2, df, p)
        if df:
            tot_chi2 += chi2
            tot_df += df
    from scipy.stats import chi2 as chi2_dist
    if tot_df:
        out["pooled"] = (tot_chi2, tot_df, float(chi2_dist.sf(tot_chi2, tot_df)))
    else:
        out["pooled"] = (float("nan"), 0, float("nan"))
    return out


def sdt_summary(sub, design):
    """一個 (design, snr) 的摘要：{axis: {'dprime': {...}, 'mri': {...}}}。"""
    return {axis: {"dprime": marginal_dprimes(sub, design, axis), "mri": mri_tests(sub, design, axis)}
            for axis in (design.axis_f, design.axis_g)}


# ------------------------------------------------------------------------------------------------ 主流程

def _fmt_d(ds):
    return "  ".join(f"{d:6.2f}" for d in ds)


def run(out_dir=OUT_DIR, snrs=MAIN_SNRS, verbose=True):
    tables = load_tables()
    designs = enumerate_subdesigns()
    os.makedirs(out_dir, exist_ok=True)
    index_rows, summary_rows, txt = [], [], []

    txt.append("Miller & Nicely (1955) 噪音混淆矩陣 → 全因子子設計 → 簡單 SDT 摘要（grt_mn_subdesigns.py）")
    txt.append("d′ = z(H) − z(FA)，log-linear 校正（每格 +0.5，列和 +1）；三水準軸給兩個相鄰水準 d′（判準 0|1、1|2）。")
    txt.append("「at」= 另一軸的刺激水準；overall = 折疊另一軸。分水準 d′ 不同 = 知覺可分性的初步跡象。")
    txt.append("MRI χ²：固定本軸刺激水準，本軸反應分布（折疊另一軸反應）對另一軸刺激水準的獨立性檢定（Pearson, 無連續校正）。")
    txt.append("所有數字都是條件化在子集內的反應上（retained = 落在子集內的反應比例）。")
    txt.append("")
    txt.append("=== 子設計清單 ===")
    for d in designs:
        txt.append(f"{d.name:42s} {d.size}  {d.axis_f} × {d.axis_g} | {d.context:40s} "
                   f"{{{' '.join(IPA[c] for c in d.consonants)}}}")
    txt.append("")

    for d in designs:
        txt.append("=" * 100)
        txt.append(f"{d.name}   {d.size}   {d.axis_f} × {d.axis_g} | {d.context}   "
                   f"{{{' '.join(IPA[c] for c in d.consonants)}}}   順序 = {' '.join(d.consonants)}")
        for snr in snrs:
            sub, retained = submatrix(tables[snr], d)
            fname = f"{d.name}_{snr}.csv"
            write_long_csv(os.path.join(out_dir, fname), long_rows(sub, d))
            idx = [CONSONANTS.index(c) for c in d.consonants]
            prop_all = sub.sum() / tables[snr][idx].sum()
            index_rows.append(dict(design=d.name, axis_f=d.axis_f, axis_g=d.axis_g,
                                   levels_f=" ".join(map(str, d.levels_f)), levels_g=" ".join(map(str, d.levels_g)),
                                   fixed=";".join(f"{a}={d.fixed[a]}" for a in AXES if a in d.fixed),
                                   context=d.context, consonants=" ".join(d.consonants), size=d.size,
                                   n_stim=len(d.consonants), snr=snr, file=fname,
                                   n_total=int(sub.sum()), prop_retained=round(float(prop_all), 4),
                                   prop_retained_min=round(float(retained.min()), 4)))
            txt.append("-" * 100)
            txt.append(f"SNR {snr:>3} dB   retained overall {prop_all:.3f}   per stimulus: "
                       + "  ".join(f"{IPA[c]} {r:.2f}" for c, r in zip(d.consonants, retained)))
            txt.append("counts (row = spoken, col = heard):")
            txt.append("        " + "".join(f"{IPA[c]:>6s}" for c in d.consonants))
            for i, c in enumerate(d.consonants):
                txt.append(f"  {IPA[c]:>4s}  " + "".join(f"{int(v):6d}" for v in sub[i]))
            summ = sdt_summary(sub, d)
            for axis in (d.axis_f, d.axis_g):
                levels = d.levels_f if axis == d.axis_f else d.levels_g
                crit = "  ".join(f"d'({levels[k]}|{levels[k+1]})" for k in range(len(levels) - 1))
                txt.append(f"  {axis} marginal d'    [{crit}]")
                for at, ds in summ[axis]["dprime"].items():
                    txt.append(f"      at {at:14s} {_fmt_d(ds)}")
                    for k, dv in enumerate(ds):
                        summary_rows.append(dict(design=d.name, snr=snr, axis=axis, at=at,
                                                 criterion=f"{levels[k]}|{levels[k+1]}", dprime=round(dv, 4)))
                for at, (chi2, df, p) in summ[axis]["mri"].items():
                    txt.append(f"  MRI {axis} resp ~ other stim | {at:12s} chi2={chi2:7.2f} df={df} p={p:.4f}")
                    summary_rows.append(dict(design=d.name, snr=snr, axis=axis, at=f"MRI {at}",
                                             criterion="", dprime="", chi2=round(chi2, 3), df=df, p=round(p, 5)))
        txt.append("")

    with open(os.path.join(out_dir, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    with open(os.path.join(out_dir, "python_sdt_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    cols = ["design", "snr", "axis", "at", "criterion", "dprime", "chi2", "df", "p"]
    with open(os.path.join(out_dir, "python_sdt_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    if verbose:
        print("\n".join(txt))
        print(f"wrote {len(index_rows)} csv files + index.csv + python_sdt_summary.{{txt,csv}} to {out_dir}")
    return designs, index_rows, summary_rows


if __name__ == "__main__":
    run()
