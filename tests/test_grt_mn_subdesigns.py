"""grt_mn_subdesigns.py 的測試：枚舉只給全因子且極大的子設計；已知的子設計都在；子矩陣 = JSON；長格式 csv 來回一致。"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grt_mn_subdesigns import (AXES, AXIS_INDEX, CONSONANTS, FEATURES, TABLES_JSON, enumerate_subdesigns,  # noqa: E402
                               load_tables, long_rows, long_to_matrix, read_long_csv, submatrix, write_long_csv,
                               adjacent_dprimes, marginal_dprimes, run)


def _by_set(designs):
    return {frozenset(d.consonants): d for d in designs}


def test_a4_features_are_consistent():
    assert set(FEATURES) == set(CONSONANTS)
    assert FEATURES["s"][2] == 2 and FEATURES["f"][2] == 1 and FEATURES["p"][2] == 0 and FEATURES["m"] == (1, 1, 0, 0)
    assert len(set(FEATURES.values())) == 16                                   # 四軸能區分 16 個子音


def test_enumeration_is_full_factorial_and_maximal():
    designs = enumerate_subdesigns()
    assert len(designs) == 11
    for d in designs:
        assert len(d.levels_f) >= 2 and len(d.levels_g) >= 2
        assert len(d.consonants) == len(d.levels_f) * len(d.levels_g)
        for c in d.consonants:                                                  # 固定軸的值真的固定
            for a, v in d.fixed.items():
                assert FEATURES[c][AXIS_INDEX[a]] == v
        cells = {d.level_of(c) for c in d.consonants}
        assert cells == set(itertools.product(d.levels_f, d.levels_g))          # 每個組合都有子音
        # 極大：再加任何一個水準就會有空格
        for extra in range(3):
            if extra not in d.levels_f:
                assert not all(any(FEATURES[c][AXIS_INDEX[d.axis_f]] == extra and FEATURES[c][AXIS_INDEX[d.axis_g]] == lg
                                   and all(FEATURES[c][AXIS_INDEX[a]] == v for a, v in d.fixed.items())
                                   for c in CONSONANTS) for lg in d.levels_g)
            if extra not in d.levels_g:
                assert not all(any(FEATURES[c][AXIS_INDEX[d.axis_g]] == extra and FEATURES[c][AXIS_INDEX[d.axis_f]] == lf
                                   and all(FEATURES[c][AXIS_INDEX[a]] == v for a, v in d.fixed.items())
                                   for c in CONSONANTS) for lf in d.levels_f)
    assert len({d.name for d in designs}) == len(designs)                      # 名字不撞


def test_known_designs_present():
    by = _by_set(enumerate_subdesigns())
    expected = {("voicing", "place", "2x3"): "p t k b d g",
                ("voicing", "place", "2x2"): "f theta v eth",
                ("voicing", "manner", "2x2"): "p f b v",
                ("voicing", "manner", "2x3"): "t theta s d eth z",
                ("nasality", "place", "2x2"): "b d m n",
                ("manner", "place", "2x2"): "p t f theta"}
    for (f, g, size), cons in expected.items():
        d = by[frozenset(cons.split())]
        assert (d.axis_f, d.axis_g, d.size) == (f, g, size)
    for cons in ("s sh z zh", "k sh g zh", "b d v eth", "t k s sh", "d g z zh"):
        assert frozenset(cons.split()) in by
    assert frozenset("m n b d g".split()) not in by and frozenset("p t k".split()) not in by


def test_submatrix_matches_json():
    raw = json.load(open(TABLES_JSON, encoding="utf-8"))["tables"]
    tables = load_tables()
    d = _by_set(enumerate_subdesigns())[frozenset("p t k b d g".split())]
    for snr in ("-12", "-6", "0", "6"):
        sub, retained = submatrix(tables[snr], d)
        for i, a in enumerate(d.consonants):
            for j, b in enumerate(d.consonants):
                assert sub[i, j] == raw[snr][a].get(b, 0)
            row_full = sum(raw[snr][a].values())
            assert abs(retained[i] - sub[i].sum() / row_full) < 1e-12
            assert 0 < retained[i] <= 1


def test_long_csv_roundtrip(tmp_path):
    tables = load_tables()
    for d in enumerate_subdesigns():
        sub, _ = submatrix(tables["-6"], d)
        rows = long_rows(sub, d)
        assert len(rows) == len(d.consonants) ** 2
        path = tmp_path / f"{d.name}.csv"
        write_long_csv(path, rows)
        back = read_long_csv(path)
        assert np.array_equal(long_to_matrix(back, d), sub)
        for r in back:
            assert (r["stim_f"], r["stim_g"]) == d.level_of(r["stim_cons"])
            assert (r["resp_f"], r["resp_g"]) == d.level_of(r["resp_cons"])


def test_dprime_helpers():
    assert abs(adjacent_dprimes(np.array([[50., 50.], [50., 50.]]))[0]) < 1e-12
    d3 = adjacent_dprimes(np.array([[90., 10., 0.], [10., 80., 10.], [0., 10., 90.]]))
    assert len(d3) == 2 and d3[0] > 1 and abs(d3[0] - d3[1]) < 1e-9
    d = _by_set(enumerate_subdesigns())[frozenset("p t k b d g".split())]
    sub, _ = submatrix(load_tables()["0"], d)
    md = marginal_dprimes(sub, d, "place")
    assert set(md) == {"overall", "voicing=0", "voicing=1"} and len(md["overall"]) == 2


def test_run_writes_index(tmp_path):
    designs, index_rows, summary_rows = run(out_dir=str(tmp_path), snrs=("-6",), verbose=False)
    assert len(index_rows) == len(designs) == 11
    for r in index_rows:
        assert (tmp_path / r["file"]).exists() and 0 < r["prop_retained"] <= 1
    assert (tmp_path / "index.csv").exists() and (tmp_path / "python_sdt_summary.txt").exists()
