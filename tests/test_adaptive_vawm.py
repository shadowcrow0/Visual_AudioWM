"""adaptive_vawm.py：模擬受試者跑完兩個校準區塊 → H/L → block csv 欄位與 practice.csv 相同、數值合理。"""
import csv
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vawm import (CSV_COLUMNS, AudioCalibrator, ColourCalibrator, audio_x, count_from_x, delta_e,  # noqa: E402
                           find_colour_at, is_in_gamut, lab_to_hex, load_confusion, random_target,
                           read_calibration, save_calibration, write_blocks)


def test_colour_helpers():
    rng = np.random.default_rng(0)
    t = random_target(rng)
    assert is_in_gamut(t) and lab_to_hex(t).startswith("#") and len(lab_to_hex(t)) == 7
    f = find_colour_at(t, 25.0, rng)
    assert f is not None and abs(delta_e(t, f) - 25.0) <= 0.5
    assert lab_to_hex([50, 0, 0]) in ("#767676", "#777777")                     # 中性灰（int() 截尾同 colorpool）


def test_confusion_table_and_axis():
    conf = load_confusion()
    assert len(conf) == 16 and ("k", 207) in conf["p"]                          # generate_trials.py 的第一列
    assert audio_x(1) > audio_x(451) and abs(count_from_x(audio_x(91)) - 91) < 1e-9


def _observer(alpha, beta, fa, lapse=0.02):
    return lambda x, is_match, rng: (int(rng.uniform() < fa) if is_match
                                     else int(rng.uniform() < fa + (1 - fa - lapse) * norm.cdf(x, alpha, beta)))


def test_colour_calibrator_recovers_observer():
    rng = np.random.default_rng(1)
    obs = _observer(alpha=12.0, beta=8.0, fa=0.15)
    cal = ColourCalibrator(p_high=0.90, p_low=0.75, p_match=1 / 3, seed=1)
    for _ in range(120):
        t = cal.next_trial()
        assert {"color1_target", "color2_target", "audio1_target_file", "probe_colour", "probe_sound"} <= set(t)
        r = obs(t["x"], t["is_match"], rng)
        cal.record("n" if r else "y", 0.6)
    res = cal.finish()
    assert res.dim == "colour" and res.n_match > 20
    assert abs(res.false_alarm - 0.15) < 0.12
    assert abs(res.alpha - 12) < 5 and abs(res.beta - 8) < 5
    assert res.low < res.high and res.in_range
    x_high_true = 12 + 8 * norm.ppf((0.90 - 0.15) / 0.83)
    assert abs(res.high - x_high_true) < 8


def test_audio_calibrator_snaps_to_available_foils():
    rng = np.random.default_rng(2)
    obs = _observer(alpha=-1.0, beta=0.6, fa=0.1)
    cal = AudioCalibrator(p_high=0.90, p_low=0.75, p_match=1 / 3, seed=2)
    conf = load_confusion()
    for _ in range(120):
        t = cal.next_trial()
        if not t["is_match"]:
            target_cons = os.path.basename(t["audio1_target_file" if t["probe_of"] == 1 else "audio2_target_file"])
            target_cons = target_cons.split("_")[1][:-5]
            assert (t["foil"], t["count"]) in conf[target_cons]                # foil 一定是該子音表裡的
            assert abs(t["x"] - audio_x(t["count"])) < 1e-12
        r = obs(t["x"], t["is_match"], rng)
        cal.record("n" if r else "y", 0.7)
    res = cal.finish()
    assert res.dim == "audio" and abs(res.alpha + 1.0) < 0.6
    assert res.extra["count_high"] < res.extra["count_low"]                    # H = 不易混淆 = 次數少


def test_record_validation_and_no_response():
    cal = ColourCalibrator(seed=3)
    with pytest.raises(RuntimeError):
        cal.record("y", 0.5)
    cal.next_trial()
    with pytest.raises(ValueError):
        cal.record("x", 0.5)
    r, c = cal.record(None, None)
    assert r is None and c is None and cal.log[-1]["key"] is None


def test_write_blocks_matches_practice_csv_layout(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "stimuli", "practice.csv"), newline="", encoding="utf-8-sig") as f:
        practice_cols = next(csv.reader(f))
    assert practice_cols == CSV_COLUMNS
    # 假的校準結果：ΔE_H 30、ΔE_L 20；count_H 1、count_L 90
    from adaptive_vawm import CalibResult
    colour = CalibResult("colour", 72, 24, 0.1, 15, 8, 30.0, 20.0, 0.9, 0.75, (2, 50), True)
    audio = CalibResult("audio", 72, 24, 0.1, -1, 0.5, -0.3, -1.96, 0.9, 0.75, (-2.7, 0), True,
                        extra=dict(count_high=1.0, count_low=90.0))
    path = tmp_path / "S_calib.json"
    save_calibration(path, colour, audio, info={"participant": "S"})
    cal = read_calibration(path)
    paths = write_blocks(cal, tmp_path / "stimuli", n_blocks=2, trials_per_block=5, seed=7)
    assert [os.path.basename(p) for p in paths] == ["block1.csv", "block2.csv"]
    with open(paths[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == CSV_COLUMNS and len(rows) == 5
    for r in rows:
        assert abs(float(r["color1_H_deltaE"]) - 30) <= 0.5 and abs(float(r["color1_L_deltaE"]) - 20) <= 0.5
        assert int(r["audio1_H_count"]) <= int(r["audio1_L_count"])
        assert r["audio1_H_file"].startswith("stimuli/T") and r["audio1_H_file"].endswith("3.wav")
        assert r["audio1_target_talker"] == r["audio1_H_talker"] == r["audio1_L_talker"]


def test_vawm_csv_to_dfp_analysis(tmp_path):
    """假造一份 VAWM_nobox.py 格式的輸出（每個探測一列），PAR-OR 的 DDM 填 RT → 判 ParallelOR。
    分析端要完整的 adaptivesft 套件（PyMC）；實驗機器沒有就跳過，其餘六個測試不受影響。"""
    pytest.importorskip("pymc")
    from adaptive_vawm import _import_experiment, analyze_vawm, vawm_to_dfp_rows
    _import_experiment()
    from adaptivesft.ddm import dfp_ddm
    rng = np.random.default_rng(9)
    drift = {2: 3.0, 1: 1.0}
    recs = []
    for cond in list(range(18)) * 40:
        if cond in (8, 9, 14, 15, 10, 11, 16, 17):
            c1, c2 = {8: (2, 2), 9: (2, 2), 14: (2, 1), 15: (2, 1), 10: (1, 2), 11: (1, 2), 16: (1, 1), 17: (1, 1)}[cond]
            rt, cr = dfp_ddm(1, drift[c1], drift[c2], 3.0, 0.1, 0.2, "PAR", "OR", rng=rng)
            key = "n" if cr[0] == 1 else "y"
            rt = float(rt[0])
        else:
            key, rt = ("y" if cond in (0, 1) else "n"), 0.8
        recs.append({"participant": "S9", "condition": cond, "ResponseBox.keys": key, "ResponseBox.rt": rt, "block1.thisN": 0})
    path = tmp_path / "S9_VAWM.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    rows, others = vawm_to_dfp_rows(path)
    assert len(rows) == 8 * 40 and len(others) == 10 * 40 and rows[0]["subject"] == "S9"
    out = analyze_vawm(path)
    assert out["n_dfp"] == 320 and out["accuracy_other"][0] == 1.0
    assert out["result"]["DFP"]["classification"]["Predicted_by"] == "ParallelOR"
    assert "ParallelOR" in out["report"]


def _lnrm_observer(alpha, alpha2, mu=1.0, varZ=0.5, psi=0.2, fa=0.1, rng=None):
    """原始 adaptiveSFT 的生成模型：d(u) = α·u + α₂·u²，兩個對數常態累積器賽跑（adaptivesft.race.lnrm_random）。"""
    from adaptive_vawm import _import_experiment
    _import_experiment()
    from adaptivesft.race import lnrm_random

    def respond(cal, t):
        if t["is_match"]:
            key, rt = ("n" if rng.uniform() < fa else "y"), float(np.exp(rng.normal(mu, varZ)) * 0.5 + psi)
        else:
            u = cal._to_model(t["x"])
            rt, correct = lnrm_random([alpha * u + alpha2 * u * u], mu, varZ, psi, rng)
            key, rt = ("n" if correct[0] == 1 else "y"), float(rt[0])
        return key, rt
    return respond


@pytest.mark.parametrize("dim", ["colour", "audio"])
def test_lnrm_calibrator_recovers_and_matches_adaptivesft_inversion(dim):
    """定值刺激法 + lnrm2 擬合 + 反解漂移差：回復 α/α₂，H/L 與 adaptivesft.salience 的反解一致，H > L。"""
    pytest.importorskip("pymc")
    from adaptive_vawm import AudioLNRMCalibrator, ColourLNRMCalibrator
    rng = np.random.default_rng(11)
    alpha, alpha2 = 1.0, -0.12
    Cal = ColourLNRMCalibrator if dim == "colour" else AudioLNRMCalibrator
    cal = Cal(n_per_level=12, seed=11, tune=400, draws=400, chains=2)
    respond = _lnrm_observer(alpha, alpha2, rng=rng)
    assert cal.n_trials == 6 * 12 + 36
    for _ in range(cal.n_trials):
        t = cal.next_trial()
        assert t["x_index"] is None and {"probe_colour", "probe_sound"} <= set(t)
        cal.record(*respond(cal, t))
    with pytest.raises(StopIteration):
        cal.next_trial()
    res = cal.finish()
    assert res.dim == dim and res.extra["method"] == "lnrm_quadratic" and res.n_match == 36
    assert abs(res.alpha - alpha) < 0.5 and abs(res.beta - alpha2) < 0.15
    assert res.extra["divergences"] < 20
    # 反解：2·d(u) = targ 的較小根（adaptiveSFT_functions.R:229-232）
    a, a2 = res.alpha, res.beta
    u_h = (-a / a2 - np.sqrt((a / a2) ** 2 + 2 / a2 * cal.h_targ)) / 2
    assert abs(cal._to_model(res.high) - u_h) < 0.15                  # posterior-mean 反解 vs 逐 draw 平均
    assert res.low < res.high and res.in_range
    if dim == "audio":
        assert res.extra["count_high"] < res.extra["count_low"]


def test_lnrm_calibrator_ogival_link_decision_b():
    """LINK = "ogival"（決定 B：½L·inv_logit、L = 10 固定）配 Houpt 的 8.0 / 1.3：回復 slope / midpoint，H/L 對上閉式反解。"""
    pytest.importorskip("pymc")
    from adaptive_vawm import ColourLNRMCalibrator, _import_experiment
    _import_experiment()
    from adaptivesft.race import lnrm_random
    rng = np.random.default_rng(6)
    slope, mid, L = 2.0, 1.5, 10.0

    def respond(cal, t):
        if t["is_match"]:
            return ("n" if rng.uniform() < 0.1 else "y"), 0.8
        u = cal._to_model(t["x"])
        rt, c = lnrm_random([0.5 * L / (1 + np.exp(-slope * (u - mid)))], 1.0, 0.5, 0.2, rng)
        return ("n" if c[0] == 1 else "y"), float(rt[0])

    cal = ColourLNRMCalibrator(n_per_level=12, seed=6, link="ogival", h_targ=8.0, l_targ=1.3, tune=400, draws=400, chains=2)
    for _ in range(cal.n_trials):
        cal.record(*respond(cal, cal.next_trial()))
    res = cal.finish()
    p = res.extra["params"]
    assert res.extra["method"] == "lnrm_ogival" and abs(p["slope"] - slope) < 0.6 and abs(p["midpoint"] - mid) < 0.3
    inv = lambda targ: (np.log((targ / L) / (1 - targ / L)) / slope + mid) * 10        # adaptiveSFT_functions.R:180-199
    assert abs(res.high - inv(8.0)) < 3 and abs(res.low - inv(1.3)) < 3 and res.in_range
