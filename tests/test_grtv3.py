"""grtv3 的純邏輯層測試：計畫、版面、計分、休息訊息、顏色 LUT、遮蔽噪音、校準水準規則。不需要 PsychoPy。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grtv3 import config as C                                                   # noqa: E402
from grtv3.calibration import Calibration, levels_for_adapt                      # noqa: E402
from grtv3.colour import ColourLUT                                               # noqa: E402
from grtv3.masking import MaskNoise, mask_reference, visual_mask_textures         # noqa: E402
from grtv3.plan import (adapt_correct, adapt_response, build_plan, build_main_plan, feedback_ok, layout_trial,  # noqa: E402
                        option_labels, option_permutation, rest_message, score_response)


def test_main_plan_counts_and_block_ratio():
    rng = np.random.default_rng(0)
    plan, n_blocks = build_main_plan(rng)
    assert len(plan) == 576 and n_blocks == 4
    for b in range(n_blocks):
        blk = plan[b * C.BLOCK_SIZE:(b + 1) * C.BLOCK_SIZE]
        valid = sum(t[0] for t in blk)
        assert valid == 96 and len(blk) - valid == 48                      # 2:1 in every block
    cells = {}
    for t in plan:
        cells[t] = cells.get(t, 0) + 1
    assert all(v == 24 for k, v in cells.items() if k[0] == 1)
    assert all(v == 4 for k, v in cells.items() if k[0] == 0)


def test_full_plan_and_phases():
    plan = build_plan(np.random.default_rng(1))
    assert plan.n_trials == 72 + 72 + 15 + 576
    assert plan.phase_of(0) == "adapt_col" and plan.phase_of(71) == "adapt_col"
    assert plan.phase_of(72) == "adapt_snd" and plan.phase_of(143) == "adapt_snd"
    assert plan.phase_of(144) == "practice" and plan.phase_of(158) == "practice" and plan.phase_of(159) == "main"
    pr = plan.trials[144:159]
    assert sum(t[0] for t in pr) == 10 and len(set(pr)) == 15                # 10 valid + 5 invalid, no repeats
    assert all(t[1] is None and t[0] == 1 for t in plan.trials[:144])        # calibration: valid, item chosen by Psi
    assert sorted(t[3] for t in plan.trials[:72]) == sorted([i % 4 for i in range(72)])


def test_layout_puts_target_in_its_serial_slot():
    rng = np.random.default_rng(2)
    for target_item in range(4):
        for serial in range(4):
            lay = layout_trial(rng, target_item, serial)
            assert sorted(lay.quad_content) == [0, 1, 2, 3] and sorted(lay.time_perm) == [0, 1, 2, 3]
            assert lay.quad_content[lay.target_quad] == target_item
            assert lay.time_perm[lay.target_quad] == serial
            assert lay.start_times[lay.target_quad] == C.STUDY_START[serial]


def test_scoring_matches_builder_rules():
    opt = [2, 0, 3, 1]                                   # UR=2 UL=0 BL=3 BR=1
    assert score_response("o", opt, target_item=2, cued_item=2).outcome == "correct"
    s = score_response("i", opt, target_item=2, cued_item=0)                 # UL -> item 0 = cued
    assert s.outcome == "intrusion" and s.err_rel == 2 and not s.is_correct
    s = score_response("l", opt, target_item=2, cued_item=0)                 # BR -> item 1
    assert s.outcome == "other" and s.err_rel == 3
    s = score_response(None, opt, 2, 0)
    assert s.outcome == "noresp" and s.chosen_item is None and s.err_rel is None
    assert score_response(["i", "o"], opt, 2, 2).outcome == "correct"        # list -> last key


def test_adapt_response_and_feedback():
    assert adapt_response("adapt_col", 3) == 1 and adapt_response("adapt_col", 2) == 0
    assert adapt_response("adapt_snd", 1) == 1 and adapt_response("adapt_snd", 3) == 0
    assert adapt_correct(1, +0.5) == 1 and adapt_correct(1, -0.5) == 0
    assert feedback_ok("adapt_col", 1, 3) and not feedback_ok("adapt_col", 0, 3)
    assert feedback_ok("adapt_snd", 2, 3) and not feedback_ok("adapt_snd", 1, 3)
    assert feedback_ok("practice", 3, 3) and not feedback_ok("practice", 1, 3)


def test_option_labels_follow_snr_sign():
    assert option_labels([12.0, -12.0]) == ["[bi]", "[bi]", "[pi]", "[pi]"]
    assert option_labels([12.0, 12.0]) == ["[bi]"] * 4                       # colour calibration: all clear /bi/
    assert sorted(option_permutation(np.random.default_rng(3))) == [0, 1, 2, 3]


def test_rest_messages_at_the_right_trials():
    plan = build_plan(np.random.default_rng(4))
    assert rest_message(72, plan, 0).startswith("Part 1")
    assert rest_message(144, plan, 0).startswith("Part 2")
    assert "10 of 15" in rest_message(159, plan, 10)
    assert rest_message(159 + 144, plan, 0).startswith("Block 1 of 4")
    assert rest_message(159 + 576, plan, 0) is None                          # end: no rest screen
    assert rest_message(10, plan, 0) is None and rest_message(200, plan, 0) is None


def test_colour_lut():
    lut = ColourLUT()
    assert lut.hex_for(0.0).startswith("#") and lut.arc_lim > 3.0
    assert lut.hex_for(3.0) != lut.hex_for(-3.0)
    with pytest.raises(ValueError):
        lut.hex_for(lut.arc_max + 1.0)
    assert lut.clip(lut.arc_max + 5) == lut.arc_max


def test_levels_for_adapt():
    lut = ColourLUT(); rng = np.random.default_rng(5)
    lv, item = levels_for_adapt("adapt_col", +2.0, lut, rng)
    assert lv.colour_arc == [-2.0, 2.0] and lv.snr_hi == lv.snr_lo == C.ADAPT_FIX_SNR and (item & 1) == 1
    lv, item = levels_for_adapt("adapt_col", -2.0, lut, rng)
    assert (item & 1) == 0
    lv, item = levels_for_adapt("adapt_snd", -6.0, lut, rng)
    assert lv.colour_arc == [0.0, 0.0] and lv.snr_levels == [6.0, -6.0] and (item >> 1) == 1
    assert lv.item_hex[0] == lv.item_hex[2]


class FakePsi:
    def __init__(self, x):
        self.nextIntensity = x; self.updates = []

    def update(self, r):
        self.updates.append(r)


class FakeHandler:
    def __init__(self):
        self._psi1, self._psi2 = FakePsi(1.5), FakePsi(-4.0)
        self.saved = None

    def estimateLambda(self):
        return [(1.0, 0.5), (2.0, 1.0)]

    def estimateGRTintensities(self, acc, lams):
        return (-30.0, 30.0), (-25.0, 25.0)                                  # deliberately out of range

    def savePosterior(self, p):
        self.saved = p


def test_calibration_wrapper_with_fake_handler(tmp_path):
    lut = ColourLUT(); rng = np.random.default_rng(6)
    cal = Calibration(lut, handler=FakeHandler())
    x, lv, item = cal.propose("adapt_col", rng)
    assert x == 1.5 and lv.colour_arc == [-1.5, 1.5]
    cal.update("adapt_col", 1); cal.update("adapt_snd", 0)
    assert cal.agrt._psi1.updates == [1] and cal.agrt._psi2.updates == [0]
    levels, lams, raw = cal.finish(0.64, posterior_path=str(tmp_path / "post"))
    assert levels.colour_arc == [lut.arc_min, lut.arc_max] and levels.snr_hi == C.SNR_LIM and levels.snr_lo == -C.SNR_LIM
    assert raw["c_hi"] == 30.0 and cal.agrt.saved.endswith("post")


def test_mask_noise(tmp_path):
    sr, ref = mask_reference(n_seeds=2)
    assert sr > 8000 and 0 < ref < 1
    m = MaskNoise(str(tmp_path), sr, ref).make(7)
    assert os.path.exists(m["path"]) and m["summary"].startswith("ssn|+0.00dB|seed=")
    tex = visual_mask_textures(123)
    assert len(tex) == 4 and tex[0].shape == (C.MASK_TEX_N, C.MASK_TEX_N) and np.abs(tex[0]).max() <= 1.0
    assert np.allclose(visual_mask_textures(123)[1], tex[1])                 # reproducible from the seed
