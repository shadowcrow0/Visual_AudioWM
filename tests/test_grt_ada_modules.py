"""grt_stimuli.py / grt_psi.py 的測試。

AGRT.py 頂層 import PsychoPy,而開發容器沒有 PsychoPy —— 這裡塞最小替身進 sys.modules,
讓**真正的** AGRTHandler / agrtPsiObject 跑起來(只替換 StairHandler 基底與 logging,
Psi 的數學不碰)。
"""
import os
import sys
import types

import numpy as np
import pytest
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_psychopy():
    if 'psychopy' in sys.modules:
        return
    class StairHandler:
        def __init__(self, *a, **kw):
            pass
    mods = {
        'psychopy': types.ModuleType('psychopy'),
        'psychopy.logging': types.ModuleType('psychopy.logging'),
        'psychopy.tools': types.ModuleType('psychopy.tools'),
        'psychopy.tools.fileerrortools': types.ModuleType('psychopy.tools.fileerrortools'),
        'psychopy.data': types.ModuleType('psychopy.data'),
        'psychopy.data.staircase': types.ModuleType('psychopy.data.staircase'),
    }
    mods['psychopy.logging'].warning = lambda *a, **kw: None
    mods['psychopy.tools.fileerrortools'].handleFileCollision = lambda f, **kw: f + '_1'
    mods['psychopy.data'].TrialHandler = object
    mods['psychopy.data.staircase'].StairHandler = StairHandler
    mods['psychopy'].logging = mods['psychopy.logging']
    sys.modules.update(mods)


_stub_psychopy()

import grt_stimuli as gs                                       # noqa: E402
from grt_psi import PsiCalibration                             # noqa: E402

AXIS = gs.ColourAxis.load(os.path.join(ROOT, 'agrt_colour_lut.json'))


# ---------------------------------------------------------------- 編碼與版面

def test_item_encoding():
    for i in range(4):
        assert gs.make_item(gs.colour_bit(i), gs.sound_bit(i)) == i
    assert [gs.colour_bit(i) for i in range(4)] == [0, 1, 0, 1]
    assert [gs.sound_bit(i) for i in range(4)] == [0, 0, 1, 1]


def test_colour_axis():
    assert AXIS.arc_min < 0 < AXIS.arc_max and AXIS.half_range > 20
    assert AXIS.hex(0.0).startswith('#') and len(AXIS.hex(0.0)) == 7
    with pytest.raises(ValueError):
        AXIS.hex(AXIS.arc_max + 1)
    assert AXIS.clip(1e6) == AXIS.arc_max


def test_levels_items_and_labels():
    lv = gs.Levels(colour_arc=(-3.0, 3.0), snr_hi=6.0, snr_lo=-6.0)
    hexes = lv.item_hex(AXIS)
    assert hexes[0] == hexes[2] != hexes[1] == hexes[3]
    assert [lv.item_snr(i) for i in range(4)] == [6.0, 6.0, -6.0, -6.0]
    assert lv.item_labels() == ['[bi]', '[bi]', '[pi]', '[pi]']
    # 校顏色時兩級同為 +12 dB -> 四個都是 [bi](原腳本刻意如此)
    assert gs.Levels((-1, 1), 12.0, 12.0).item_labels() == ['[bi]'] * 4


def test_study_layout_puts_target_at_planned_serial():
    rng = np.random.default_rng(1)
    for _ in range(200):
        t, sp = int(rng.integers(4)), int(rng.integers(4))
        lay = gs.study_layout(t, sp, rng)
        assert sorted(lay.quad_content) == [0, 1, 2, 3] and sorted(lay.time_perm) == [0, 1, 2, 3]
        assert lay.quad_content[lay.target_quad] == t
        assert lay.time_perm[lay.target_quad] == sp
        assert lay.onset(lay.target_quad) == gs.STUDY_ONSETS[sp]


def test_cue_and_score():
    lay = gs.StudyLayout(quad_content=[2, 0, 3, 1], time_perm=[0, 1, 2, 3], target_quad=1)
    for rel in range(4):
        cued, q = gs.cue_layout(0, rel, lay)
        assert cued == 0 ^ rel and lay.quad_content[q] == cued
    # valid:cued == target,答對必須是 correct 不是 intrusion
    assert gs.score(0, 0, 0) == ('correct', None)
    assert gs.score(None, 0, 1) == ('noresp', None)
    assert gs.score(1, 0, 1) == ('intrusion', 1)
    assert gs.score(3, 0, 1) == ('other', 3)


def test_main_plan_balance():
    plan, n_blocks = gs.main_plan(np.random.default_rng(0))
    assert len(plan) == 576 and n_blocks == 4
    for b in range(4):
        blk = plan[b * 144:(b + 1) * 144]
        n_valid = sum(t[0] for t in blk)
        assert n_valid == 96 and len(blk) - n_valid == 48             # 每個 block 都是 2:1
    for t in plan:
        assert (t[0] == 1) == (t[2] == 0)


def test_adapt_and_practice_plans():
    rng = np.random.default_rng(0)
    ap = gs.adapt_plan(72, rng)
    assert len(ap) == 72 and all(t[0] == 1 and t[1] is None and t[2] == 0 for t in ap)
    assert [sum(t[3] == s for t in ap) for s in range(4)] == [18] * 4
    pp = gs.practice_plan(15, rng)
    assert len(pp) == 15 and sum(t[0] for t in pp) == 10


# ---------------------------------------------------------------- Psi

@pytest.fixture(scope='module')
def cal():
    return PsiCalibration(AXIS.half_range, snr_lim=18.0, n_col=72, n_snd=72)


def test_phase_of(cal):
    assert cal.phase_of(0) == 'adapt_col' and cal.phase_of(71) == 'adapt_col'
    assert cal.phase_of(72) == 'adapt_snd' and cal.phase_of(143) == 'adapt_snd'
    assert cal.phase_of(144) is None


def test_propose_puts_target_on_x_side(cal):
    rng = np.random.default_rng(0)
    t = cal.propose('adapt_col', rng)
    assert t.levels.colour_arc == (-abs(t.x), abs(t.x)) and t.levels.snr_hi == t.levels.snr_lo == 12.0
    assert t.levels.colour_arc[gs.colour_bit(t.target_item)] == pytest.approx(t.x)
    t = cal.propose('adapt_snd', rng)
    assert t.levels.colour_arc == (0.0, 0.0)
    assert t.levels.item_snr(t.target_item) == pytest.approx(t.x)
    assert PsiCalibration.response_bit('adapt_snd', 0) == 1          # item 0 = [bi] = 高端
    assert PsiCalibration.response_bit('adapt_col', 1) == 1          # item 1 = pink = 高端


def _simulate(alpha, beta, seed, lapse=0.04):
    """模擬受試者：P(答高端 | x) = lapse/2 + (1-lapse)·Φ((x-α)/β),兩維各自獨立。"""
    rng = np.random.default_rng(seed)
    cal = PsiCalibration(AXIS.half_range, snr_lim=18.0, n_col=72, n_snd=72)
    for i in range(cal.n_total):
        phase = cal.phase_of(i)
        t = cal.propose(phase, rng)
        a, b = alpha[phase], beta[phase]
        p_hi = lapse / 2 + (1 - lapse) * norm.cdf((t.x - a) / b)
        say_hi = rng.random() < p_hi
        # 從四個 item 中選一個在被校那一維符合反應的
        cands = [it for it in range(4) if PsiCalibration.response_bit(phase, it) == int(say_hi)]
        resp, corr = cal.update(t, int(rng.choice(cands)))
        assert resp == int(say_hi) and corr == int(resp == t.high_side)
    return cal


def test_psi_recovers_alpha_and_orders_levels():
    alpha = {'adapt_col': 2.0, 'adapt_snd': -3.0}
    beta = {'adapt_col': 3.0, 'adapt_snd': 4.0}
    cal = _simulate(alpha, beta, seed=3)
    res = cal.finish(AXIS)
    (ca, cb), (sa, sb) = res.lambdas
    assert abs(ca - 2.0) < 2.0 and abs(sa + 3.0) < 3.0
    lv = res.levels
    assert lv.colour_arc[0] < lv.colour_arc[1] and lv.snr_lo < lv.snr_hi
    assert set(res.data) >= {'psi_col_alpha', 'final_snr_hi', 'final_colour_arc_lo'}


def test_no_response_does_not_update(cal):
    t = cal.propose('adapt_col', np.random.default_rng(0))
    before = cal.handler._psi1.nextIntensity
    assert cal.update(t, None) == (None, None)
    assert cal.handler._psi1.nextIntensity == before
