"""兩個獨立 Psi（AGRT.AGRTHandler 內的 _psi1 = 顏色、_psi2 = 聲音）的包裝。

水準的規則（GRTv3_ada_EN.py:1139-1156, 2379-2416）：
  adapt_col：Psi 提議 x，兩個顏色 = ±|x|，四格同一個清楚的 /bi/（ADAPT_FIX_SNR）；目標的 colour bit 由 x 的正負決定
  adapt_snd：Psi 提議 s，兩個 SNR = ±|s|，四格同一色（ADAPT_FIX_ARC）；目標的 sound bit 由 s 的正負決定
  收尾：estimateGRTintensities(OVERALL_ACC) → 主實驗的兩個顏色與兩個 SNR，截在可呈現範圍內
"""
from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class Levels:
    colour_arc: list     # [lo, hi]
    colour_hex: list
    snr_hi: float
    snr_lo: float

    @property
    def snr_levels(self):
        return [self.snr_hi, self.snr_lo]        # index 0 = hi_snr([bi]), 1 = lo_snr([pi])

    @property
    def item_hex(self):
        return [self.colour_hex[0], self.colour_hex[1], self.colour_hex[0], self.colour_hex[1]]


def levels_for_adapt(phase, adapt_x, lut, rng, fix_arc=C.ADAPT_FIX_ARC, fix_snr=C.ADAPT_FIX_SNR):
    """純邏輯：給 Psi 提議值，回傳 (Levels, target_item)。
    adapt_col / adapt_snd：adapt_x 是一個數，另一維固定、目標的另一個 bit 隨機；
    adapt_joint：adapt_x = (x, s)，兩維都是 ±|·|，目標的兩個 bit 都由正負決定（沒有隨機成分）。"""
    if phase == "adapt_joint":
        x, s = adapt_x
        arc = [-abs(x), abs(x)]
        lv = Levels(arc, [lut.hex_for(a) for a in arc], abs(s), -abs(s))
        cb = 1 if x > 0 else 0
        sb = 0 if s > 0 else 1
        return lv, cb + 2 * sb
    if phase == "adapt_col":
        arc = [-abs(adapt_x), abs(adapt_x)]
        lv = Levels(arc, [lut.hex_for(a) for a in arc], fix_snr, fix_snr)
        cb = 1 if adapt_x > 0 else 0
        sb = int(rng.integers(2))
    elif phase == "adapt_snd":
        arc = [fix_arc, fix_arc]
        lv = Levels(arc, [lut.hex_for(a) for a in arc], abs(adapt_x), -abs(adapt_x))
        sb = 0 if adapt_x > 0 else 1
        cb = int(rng.integers(2))
    else:
        raise ValueError(phase)
    return lv, cb + 2 * sb


class Calibration:
    def __init__(self, lut, n_col=C.N_ADAPT_COL, n_snd=C.N_ADAPT_SND, lapse=C.LAPSE, snr_lim=C.SNR_LIM,
                 steps=C.PSI_STEPS, handler=None, joint=bool(C.CALIB_JOINT), n_joint=C.N_ADAPT_JOINT):
        self.lut, self.snr_lim, self.joint = lut, snr_lim, joint
        n_trials = n_joint if joint else n_col + n_snd
        if handler is None:
            from AGRT import AGRTHandler                      # 需要 PsychoPy（StairHandler）
            handler = AGRTHandler(nTrials=n_trials, lapse=lapse,
                                  dim1range=[-lut.arc_lim, lut.arc_lim], dim2range=[-snr_lim, snr_lim],
                                  dim1steps=steps, dim2steps=steps)
        self.agrt = handler
        self.lambdas = None

    def psi(self, phase):
        return self.agrt._psi1 if phase == "adapt_col" else self.agrt._psi2

    def propose(self, phase, rng):
        """回傳 (adapt_x, Levels, target_item)。joint 走 AGRTHandler.__next__（兩維一起，handler 自己記 intensities）。"""
        if phase == "adapt_joint":
            x = self.agrt.next()
            if x is None:
                raise RuntimeError("AGRTHandler has no more joint trials (nTrials reached)")
            x = (float(x[0]), float(x[1]))
        else:
            x = float(self.psi(phase).nextIntensity)
        lv, item = levels_for_adapt(phase, x, self.lut, rng)
        return x, lv, item

    def update(self, phase, resp):
        """分開校：更新一個 Psi；joint：AGRTHandler.addResponse((colour_resp, sound_resp)) 同時更新兩個。"""
        if phase == "adapt_joint":
            self.agrt.addResponse(tuple(resp))
        else:
            self.psi(phase).update(resp)

    def finish(self, overall_acc=C.OVERALL_ACC, posterior_path=None):
        """估 α/β，反解主實驗水準，截到可呈現範圍。回傳 (Levels, lambdas, raw)。"""
        lams = self.agrt.estimateLambda()
        (c_lo, c_hi), (s_lo, s_hi) = self.agrt.estimateGRTintensities(overall_acc, lams)
        raw = dict(c_lo=c_lo, c_hi=c_hi, s_lo=s_lo, s_hi=s_hi)
        c_lo, c_hi = self.lut.clip(c_lo), self.lut.clip(c_hi)
        s_lo = float(np.clip(s_lo, -self.snr_lim, self.snr_lim))
        s_hi = float(np.clip(s_hi, -self.snr_lim, self.snr_lim))
        if posterior_path:
            self.agrt.savePosterior(posterior_path)
        self.lambdas = lams
        return Levels([c_lo, c_hi], [self.lut.hex_for(c_lo), self.lut.hex_for(c_hi)], s_hi, s_lo), lams, raw
