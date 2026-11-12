"""
GRTv3_ada 的 Psi 校準(區塊 1 顏色、區塊 2 聲音)。包住 AGRT.AGRTHandler 的兩條
獨立 Psi(Kontsevich & Tyler 1999;GRT 變體 Glavan 2022),不碰畫面。

    cal = PsiCalibration(colour_axis.half_range, snr_lim=18.0)
    trial  = cal.propose('adapt_col', rng)      # -> Psi 提的 x、這一試的 Levels、target_item
    ...呈現、作答...
    resp, corr = cal.update(trial, chosen_item) # 解出被校那一維的位元，餵回 Psi
    ...跑完兩個區塊...
    result = cal.finish(colour_axis)            # -> 主實驗的 Levels + 要記進資料的欄位

模型(AGRT.py agrtPsiObject):P(r=1|x) = δ/2 + (1-δ)·Φ((x-α)/β),值域 [δ/2, 1-δ/2]。
  顏色  r = 1 是 pink(arc > 0 那一側)
  聲音  r = 1 是 [bi](SNR > 0 那一側)
兩者都是「答高端的機率隨 x 從 0 走到 1」,正是這條模型的形狀。

聲音軸為什麼能用這條模型:只播 /bi/ 的話，不給回饋時 P(答 bi) 會在低 SNR 停在 0.5
(聽不清楚就用猜的),模型表示不了。所以校準給逐試回饋(SND_FEEDBACK,在實驗腳本)，
把 b/p 判準錨在 0 dB:SNR > 0 的正解是 [bi],< 0 的正解是 [pi]。代價是估到的 α 是
「回饋把他推到哪」,而且低 SNR 能否真的把 /bi/ 推向 /pi/ 尚未前測(見 CLAUDE.md)。

AGRT.py 頂層 import 了 PsychoPy,所以這裡延遲到建構時才載入;測試可以傳 handler_cls 替換。
"""
from dataclasses import dataclass, field

import numpy as np

from grt_stimuli import Levels, colour_bit, make_item, sound_bit

PHASES = ('adapt_col', 'adapt_snd')


@dataclass
class AdaptTrial:
    phase: str
    x: float                # Psi 提出的帶號值：顏色是弧長(dE00),聲音是 SNR(dB)
    levels: Levels
    target_item: int

    @property
    def high_side(self):
        """正解是否在高端(pink / [bi])。"""
        return int(self.x > 0)


@dataclass
class CalibrationResult:
    levels: Levels
    lambdas: tuple          # ((col_alpha, col_beta), (snd_alpha, snd_beta))
    data: dict = field(default_factory=dict)    # 要記進 thisExp 的欄位


class PsiCalibration:
    """
    overall_acc  0.64 -> 聯合正確率 0.80、每維 89.4%。AGRT 內部開兩次根號:
                 estimateGRTintensities 傳 sqrt(0.64)=0.8 給 estimateThreshold,
                 後者再開一次 -> 每維 0.894,兩維相乘 0.80。
    lapse        整體 lapse;handler 內部轉成邊際 lapse 1-sqrt(1-lapse)。
    fix_arc      校聲音時四個項目的顏色(弧長 0 = 錨點色,CIELCh h=303,薰衣草紫)。
    fix_snr      校顏色時四個項目的 SNR(dB;清楚的 /bi/)。
    校準時「另一個維度」只出現一個固定值，四個項目在那一維完全相同，
    這一維不進 Psi,也不影響對錯判定。
    """

    def __init__(self, colour_lim, snr_lim, n_col=72, n_snd=72,
                 overall_acc=0.64, lapse=0.08, fix_arc=0.0, fix_snr=12.0,
                 steps=100, handler_cls=None):
        if handler_cls is None:
            from AGRT import AGRTHandler as handler_cls
        self.colour_lim = float(colour_lim)
        self.snr_lim = float(snr_lim)
        self.n_col, self.n_snd = n_col, n_snd
        self.overall_acc = overall_acc
        self.fix_arc, self.fix_snr = fix_arc, fix_snr
        self.handler = handler_cls(nTrials=n_col + n_snd, lapse=lapse,
                                   dim1range=[-self.colour_lim, self.colour_lim],
                                   dim2range=[-self.snr_lim, self.snr_lim],
                                   dim1steps=steps, dim2steps=steps)
        self._psi = {'adapt_col': self.handler._psi1, 'adapt_snd': self.handler._psi2}

    @property
    def n_total(self):
        return self.n_col + self.n_snd

    def phase_of(self, trial_i):
        """校準段內第 trial_i 試(0 起算)屬於哪個區塊;超出校準段回傳 None。"""
        if trial_i < self.n_col:
            return 'adapt_col'
        if trial_i < self.n_total:
            return 'adapt_snd'
        return None

    def propose(self, phase, rng):
        """Psi 提出一個帶號值 x。兩個項目放在 ±|x|,target 落在 x 的那一側;
        另一維四個項目全部同一個固定值，所以 target 在另一維的位元隨機給
        (只影響 item 編號，不影響內容)。"""
        x = float(self._psi[phase].nextIntensity)
        hi = int(x > 0)
        if phase == 'adapt_col':
            levels = Levels(colour_arc=(-abs(x), abs(x)), snr_hi=self.fix_snr, snr_lo=self.fix_snr)
            target = make_item(cbit=hi, sbit=int(rng.integers(2)))
        elif phase == 'adapt_snd':
            levels = Levels(colour_arc=(self.fix_arc, self.fix_arc), snr_hi=abs(x), snr_lo=-abs(x))
            target = make_item(cbit=int(rng.integers(2)), sbit=1 - hi)   # sound 位元 0 = [bi] = SNR > 0
        else:
            raise ValueError(f"not a calibration phase: {phase!r}")
        return AdaptTrial(phase, x, levels, target)

    @staticmethod
    def response_bit(phase, chosen_item):
        """從選中的 item 解出被校那一維的 r(1 = 高端:pink / [bi])。"""
        if phase == 'adapt_col':
            return colour_bit(chosen_item)
        return 1 - sound_bit(chosen_item)

    def update(self, trial, chosen_item):
        """回傳 (adapt_resp, adapt_corr)。

        另一維四個項目相同，所以選到「內容相同的另一個角落」一樣算對：只看被校的那一維。
        沒作答就不更新、回傳 (None, None),Psi 下一試會再提同一個值。
        """
        if chosen_item is None:
            return None, None
        r = self.response_bit(trial.phase, chosen_item)
        self._psi[trial.phase].update(r)
        return r, int(r == trial.high_side)

    @staticmethod
    def feedback_ok(phase, target_item):
        """回饋要框哪些選項：校準時另一維相同，兩個角落內容一樣，兩個都框;其餘只框 target。"""
        if phase == 'adapt_col':
            return lambda i: colour_bit(i) == colour_bit(target_item)
        if phase == 'adapt_snd':
            return lambda i: sound_bit(i) == sound_bit(target_item)
        return lambda i: i == target_item

    def finish(self, colour_axis):
        """官方估計 -> 主實驗的 Levels。

        estimateGRTintensities 回傳以 α 為中心對稱推開的一對 (lo, hi)。
        顏色夾到查表範圍內;SNR 是連續的(混音器解析度 0.001 dB),估計值直接就是要播的值，
        只夾到 ±snr_lim。lo 是 lo_snr([pi] 端)、hi 是 hi_snr([bi] 端)。
        """
        lams = self.handler.estimateLambda()
        (c_lo, c_hi), (s_lo, s_hi) = self.handler.estimateGRTintensities(self.overall_acc, lams)
        c_lo, c_hi = colour_axis.clip(c_lo), colour_axis.clip(c_hi)
        s_lo = float(np.clip(s_lo, -self.snr_lim, self.snr_lim))
        s_hi = float(np.clip(s_hi, -self.snr_lim, self.snr_lim))
        levels = Levels(colour_arc=(c_lo, c_hi), snr_hi=s_hi, snr_lo=s_lo)
        data = {
            'psi_col_alpha': float(lams[0][0]), 'psi_col_beta': float(lams[0][1]),
            'psi_snd_alpha': float(lams[1][0]), 'psi_snd_beta': float(lams[1][1]),
            'final_colour_arc_lo': c_lo, 'final_colour_arc_hi': c_hi,
            'final_snr_hi': s_hi, 'final_snr_lo': s_lo,
            'snr_lim': self.snr_lim,
        }
        return CalibrationResult(levels, lams, data)

    def save_posterior(self, path):
        self.handler.savePosterior(path)
