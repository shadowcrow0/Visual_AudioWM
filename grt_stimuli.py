"""
GRTv3_ada 的刺激定義與試次計畫。不碰 PsychoPy —— 畫面、播放、計時留在實驗腳本，
這裡只決定「這一試要呈現什麼、放哪裡、怎麼計分」，所以可以在沒有 PsychoPy 的環境測試
(tests/test_grt_ada_modules.py)。

    item 編碼    colour = i & 1 ,  sound = (i >> 1) & 1
                 target ^ 1 = 只差顏色 , ^ 2 = 只差聲音 , ^ 3 = 兩者都差
    Levels       兩級顏色(弧長 dE00)+ 兩級 SNR(dB)—— 校準每試改、校準結束定案
    ColourAxis   弧長 -> hex 的查表(agrt_colour_lut.json)
    study_layout 四個項目擺哪個角落、哪個時間出現
    cue_layout   提示框在哪、裡面放什麼
    score        四選一的結果分類
    *_plan       校準 / 練習 / 主實驗的試次計畫
    MaskNoise    study 與 cue 之間的聽覺遮蔽噪音

⚠「聲音」那一位在**物理上是 SNR**，在**反應上是 b/p**。
  b/p(VOT)是連續值，但程式做不出連續體，所以用 SNR 當替代軸:
  item 0,1 = 高 SNR 的 /bi/(標籤 [bi]),item 2,3 = 低 SNR 的 /bi/(標籤 [pi])。
  資料欄位 target_snr / cued_snr / snr_* 記物理層(hi_snr / lo_snr),
  relation 與選項標籤用 sound / [bi] [pi],與 GRTv3.py 一致以便合併分析。
"""
import json
import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- item 編碼

REL_NAME = {0: 'valid', 1: 'colour_only', 2: 'sound_only', 3: 'both'}

SNR_NAMES  = ['hi_snr', 'lo_snr']   # 物理層標籤，與 GRTv3.py 同名
SND_LABELS = ['[bi]', '[pi]']       # 反應層標籤：高 SNR -> [bi],低 SNR -> [pi]
ITEM_SNR   = [0, 0, 1, 1]           # item 0,1 -> 高 SNR([bi]);item 2,3 -> 低 SNR([pi])


def colour_bit(item):
    return item & 1


def sound_bit(item):
    return (item >> 1) & 1


def make_item(cbit, sbit):
    return int(cbit) + 2 * int(sbit)


# ---------------------------------------------------------------- 版面與時間

_D = 10.0 / np.sqrt(2)              # 偏心度 5 度
POS = [( _D,  _D),                  # 0 = UR 右上
       (-_D,  _D),                  # 1 = UL 左上
       (-_D, -_D),                  # 2 = BL 左下
       ( _D, -_D)]                  # 3 = BR 右下
QUAD_NAMES   = ['UR', 'UL', 'BL', 'BR']
STUDY_ONSETS = [0.3, 1.3, 2.3, 3.3]
STUDY_DUR    = 1.0
OPT_GAP      = 4.2                  # 選項：標籤在上、色塊在下，兩者中心相距
POS_TXT = [(x, y + OPT_GAP / 2) for x, y in POS]
POS_COL = [(x, y - OPT_GAP / 2) for x, y in POS]

# 右手自然位置:i o 上排 = 上兩角,k l 下排 = 下兩角(原本是 g j / f h)
KEY_TO_SLOT = {'i': 'UL', 'o': 'UR', 'k': 'BL', 'l': 'BR'}


# ---------------------------------------------------------------- 色彩軸

class ColourAxis:
    """弧長座標(dE00)-> '#RRGGBB'。

    agrt_setup.py 需要 colour-science,而 PsychoPy 內建的 Python 沒有它 ——
    在實驗裡 import 會讓整個實驗開不起來。改用 export_lut() 離線產生的 JSON,
    執行期只靠 numpy 查表。要改色彩區域就重跑 agrt_setup.export_lut()。
    """

    def __init__(self, lut):
        self.lut = lut
        self.hexes = lut['hex']
        self.arc = lut['arc_min'] + np.arange(len(self.hexes)) * lut['step']
        self.arc_min = float(self.arc[0])
        self.arc_max = float(self.arc[-1])
        # 對稱可用半長(~24.13 dE00):Psi 的格點要對稱於錨點
        self.half_range = float(min(-self.arc_min, self.arc_max))

    @classmethod
    def load(cls, path='agrt_colour_lut.json'):
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def hex(self, arc):
        """超出可用範圍就報錯，不靜默夾擠。"""
        if not (self.arc_min <= arc <= self.arc_max):
            raise ValueError(f"colour arc length {arc} is outside the usable range "
                             f"[{self.arc_min:.2f}, {self.arc_max:.2f}] dE00")
        return self.hexes[int(np.abs(self.arc - arc).argmin())]

    def clip(self, arc):
        return float(np.clip(arc, self.arc_min, self.arc_max))


# ---------------------------------------------------------------- 刺激值

@dataclass
class Levels:
    """這一試的兩級顏色與兩級 SNR。

    colour_arc[0] 是 colour 位元 0(blue 端),[1] 是位元 1(pink 端)。
    snr_hi 是 sound 位元 0([bi] 端),snr_lo 是位元 1([pi] 端)。
    """
    colour_arc: tuple
    snr_hi: float
    snr_lo: float

    def snr_levels(self):
        return [self.snr_hi, self.snr_lo]       # index 0 = hi_snr([bi]), 1 = lo_snr([pi])

    def item_snr(self, item):
        return self.snr_levels()[ITEM_SNR[item]]

    def colour_hex(self, axis):
        return [axis.hex(self.colour_arc[0]), axis.hex(self.colour_arc[1])]

    def item_hex(self, axis):
        """四個 item 的顏色，item 順序:[C1, C2, C1, C2]。"""
        c = self.colour_hex(axis)
        return [c[colour_bit(i)] for i in range(4)]

    def item_labels(self):
        """四個選項的音節標籤。

        標籤看實際 SNR 的正負號，不看 index:主實驗是 [bi][bi][pi][pi];
        校顏色時兩級同為 +12 dB,四個都是 [bi]。
        ⚠ 這也表示：若校準後兩級落在 0 dB 同一側，選項會出現兩個相同的標籤。
        """
        return [SND_LABELS[0] if self.item_snr(i) > 0 else SND_LABELS[1] for i in range(4)]


DEFAULT_LEVELS = Levels(colour_arc=(-3.0, +3.0), snr_hi=+6.0, snr_lo=-6.0)   # 開機預設，校準會覆寫


# ---------------------------------------------------------------- 單一試次的版面

@dataclass
class StudyLayout:
    quad_content: list      # 象限 q 上放 item quad_content[q]
    time_perm: list         # 象限 q 在 STUDY_ONSETS[time_perm[q]] 出現
    target_quad: int

    def onset(self, quad):
        return STUDY_ONSETS[self.time_perm[quad]]


def study_layout(target_item, target_serial, rng):
    """隨機擺放四個 item,但強制 target 落在計畫指定的序列位置，其餘仍隨機。"""
    quad_content = [int(x) for x in rng.permutation(4)]
    target_quad = quad_content.index(target_item)
    time_perm = [int(x) for x in rng.permutation(4)]
    i = time_perm.index(target_serial)
    time_perm[i], time_perm[target_quad] = time_perm[target_quad], time_perm[i]
    return StudyLayout(quad_content, time_perm, target_quad)


def cue_layout(target_item, relation, layout):
    """提示框畫在 cued_item 學過的位置，框裡放的卻是 target_item —— 刻意製造的衝突。

    回傳 (cued_item, cue_quad)。invalid 試次上受試者報告 cued_item 即為 intrusion。
    """
    cued_item = target_item ^ relation      # relation = 0 -> cued == target (valid)
    return cued_item, layout.quad_content.index(cued_item)


def option_perm(rng):
    """每試次只洗「哪個 item 擺哪個位置」:回傳 [UR, UL, BL, BR] 各放哪個 item。"""
    return [int(x) for x in rng.permutation(4)]


def score(chosen_item, target_item, cued_item):
    """回傳 (outcome, err_rel)。

    順序不可調換:valid 試次的 cued_item == target_item,必須先被 correct 接走，
    否則所有答對的 valid 試次都會被誤標成 intrusion。
    err_rel:錯在哪個向度,1 = 只差顏色,2 = 只差聲音,3 = 兩者都差。
    """
    if chosen_item is None:
        return 'noresp', None
    if chosen_item == target_item:
        return 'correct', None
    err_rel = chosen_item ^ target_item
    if chosen_item == cued_item:
        return 'intrusion', err_rel
    return 'other', err_rel


# ---------------------------------------------------------------- 試次計畫
# 一個計畫列 = (cue_valid, target_item, relation, target_serial)

def main_plan(rng, n_valid_per_cell=24, n_invalid_per_cell=4, block_size=144):
    """主實驗的平衡計畫。回傳 (plan, n_blocks)。

    每個 (target_item x serial_pos) 有 n_valid_per_cell 個 valid,
    每個 (target_item x relation x serial_pos) 有 n_invalid_per_cell 個 invalid。
    連續輪流發牌到各 block、跨 stratum 不重置，所以每個 block 內 valid:invalid 都是 2:1。
    """
    strata = []
    for it in range(4):
        for sp in range(4):
            strata.append(((1, it, 0, sp), n_valid_per_cell))
            for rel in (1, 2, 3):
                strata.append(((0, it, rel, sp), n_invalid_per_cell))
    n_main = sum(n for _, n in strata)
    n_blocks = n_main // block_size
    blocks = [[] for _ in range(n_blocks)]
    k = 0
    for cond, n in strata:
        for _ in range(n):
            blocks[k % n_blocks].append(cond)
            k += 1
    for bi in range(n_blocks):              # block 內打散
        blocks[bi] = [blocks[bi][i] for i in rng.permutation(len(blocks[bi]))]
    return [t for b in blocks for t in b], n_blocks


def adapt_plan(n, rng):
    """校準計畫。全部 valid;target_item 要等 Psi 提出值的正負號才決定
    (grt_psi.PsiCalibration.propose),計畫裡先放 None,只平均分配 serial position。"""
    sp = [i % 4 for i in range(n)]
    return [(1, None, 0, int(sp[i])) for i in rng.permutation(n)]


def practice_plan(n, rng):
    """練習用與主實驗**相同的 valid:invalid 比例**(2:1)。若練習全是 valid,受試者會
    帶著「提示一定準」的預期進入正式階段，前幾個 invalid 試次的效果會異常大、
    之後才衰減 —— 那會讓平衡設計失效。"""
    n_valid = round(n * 2 / 3)
    n_invalid = n - n_valid
    valid_pool = [(1, it, 0, sp) for it in range(4) for sp in range(4)]
    inval_pool = [(0, it, rel, sp) for it in range(4) for rel in (1, 2, 3) for sp in range(4)]
    plan = ([valid_pool[i] for i in rng.permutation(len(valid_pool))[:n_valid]] +
            [inval_pool[i] for i in rng.permutation(len(inval_pool))[:n_invalid]])
    return [plan[i] for i in rng.permutation(len(plan))]


# ---------------------------------------------------------------- 聽覺遮蔽

class MaskNoise:
    """每試現做一段 dur 秒的純語音形狀噪音，寫成 wav。

    位準對齊「0 dB SNR 刺激裡、語音那一段的噪音 RMS」。
    ⚠ 不能直接用 snr_audio.TARGET_RMS 或 OUTPUT_RMS:mix_components 最後會把
      整段(語音 + 噪音，含前後引導噪音)正規化到 OUTPUT_RMS,所以受試者實際
      聽到的噪音位準是「縮放之後」的值，只能從混出來的成分量。0 dB 時
      語音段裡 noise RMS == speech RMS,所以這也就是「與語音 token 同 RMS」。
    輸出正規化的增益會隨噪音樣本微幅變動(實測 16 個種子 0.0385-0.0410,
    約 ±0.3 dB),所以用固定種子 0..15 的功率平均當參考 —— 確定性、可重現，
    每次開實驗都得到同一個數字。

    每試新種子(running noise,理由同 snr_audio.speech_shaped_noise)。
    重建:snr_audio.speech_shaped_noise(n_samp, np.random.default_rng(seed))
          * ref_rms * 10 ** (db / 20)
    """

    def __init__(self, token, outdir, dur=0.3, db=0.0, n_ref_seeds=16):
        import snr_audio
        self._sa = snr_audio
        self.outdir = outdir
        self.dur = dur
        self.db = float(db)
        v = []
        for s in range(n_ref_seeds):
            sr, _sp, nz, lead = snr_audio.mix_components(token, 0.0, np.random.default_rng(s))
            tail = int(round(snr_audio.NOISE_TAIL_MS * sr / 1000.0))
            v.append(float(np.mean(nz[lead:len(nz) - tail] ** 2)))
        self.sr = sr
        self.ref_rms = float(np.sqrt(np.mean(v)))
        self.n_samp = int(round(dur * sr))

    def make(self, trial_idx):
        """回傳 (seed, db, rms, path, summary);summary 的格式比照 snd_* 欄位。"""
        sa = self._sa
        seed = sa.new_seed()
        rms = self.ref_rms * 10.0 ** (self.db / 20.0)
        y = sa.speech_shaped_noise(self.n_samp, np.random.default_rng(seed)) * rms
        name = f"mask_{trial_idx:04d}_{self.db:+.2f}dB.wav"
        path = os.path.join(self.outdir, name)
        scale = sa.write_wav(path, self.sr, y)
        summary = f"ssn|{self.db:+.2f}dB|seed={seed}|{name}|scale={scale:.3f}"
        return seed, self.db, rms, path, summary
