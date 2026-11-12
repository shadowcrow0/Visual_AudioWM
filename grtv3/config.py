"""GRTv3_ada 的常數。數值與 GRTv3_ada_EN.py 相同；要改任何數字先看 DECISIONS.md 與 CLAUDE.md。"""
import numpy as np

EXP_NAME = "GRTv3_ada"
PSYCHOPY_VERSION_NOTE = "written for PsychoPy 2026.x"

# ── 試次計畫（GRTv3_ada_EN.py:434-479）──────────────────────────────────────
N_VALID_PER_CELL = 24      # 每個 (target_item, target_serial) 格的 valid 試次
N_INVALID_PER_CELL = 4     # 每個 (target_item, relation, target_serial) 格的 invalid 試次 → valid:invalid = 2:1
BLOCK_SIZE = 144
CALIB_JOINT = 0            # 0 = 兩個區塊分開校（先顏色 N_ADAPT_COL 試，再聲音 N_ADAPT_SND 試；GRTv3_ada 原設計）
                           # 1 = Glavan 的 joint：每一試同時提議兩維（AGRTHandler.__next__ / addResponse），共 N_ADAPT_JOINT 試
N_ADAPT_COL = 72           # 校準區塊 1：顏色（CALIB_JOINT = 0）
N_ADAPT_SND = 72           # 校準區塊 2：聲音（CALIB_JOINT = 0）
N_ADAPT_JOINT = 72         # joint 模式的校準試次數（CALIB_JOINT = 1）
N_PRACTICE = 15
FEEDBACK_PRE_MAIN = True   # 校準與練習後框出正確選項
FEEDBACK_SEC = 1.5
REL_NAME = {0: "valid", 1: "colour_only", 2: "sound_only", 3: "both"}

# ── 時序（秒）──────────────────────────────────────────────────────────────
FIX_DUR = 0.3
STUDY_START = [0.3, 1.3, 2.3, 3.3]   # 四個項目的起始時間（順序每試重排）
STUDY_DUR = 1.0
MASK_DUR = 0.3
CUE_DUR = 2.0                        # 0–1 s 空白框，1–2 s 框內出現內容
CUE_CONTENT_AT = 1.0
POST_FEEDBACK_BLANK = 0.3

# ── 幾何（deg）────────────────────────────────────────────────────────────
_d = 10.0 / np.sqrt(2)
POS = [(_d, _d), (-_d, _d), (-_d, -_d), (_d, -_d)]     # 0=UR 1=UL 2=BL 3=BR
SLOT_NAMES = ["UR", "UL", "BL", "BR"]
SQUARE = 4.0
TASK_GAP = 4.2                                          # 選項畫面：文字在上、色塊在下，相距 GAP
BOX_SIZE = 6.0
KEY_TO_SLOT = {"i": "UL", "o": "UR", "k": "BL", "l": "BR"}
RESPONSE_KEYS = list(KEY_TO_SLOT)

# ── 聲音（GRTv3_ada_EN.py:870-889）────────────────────────────────────────
SND_TOKEN = "bi"                    # snr_audio.SPEECH_FILES 現在只有 'bi'（8a75be2 改指向自然 [bi] token）；原腳本仍寫 'be'，在這個 repo 會 KeyError
SNR_LIM = 18.0
SNR_NAMES = ["hi_snr", "lo_snr"]
SND_LABELS = ["[bi]", "[pi]"]
ITEM_SNR = [0, 0, 1, 1]              # item = colour_bit + 2*sound_bit；sound bit 0 = hi_snr([bi])
ADAPT_FIX_ARC = 0.0                  # 校聲音時四格同一色
ADAPT_FIX_SNR = 12.0                 # 校顏色時四格同一個清楚的 /bi/
SPEAKER_LABEL = "Laptop"             # audio_pick.register_label 把真實裝置掛在這個標籤下

# ── 遮蔽 ──────────────────────────────────────────────────────────────────
MASK_VISUAL = True
MASK_AUDIO = True
MASK_NOISE_DB = 0.0                  # 相對 0 dB SNR 噪音參考 RMS 的偏移
MASK_TEX_N = 32
MASK_SIZE = 4.0

# ── Psi / AGRT（GRTv3_ada_EN.py:866-868, 964-968）─────────────────────────
OVERALL_ACC = 0.64
LAPSE = 0.08
PSI_STEPS = 100

# ── 視窗 ──────────────────────────────────────────────────────────────────
WIN_SIZE = (1024, 768)
WIN_FULLSCR = True
WIN_UNITS = "deg"
MONITOR = "testMonitor"
FRAME_TOLERANCE = 0.001
