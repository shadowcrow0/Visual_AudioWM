"""
聽覺基準線刺激產生：40 組 b3/p3 純比較對

設計：
  - 無顏色、無視覺、無工作記憶延遲
  - 單純問受試者「兩個音一樣還是不一樣」
  - 目的：測量「無工作記憶負荷」下 b3 vs p3 的知覺 JND

配置：
  - 20 個 b-p 對比組（b3 vs p3 with 不同時長差異)
  - 20 個 control 組（同個音重複，例如 b3-b3 或 p3-p3）
  - 總 40 組

每組由 3 個 talker 隨機抽選，產生 40 個 wav 檔案
"""

import subprocess
import os
import csv
import tempfile
import json
from pathlib import Path
import numpy as np


# ──────────────────────────────────────────────
# 基本設定（與 GetAudioStim.py 同步）
# ──────────────────────────────────────────────

VOICES = {
    "us1": {"path": "/usr/share/mbrola/us1/us1", "base_pitch": 180, "label": "F"},
    "us2": {"path": "/usr/share/mbrola/us2/us2", "base_pitch": 115, "label": "M1"},
    "us3": {"path": "/usr/share/mbrola/us3/us3", "base_pitch": 125, "label": "M2"},
}

TALKER_GRID = {
    "us1": [(15000, 0.85), (15500, 0.95), (16000, 1.00), (16000, 1.10), (16500, 1.05), (17000, 1.15)],
    "us2": [(15000, 0.88), (16000, 1.00), (17000, 1.12)],
    "us3": [(15500, 0.92), (16500, 1.05), (17000, 0.85)],
}

SPEED_FACTOR = 1.0
CONSONANT_FACTOR = 1.5
VOWEL_STRESS_FACTOR = 1.5

BASE_VOWEL_DUR = 250
VOWEL_DUR = int(BASE_VOWEL_DUR * SPEED_FACTOR * VOWEL_STRESS_FACTOR)
SILENCE_DUR = int(50 * SPEED_FACTOR)
VOWEL_SAMPA = "r="

_BASE_CONS_DUR = {
    'p': 80, 'b': 60,
}
CONS_DURATIONS = {k: int(v * SPEED_FACTOR * CONSONANT_FACTOR) for k, v in _BASE_CONS_DUR.items()}

# ──────────────────────────────────────────────
# Talker 生成
# ──────────────────────────────────────────────

def build_talkers():
    """產生所有 talker"""
    talkers = []
    idx = 0
    for voice_name, combos in TALKER_GRID.items():
        for vf, fr in combos:
            idx += 1
            talkers.append({
                'id': f'T{idx:02d}',
                'voice': voice_name,
                'voice_freq': vf,
                'pitch_ratio': fr,
            })
    return talkers


# ──────────────────────────────────────────────
# PHO 生成（支持動態子音時長因子）
# ──────────────────────────────────────────────

def make_pho_single(cons_sampa, base_pitch, cons_duration_factor=1.0, vowel_dur=None):
    """
    產生 /Cɜ/ 的 .pho

    Parameters
    ----------
    cons_sampa : str
        SAMPA 子音代碼('p' 或 'b')
    base_pitch : int
        基頻 Hz
    cons_duration_factor : float
        子音時長倍數(1.0 = 基準, 0.8 = 縮短 20%, 1.2 = 延長 20%)
    vowel_dur : int, optional
        母音長度 ms,預設使用 VOWEL_DUR
    """
    if vowel_dur is None:
        vowel_dur = VOWEL_DUR

    # 基準子音時長
    base_cons_dur = CONS_DURATIONS[cons_sampa]
    # 動態調整
    cons_dur = int(base_cons_dur * cons_duration_factor)

    def flat(dur):
        return f"(0,{base_pitch}) (100,{base_pitch})"

    cons_line = f"{cons_sampa} {cons_dur} {flat(cons_dur)}"

    return (
        f"_ {SILENCE_DUR}\n"
        f"{cons_line}\n"
        f"{VOWEL_SAMPA} {vowel_dur} {flat(vowel_dur)}\n"
        f"_ {SILENCE_DUR}\n"
    )


def synthesize_single(cons_sampa, talker, output_wav, cons_duration_factor=1.0):
    """
    用 MBROLA 合成單個 /Cɜ/

    Parameters
    ----------
    cons_sampa : str
        子音 ('p' 或 'b')
    talker : dict
        Talker 信息
    output_wav : str
        輸出 WAV 路徑
    cons_duration_factor : float
        子音時長倍數
    """
    cfg = VOICES[talker['voice']]
    pho_content = make_pho_single(cons_sampa, cfg['base_pitch'], cons_duration_factor)
    pho_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pho', delete=False).name

    try:
        with open(pho_file, "w") as f:
            f.write(pho_content)

        cmd = [
            "mbrola",
            "-l", str(talker['voice_freq']),
            "-f", str(talker['pitch_ratio']),
            cfg['path'],
            pho_file,
            output_wav,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    finally:
        os.unlink(pho_file)


# ──────────────────────────────────────────────
# 聽覺基準線刺激集
# ──────────────────────────────────────────────

class AuditoryBaselineStimuli:
    """
    40 組 b3/p3 聽覺比較刺激集

    結構：
      - 20 個「不同」對(b3 vs p3，時長差異範圍 5-30ms)
      - 20 個「相同」對(b3-b3 或 p3-p3)
    """

    def __init__(self, output_dir: str = "stimuli/auditory_baseline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.talkers = build_talkers()
        self.rng = np.random.default_rng(42)  # 固定種子，可重現

    def generate(self, n_trials: int = 40) -> str:
        """
        產生 40 組刺激並輸出 CSV

        Parameters
        ----------
        n_trials : int
            總刺激數（預設 40 = 20 不同對 + 20 相同對）

        Returns
        -------
        str
            輸出 CSV 路徑
        """
        n_different = n_trials // 2  # 20
        n_same = n_trials - n_different  # 20

        # 生成時長差異分佈（5–30ms）
        duration_diffs = self.rng.uniform(5, 30, n_different)

        # 生成隨機 talker 指派（每組獨立）
        talker_indices = self.rng.integers(0, len(self.talkers), n_trials)

        rows = []

        # ── 不同對 (b3 vs p3) ──
        for i in range(n_different):
            talker = self.talkers[talker_indices[i]]
            duration_diff = duration_diffs[i]

            # b3 和 p3 的時長因子
            # 基準：b3 子音 60ms, p3 子音 80ms，差 20ms
            # 要造出 duration_diff ms 的差異
            base_b3_dur = 60 * SPEED_FACTOR * CONSONANT_FACTOR
            base_p3_dur = 80 * SPEED_FACTOR * CONSONANT_FACTOR
            base_diff = base_p3_dur - base_b3_dur  # ~30ms

            scale = duration_diff / base_diff  # 縮放倍數
            b3_factor = 1.0 / scale
            p3_factor = 1.0 / scale

            # 合成
            b3_path = self.output_dir / f"trial_{i+1:03d}_b3_{talker['id']}.wav"
            p3_path = self.output_dir / f"trial_{i+1:03d}_p3_{talker['id']}.wav"

            success_b, _ = synthesize_single('b', talker, str(b3_path), b3_factor)
            success_p, _ = synthesize_single('p', talker, str(p3_path), p3_factor)

            if success_b and success_p:
                rows.append({
                    'trial_num': i + 1,
                    'trial_type': 'different',
                    'consonant1': 'b',
                    'consonant2': 'p',
                    'talker_id': talker['id'],
                    'duration_diff_ms': round(duration_diff, 1),
                    'file1': str(b3_path.relative_to(self.output_dir.parent)),
                    'file2': str(p3_path.relative_to(self.output_dir.parent)),
                })

        # ── 相同對 (control) ──
        for i in range(n_same):
            talker = self.talkers[talker_indices[n_different + i]]
            # 隨機選 b 或 p
            cons = self.rng.choice(['b', 'p'])

            wav1_path = self.output_dir / f"trial_{n_different + i + 1:03d}_{cons}3_a_{talker['id']}.wav"
            wav2_path = self.output_dir / f"trial_{n_different + i + 1:03d}_{cons}3_b_{talker['id']}.wav"

            # 兩個檔案相同內容(同子音、同時長)
            success1, _ = synthesize_single(cons, talker, str(wav1_path), 1.0)
            success2, _ = synthesize_single(cons, talker, str(wav2_path), 1.0)

            if success1 and success2:
                rows.append({
                    'trial_num': n_different + i + 1,
                    'trial_type': 'same',
                    'consonant1': cons,
                    'consonant2': cons,
                    'talker_id': talker['id'],
                    'duration_diff_ms': 0.0,
                    'file1': str(wav1_path.relative_to(self.output_dir.parent)),
                    'file2': str(wav2_path.relative_to(self.output_dir.parent)),
                })

        # ── 輸出 CSV ──
        csv_path = self.output_dir.parent / "auditory_baseline_trials.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ 已產生 {len(rows)} 個刺激")
        print(f"  - 不同對(b vs p): {n_different}")
        print(f"  - 相同對(control): {n_same}")
        print(f"  - 輸出 CSV: {csv_path}")

        return str(csv_path)


if __name__ == "__main__":
    print("=== 聽覺基準線刺激產生 ===\n")

    try:
        # 檢查 MBROLA voices
        for name, cfg in VOICES.items():
            if not os.path.exists(cfg['path']):
                print(f"✗ {name} 未安裝")
                print(f"  install: sudo apt-get install mbrola-{name}")
                exit(1)

        # 產生刺激
        baseline = AuditoryBaselineStimuli()
        csv_path = baseline.generate(n_trials=40)
        print(f"\n✓ 完成：{csv_path}")

    except Exception as e:
        print(f"✗ 錯誤: {e}")
        exit(1)
