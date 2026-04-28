"""
完整刺激產生 pipeline
- 18 talkers (3 base voices × 6 VTL/F0 combos)
- 所有 confusion_analysis.csv 裡的子音
- 輸出 PsychoPy conditions.csv

使用方式：
  sudo apt-get install mbrola mbrola-us1 mbrola-us2 mbrola-us3
  python generate_all_stimuli.py
"""

import subprocess
import os
import csv

# ══════════════════════════════════════════════
#  設定
# ══════════════════════════════════════════════

# MBROLA voice paths
VOICES = {
    "us1": {"path": "/usr/share/mbrola/us1/us1", "base_pitch": 180, "label": "F"},
    "us2": {"path": "/usr/share/mbrola/us2/us2", "base_pitch": 115, "label": "M1"},
    "us3": {"path": "/usr/share/mbrola/us3/us3", "base_pitch": 125, "label": "M2"},
}

# 18 talkers: 每個 base voice × 6 個 (VTL, F0_ratio) 組合
# VTL 範圍: 15000-17000
# F0 ratio: 拉開差異
TALKER_GRID = {
    "us1": [
        # (voice_freq, pitch_ratio)
        (15000, 0.85),
        (15500, 0.95),
        (16000, 1.00),
        (16000, 1.10),
        (16500, 1.05),
        (17000, 1.15),
    ],
    "us2": [
        # 3 talkers — VTL 分散，F0 拉開
        (15000, 0.88),
        (16000, 1.00),
        (17000, 1.12),
    ],
    "us3": [
        # 3 talkers — 跟 us2 錯開，避免重疊
        (15500, 0.92),
        (16500, 1.05),
        (17000, 0.85),
    ],
}

# 子音名稱 → MBROLA SAMPA
CONS_SAMPA = {
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd',
    'k': 'k', 'g': 'g', 'f': 'f', 'v': 'v',
    'theta': 'T', 'eth': 'D',
    's': 's', 'z': 'z', 'sh': 'S', 'zh': 'Z',
    'tch': 'tS', 'dj': 'dZ',
    'm': 'm', 'n': 'n',
    'l': 'l', 'r': 'r',
    'y': 'j', 'w': 'w'
}

# ── 速度控制 ──
SPEED_FACTOR = .85  # 1.0 = 原速, >1.0 = 較慢, <1.0 = 較快

# 基礎子音長度 (ms)，會乘上 SPEED_FACTOR
_BASE_CONS_DUR = {
    'p': 80, 'b': 60, 't': 80, 'd': 60, 'k': 80, 'g': 60,
    'f': 130, 'v': 100, 'T': 130, 'D': 100,
    's': 140, 'z': 110, 'S': 140, 'Z': 110,
    'tS': 120, 'dZ': 100,
    'm': 90, 'n': 90,
    'l': 90, 'r': 90, 'j': 80, 'w': 80,
}
CONS_DURATIONS = {k: int(v * SPEED_FACTOR) for k, v in _BASE_CONS_DUR.items()}

# 基礎母音長度 (ms)
BASE_VOWEL_DUR = 250
VOWEL_DUR = int(BASE_VOWEL_DUR * SPEED_FACTOR)  # 375ms at 1.5x

# 前後靜音 (ms)
SILENCE_DUR = int(50 * SPEED_FACTOR)  # 75ms at 1.5x


# ══════════════════════════════════════════════
#  建立 talker list
# ══════════════════════════════════════════════

def build_talkers():
    """產生 18 個 talker 的定義"""
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
                'label': f"{VOICES[voice_name]['label']}_vf{vf}_fr{fr:.2f}",
            })
    return talkers


# ══════════════════════════════════════════════
#  PHO 產生 + 合成
# ══════════════════════════════════════════════

def make_pho(cons_sampa, base_pitch, vowel_dur=None):
    """產生 /aCa/ 的 .pho，平坦 F0 + gentle declination

    Args:
        cons_sampa: MBROLA SAMPA 格式的子音 (e.g., 'p', 'T', 'S')
        base_pitch: 基頻 (Hz)
        vowel_dur: 母音長度 (ms)，預設使用 VOWEL_DUR
    """
    if vowel_dur is None:
        vowel_dur = VOWEL_DUR
    cons_dur = CONS_DURATIONS.get(cons_sampa, int(100 * SPEED_FACTOR))

    # 平坦微降 F0：模擬自然 declination
    p1 = base_pitch
    p2 = int(base_pitch * 0.93)
    # 第二個母音略短（final shortening ~10%）
    vowel_dur2 = int(vowel_dur * 0.9)

    if cons_sampa in ('tS', 'dZ'):
        cons_line = f"{cons_sampa[0]} {cons_dur // 2}\n{cons_sampa[1]} {cons_dur // 2}"
    else:
        cons_line = f"{cons_sampa} {cons_dur}"

    return (
        f"_ {SILENCE_DUR}\n"
        f"A {vowel_dur} (50, {p1})\n"
        f"{cons_line}\n"
        f"A {vowel_dur2} (50, {p2})\n"
        f"_ {SILENCE_DUR}\n"
    )


def synthesize(cons_sampa, talker, output_wav):
    """用 MBROLA 合成一個 /aCa/"""
    cfg = VOICES[talker['voice']]
    pho_content = make_pho(cons_sampa, cfg['base_pitch'])
    pho_file = "/tmp/_gen_stim.pho"

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


# ══════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════

def main():
    # 1. 檢查 voices
    print("Checking MBROLA voices...")
    for name, cfg in VOICES.items():
        if os.path.exists(cfg['path']):
            print(f"  ✓ {name} ({cfg['label']})")
        else:
            print(f"  ✗ {name} — install: sudo apt-get install mbrola-{name}")
            return

    # 2. 讀 CSV
    csv_path = "confuse.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "confuse.csv")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    print(f"\nRead {len(rows)} rows from CSV")

    # 收集所有需要的子音
    all_cons = set()
    for row in rows:
        all_cons.add(row['sound'])
        all_cons.add(row['target'])
    all_cons = sorted(all_cons)
    print(f"Consonants to generate: {all_cons}")

    # 3. 建立 talkers
    talkers = build_talkers()
    print(f"\nTalkers: {len(talkers)}")
    for t in talkers:
        print(f"  {t['id']}: {t['voice']} vf={t['voice_freq']} fr={t['pitch_ratio']:.2f}")

    # 4. 產生所有 wav（扁平結構：T01_apa.wav）
    wav_dir = "stimuli"
    os.makedirs(wav_dir, exist_ok=True)
    total_files = len(all_cons) * len(talkers)
    print(f"\nGenerating {len(all_cons)} consonants × {len(talkers)} talkers = {total_files} wav files...")

    success = 0
    fail = 0

    for talker in talkers:
        tid = talker['id']
        for cons_name in all_cons:
            if cons_name not in CONS_SAMPA:
                print(f"  SKIP: {cons_name}")
                continue

            sampa = CONS_SAMPA[cons_name]
            wav_path = os.path.join(wav_dir, f"{tid}_a{cons_name}a.wav")

            ok, err = synthesize(sampa, talker, wav_path)
            if ok:
                success += 1
            else:
                fail += 1
                print(f"  ✗ {tid}_a{cons_name}a: {err}")

    print(f"\nGenerated: {success} OK, {fail} failed")

    # 5. 產生 PsychoPy conditions.csv
    # 每個 CSV row × 每個 talker = 一個 trial
    conditions_path = "stimuli/conditions.csv"
    fieldnames = [
        'sound', 'target', 'confused', 'count',
        'sound_file', 'target_file',
        'talker', 'voice_base', 'condition',
    ]

    trial_count = 0
    with open(conditions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            sound = row['sound']
            target = row['target']
            confused = int(row['confused'])
            count = int(row['count'])
            condition = "confusable" if confused == 0 else "distinct"

            for talker in talkers:
                tid = talker['id']
                sound_file = f"{tid}_a{sound}a.wav"
                target_file = f"{tid}_a{target}a.wav"

                writer.writerow({
                    'sound': sound,
                    'target': target,
                    'confused': confused,
                    'count': count,
                    'sound_file': sound_file,
                    'target_file': target_file,
                    'talker': tid,
                    'voice_base': talker['voice'],
                    'condition': condition,
                })
                trial_count += 1

    print(f"Conditions file: {conditions_path}")
    print(f"Total trials: {trial_count}")
    print(f"  = {len(rows)} pairs × {len(talkers)} talkers")

    # 6. 印出 talker 對照表
    talker_info_path = "stimuli/talker_info.csv"
    with open(talker_info_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['talker_id', 'base_voice', 'voice_freq', 'pitch_ratio'])
        writer.writeheader()
        for t in talkers:
            writer.writerow({
                'talker_id': t['id'],
                'base_voice': t['voice'],
                'voice_freq': t['voice_freq'],
                'pitch_ratio': t['pitch_ratio'],
            })
    print(f"Talker info: {talker_info_path}")

    # 7. PsychoPy 說明
    print(f"""
{'=' * 50}
PsychoPy 設定：

1. 把 stimuli/ 資料夾放進實驗目錄

2. Loop 設定：
   - conditions file: stimuli/conditions.csv
   - 如果 1728 trials 太多，可以：
     a. 用 nReps=1 但只選部分 talker
        （在 conditions.csv 裡 filter by talker）
     b. 或用 random sampling: Selected rows = 隨機取子集

3. Trial routine：
   - Sound component: $sound_file
   - 受試者做 identification: 聽到什麼子音？
   - Response: keyboard 或 button box

4. 分析時用 condition 欄位區分 confusable vs distinct

目錄結構（扁平格式）：
  stimuli/
    conditions.csv      ← PsychoPy loop 讀這個
    talker_info.csv     ← talker 參數對照表
    T01_apa.wav         ← 扁平檔名：{talker}_{sound}.wav
    T01_aba.wav
    T02_apa.wav
    ...
""")


if __name__ == "__main__":
    main()