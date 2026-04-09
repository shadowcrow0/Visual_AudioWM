"""
Pilot test: 用 MBROLA 測試不同 VTL × F0 組合產生 /aCa/ 刺激

產出結構：
  pilot_stimuli/
    test1_vtl_sweep/{voice}/{consonant}_vf{vtl}.wav
    test2_f0_sweep/{voice}/{consonant}_fr{ratio}.wav
    test3_grid/{voice}/{consonant}_vf{vtl}_fr{ratio}.wav

使用方式：
  1. 確認已安裝 mbrola, mbrola-us1, mbrola-us2, mbrola-us3
     sudo apt-get install mbrola mbrola-us1 mbrola-us2 mbrola-us3
  2. python check_available.py
  3. 用任何播放器聽 pilot_stimuli/ 裡的 wav 檔
"""

import subprocess
import os

# ── Miller & Nicely 1955 的 16 個子音 (MBROLA SAMPA 格式) ──
CONSONANTS = {
    # 塞音 (stops)
    'p': 'p',    # voiceless bilabial
    'b': 'b',    # voiced bilabial
    't': 't',    # voiceless alveolar
    'd': 'd',    # voiced alveolar
    'k': 'k',    # voiceless velar
    'g': 'g',    # voiced velar
    # 擦音 (fricatives)
    'f': 'f',    # voiceless labiodental
    'v': 'v',    # voiced labiodental
    'theta': 'T',  # θ voiceless dental (think)
    'eth': 'D',    # ð voiced dental (this)
    's': 's',    # voiceless alveolar
    'z': 'z',    # voiced alveolar
    'sh': 'S',   # ʃ voiceless postalveolar (ship)
    'zh': 'Z',   # ʒ voiced postalveolar (measure)
    # 鼻音 (nasals)
    'm': 'm',
    'n': 'n',
}

# 每個 base voice 的 default pitch（Hz）
VOICE_CONFIG = {
    "us1": {"path": "/usr/share/mbrola/us1/us1", "base_pitch": 180, "label": "Female"},
    "us2": {"path": "/usr/share/mbrola/us2/us2", "base_pitch": 115, "label": "Male 1"},
    "us3": {"path": "/usr/share/mbrola/us3/us3", "base_pitch": 125, "label": "Male 2"},
}

# VTL manipulation: voice_freq 參數
# 數值越低 → 聲道越長 → formants 越低（大體型）
# 數值越高 → 聲道越短 → formants 越高（小體型）
# 16000 = 原始（us1 的 sample rate）
VTL_VALUES = [13000, 14000, 15000, 16000, 17000, 18000, 19000]

# F0 manipulation: pitch ratio
# < 1.0 → 更低沉
# > 1.0 → 更高亢
F0_RATIOS = [0.75, 0.85, 1.0, 1.15, 1.30]

OUT_DIR = "pilot_stimuli"

# ── 速度控制 ──
SPEED_FACTOR = 1.5  # 1.0 = 原速, >1.0 = 較慢, <1.0 = 較快

# ── PHO 產生 ──
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

def make_pho(cons_sampa, base_pitch, vowel_dur=None):
    """產生 /aCa/ 的 .pho 內容

    Args:
        cons_sampa: MBROLA SAMPA 格式的子音 (e.g., 'p', 'T', 'S')
        base_pitch: 基頻 (Hz)
        vowel_dur: 母音長度 (ms)，預設使用 VOWEL_DUR
    """
    if vowel_dur is None:
        vowel_dur = VOWEL_DUR
    cons_dur = CONS_DURATIONS.get(cons_sampa, int(100 * SPEED_FACTOR))
    # 平坦微降 F0：模擬自然 declination
    # 第一個母音：平坦
    p1 = base_pitch
    # 子音後第二個母音：降約 5-8%
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





# ── 合成 ──

def synthesize(voice_name, cons_name, cons_sampa, voice_freq, pitch_ratio, output_wav):
    """合成單一 /aCa/ 刺激

    Args:
        voice_name: MBROLA voice name (e.g., 'us1')
        cons_name: 子音名稱 (用於檔名, e.g., 'theta')
        cons_sampa: MBROLA SAMPA 格式 (e.g., 'T')
        voice_freq: VTL 參數
        pitch_ratio: F0 ratio
        output_wav: 輸出檔案路徑
    """
    cfg = VOICE_CONFIG[voice_name]
    pho_content = make_pho(cons_sampa, cfg["base_pitch"])
    pho_file = "/tmp/_pilot_test.pho"

    with open(pho_file, "w") as f:
        f.write(pho_content)

    cmd = [
        "mbrola",
        "-t", "1",
        "-l", str(voice_freq),
        "-f", str(pitch_ratio),
        cfg["path"],
        pho_file,
        output_wav,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()


# ── 主程式 ──

def main():
    # 檢查哪些 voice 可用
    available_voices = {}
    for name, cfg in VOICE_CONFIG.items():
        if os.path.exists(cfg["path"]):
            available_voices[name] = cfg
            print(f"✓ {name} ({cfg['label']}) found")
        else:
            print(f"✗ {name} ({cfg['label']}) NOT found — skip")
            print(f"  Install: sudo apt-get install mbrola-{name}")

    if not available_voices:
        print("\nNo voices available. Install at least one.")
        return

    # 只用一個子音做 VTL/F0 測試
    test_consonant = ('p', 'p')  # (name, sampa)

    # === Test 1: 固定 F0，只變 VTL ===
    print("\n" + "=" * 60)
    print("TEST 1: VTL sweep (固定 F0 = 1.0，只改聲道長度)")
    print("  聽看看 formant 高低是否讓它們像不同的人")
    print("=" * 60)

    for voice_name, cfg in available_voices.items():
        voice_dir = os.path.join(OUT_DIR, f"test1_vtl_sweep/{voice_name}")
        os.makedirs(voice_dir, exist_ok=True)

        for vf in VTL_VALUES:
            cons_name, cons_sampa = test_consonant
            wav = os.path.join(voice_dir, f"a{cons_name}a_vf{vf}.wav")
            ok, err = synthesize(voice_name, cons_name, cons_sampa, vf, 1.0, wav)
            status = "✓" if ok else f"✗ {err}"
            print(f"  {voice_name} vf={vf:5d} → {status}")

    # === Test 2: 固定 VTL，只變 F0 ===
    print("\n" + "=" * 60)
    print("TEST 2: F0 sweep (固定 VTL = 16000，只改基頻)")
    print("  聽看看純粹改 pitch 是否像不同的人")
    print("=" * 60)

    for voice_name, cfg in available_voices.items():
        voice_dir = os.path.join(OUT_DIR, f"test2_f0_sweep/{voice_name}")
        os.makedirs(voice_dir, exist_ok=True)

        for fr in F0_RATIOS:
            cons_name, cons_sampa = test_consonant
            wav = os.path.join(voice_dir, f"a{cons_name}a_fr{fr:.2f}.wav")
            ok, err = synthesize(voice_name, cons_name, cons_sampa, 16000, fr, wav)
            status = "✓" if ok else f"✗ {err}"
            print(f"  {voice_name} fr={fr:.2f} → {status}")

    # === Test 3: 所有 16 子音 (固定 VTL=16000, F0=1.0) ===
    print("\n" + "=" * 60)
    print("TEST 3: 所有 16 個 Miller & Nicely 子音")
    print("  聽看看每個子音是否正確")
    print("=" * 60)

    for voice_name, cfg in available_voices.items():
        voice_dir = os.path.join(OUT_DIR, f"test3_all_consonants/{voice_name}")
        os.makedirs(voice_dir, exist_ok=True)

        for cons_name, cons_sampa in CONSONANTS.items():
            wav = os.path.join(voice_dir, f"a{cons_name}a.wav")
            ok, err = synthesize(voice_name, cons_name, cons_sampa, 16000, 1.0, wav)
            status = "✓" if ok else f"✗ {err}"
            print(f"  {voice_name} /{cons_name}/ ({cons_sampa}) → {status}")

    # === Test 4: VTL × F0 grid（核心測試，用 /p/）===
    print("\n" + "=" * 60)
    print("TEST 4: VTL × F0 grid (兩個維度同時變)")
    print("  這才是最終會用的方式")
    print("=" * 60)

    # 只用較少的組合，避免產生太多檔案
    vtl_subset = [14000, 16000, 18000]
    f0_subset = [0.85, 1.0, 1.15]

    for voice_name, cfg in available_voices.items():
        voice_dir = os.path.join(OUT_DIR, f"test4_grid/{voice_name}")
        os.makedirs(voice_dir, exist_ok=True)

        cons_name, cons_sampa = test_consonant
        for vf in vtl_subset:
            for fr in f0_subset:
                wav = os.path.join(voice_dir, f"a{cons_name}a_vf{vf}_fr{fr:.2f}.wav")
                ok, err = synthesize(voice_name, cons_name, cons_sampa, vf, fr, wav)
                status = "✓" if ok else f"✗ {err}"
                print(f"  {voice_name} vf={vf:5d} fr={fr:.2f} → {status}")

    # === 總結 ===
    total = 0
    for root, dirs, files in os.walk(OUT_DIR):
        total += len([f for f in files if f.endswith(".wav")])

    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total} wav files in {OUT_DIR}/")
    print(f"\n聽的時候注意：")
    print(f"  Test 1: 同一排裡，vf 從低到高 → formants 應從低變高")
    print(f"  Test 2: 同一排裡，fr 從低到高 → pitch 應從低變高")
    print(f"  Test 3: 檢查每個子音是否正確發音")
    print(f"  Test 4: 每個組合應該聽起來像不同的人")
    print(f"\n如果 WSL，複製到 Windows 側聽：")
    print(f"  cp -r {OUT_DIR} /mnt/c/Users/spt904/Desktop/")


if __name__ == "__main__":
    main()
