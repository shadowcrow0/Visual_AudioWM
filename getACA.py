import subprocess
import os

consonants = {
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd',
    'k': 'k', 'g': 'g', 'f': 'f', 'v': 'v',
    'theta': 'T', 'eth': 'D',
    's': 's', 'z': 'z', 'sh': 'S', 'zh': 'Z',
    'tch': 'tS', 'dj': 'dZ',
    'm': 'm', 'n': 'n',
    'l': 'l', 'r': 'r',
    'y': 'j', 'w': 'w'
}

voice_path = '/usr/share/mbrola/us3/us3'  # 替換為你的 MBROLA 聲音檔路徑

for name, sampa in consonants.items():
    # 處理 affricates（tS, dZ 拆成兩個音素）
    if sampa in ('tS', 'dZ'):
        pho = f"_ 30\nA 200 (50,120)\n{sampa[0]} 40\n{sampa[1]} 40\nA 200 (50,110)\n_ 30\n"
    else:
        pho = f"_ 30\nA 200 (50,120)\n{sampa} 80\nA 200 (50,110)\n_ 30\n"

    pho_file = '/tmp/test_diphone.pho'
    wav_file = f'/tmp/test_{name}.wav'

    with open(pho_file, 'w') as f:
        f.write(pho)

    result = subprocess.run(
        ['mbrola', voice_path, pho_file, wav_file],
        capture_output=True, text=True
    )

    if result.returncode == 0 and not result.stderr:
        print(f"✓ {name:8s} ({sampa:3s}): OK")
    else:
        err = result.stderr.strip().split('\n')[0] if result.stderr else 'unknown error'
        print(f"✗ {name:8s} ({sampa:3s}): {err}")