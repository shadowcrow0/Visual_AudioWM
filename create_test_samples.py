"""
快速生成測試樣本：比較距離和仰角差異
"""
import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
import scipy.io as sio
import soundfile as sf
import librosa
import os

# 載入 CIPIC
data = sio.loadmat('/mnt/c/Users/spt904/Desktop/hrir_final.mat')
sr_cipic = 44100

cipic_az = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
            -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 55, 65, 80]

sr = 48000
def make_pink_noise(duration_ms=500):
    """生成 pink noise (1/f noise)"""
    samples = int(sr * duration_ms / 1000)

    # 生成白噪音
    white = np.random.randn(samples)

    # FFT 轉換
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(samples, 1/sr)

    # 1/f 濾波 (避免除以零)
    freqs[0] = 1
    fft = fft / np.sqrt(freqs)

    # 轉回時域
    signal = np.fft.irfft(fft, samples)

    # Fade in/out
    fade_samples = int(sr * 0.01)
    envelope = np.ones(samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    signal = signal * envelope
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def apply_air_absorption(signal, distance, sr=44100):
    if distance <= 1.0:
        return signal
    cutoff = max(2000, 20000 / (1 + 0.1 * (distance - 1)))
    b, a = butter(1, cutoff / (sr / 2), btype='low')
    return lfilter(b, a, signal)
np.random.seed(42) 
signal = make_pink_noise()
signal_44k = librosa.resample(signal, orig_sr=48000, target_sr=44100)

ref_dist = 1.0
output_dir = '/mnt/c/Users/spt904/Desktop/stimuli/test_samples/'
os.makedirs(output_dir, exist_ok=True)

# 測試組合
test_cases = [
    # 距離比較 (固定 az=0°, el=0°)
    {'name': 'dist_0m_az0_el0',   'az': 0, 'el_idx': 8,  'el_deg': 0,   'dist': 0.5},
    {'name': 'dist_2m_az0_el0',     'az': 0, 'el_idx': 8,  'el_deg': 0,   'dist': 2},
    {'name': 'dist_4m_az0_el0',    'az': 0, 'el_idx': 8,  'el_deg': 0,   'dist': 4},
    {'name': 'dist_16m_az0_el0',    'az': 0, 'el_idx': 8,  'el_deg': 0,   'dist': 16},

    # 仰角比較 (固定 az=0°, dist=2m)
    {'name': 'el_-45_az0_dist2m',   'az': 0, 'el_idx': 0,  'el_deg': -45, 'dist': 2},
    {'name': 'el_0_az0_dist4m',     'az': 0, 'el_idx': 8,  'el_deg': 0,   'dist': 4},
    {'name': 'el_+45_az0_dist2m',   'az': 0, 'el_idx': 16, 'el_deg': 45,  'dist': 2},

    # 方位角比較 (固定 el=0°, dist=2m)
    {'name': 'az_-45_el0_dist6m',   'az': -45, 'el_idx': 8, 'el_deg': 0, 'dist': 6},
    {'name': 'az_0_el0_dist4m',     'az': 0,   'el_idx': 8, 'el_deg': 0, 'dist': 4},
    {'name': 'az_+45_el0_dist6m',   'az': 45,  'el_idx': 8, 'el_deg': 0, 'dist': 6},
]

print("生成測試樣本...")
print("=" * 50)

for tc in test_cases:
    az_idx = cipic_az.index(tc['az'])
    el_idx = tc['el_idx']
    dist = tc['dist']

    hrir_l = data['hrir_l'][az_idx, el_idx, :]
    hrir_r = data['hrir_r'][az_idx, el_idx, :]

    left  = fftconvolve(signal_44k, hrir_l)
    right = fftconvolve(signal_44k, hrir_r)

    left  = apply_air_absorption(left, dist, sr_cipic)
    right = apply_air_absorption(right, dist, sr_cipic)

    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)

    # 距離衰減
    gain = ref_dist / (dist ** 0.8)
    stereo = stereo * gain
    stereo = np.clip(stereo, -1.0, 1.0)



    filename = f"{output_dir}{tc['name']}.wav"
    sf.write(filename, stereo.astype(np.float32), sr_cipic)
    print(f"✓ {tc['name']}.wav  (az={tc['az']}°, el={tc['el_deg']}°, dist={tc['dist']}m)")

print("=" * 50)
print(f"\n測試樣本已儲存至: {output_dir}")
print("\n建議比較順序:")
print("1. 距離: dist_0.5m → dist_5m → dist_10m → dist_20m")
print("2. 仰角: el_-45 (下) → el_0 (水平) → el_+45 (上)")
print("3. 方位: az_-45 (左) → az_0 (正前) → az_+45 (右)")
