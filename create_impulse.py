import numpy as np
from scipy.signal import fftconvolve
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

# 訊號
sr = 48000
def make_harmonic_burst(f0=2000, n_harmonics=20, duration_ms=200):
    samples = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms/1000, samples)
    signal = np.zeros(samples)
    for n in range(1, n_harmonics + 1):
        if f0 * n > sr / 2:
            break
        signal += np.sin(2 * np.pi * f0 * n * t)
    fade_samples = int(sr * 0.005)
    envelope = np.ones(samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    signal = signal * envelope
    return (signal / np.max(np.abs(signal))).astype(np.float32)

signal = make_harmonic_burst()
signal_44k = librosa.resample(signal, orig_sr=48000, target_sr=44100)

# 9x9 格子
# x 軸：9 個 azimuth（左到右）
# y 軸：9 個距離（近到遠）
az_values  = [-45, -35, -20, -10, 0, 10, 20, 35, 45]   # 左到右
dist_values = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]          # 近到遠（公尺）

# 計算基準音量（距離 1m 的音量）
ref_dist = 1.0

output_dir = '/mnt/c/Users/spt904/Desktop/stimuli/'
os.makedirs(output_dir, exist_ok=True)

n = 1
for dist in dist_values:
    for az in az_values:
        # 取 HRTF
        az_idx = cipic_az.index(az)
        el_idx = 8  # el=0 水平面

        hrir_l = data['hrir_l'][az_idx, el_idx, :]
        hrir_r = data['hrir_r'][az_idx, el_idx, :]

        left  = fftconvolve(signal_44k, hrir_l)
        right = fftconvolve(signal_44k, hrir_r)

        min_len = min(len(left), len(right))
        stereo = np.stack([left[:min_len], right[:min_len]], axis=1)

        # 距離衰減（反平方定律）
        gain = ref_dist / dist
        stereo = stereo * gain
        stereo = np.clip(stereo, -1.0, 1.0)

        sf.write(f'{output_dir}stimulus_{n}.wav', stereo.astype(np.float32), sr_cipic)
        print(f"生成 stimulus_{n}.wav  az={az}°  dist={dist}m")
        n += 1