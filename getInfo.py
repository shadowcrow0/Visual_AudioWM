import scipy.io as sio
import numpy as np

# 載入
data = sio.loadmat('/mnt/c/Users/spt904/Desktop/hrir_final.mat')

# 看裡面有什麼
#print(data.keys())


print("hrir_l shape:", data['hrir_l'].shape)
print("hrir_r shape:", data['hrir_r'].shape)

# CIPIC 的 azimuth 和 elevation 是固定的
cipic_az = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, 
            -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 
            40, 45, 55, 65, 80]

cipic_el = [-45 + i * 5.625 for i in range(50)]

print("azimuth:", cipic_az)
print("elevation 前幾個:", cipic_el[:5])
print("elevation 後幾個:", cipic_el[-5:])

import scipy.io as sio
from scipy.signal import fftconvolve
import soundfile as sf
import numpy as np

data = sio.loadmat('/mnt/c/Users/spt904/Desktop/hrir_final.mat')
sr_cipic = 44100

# 用你現有的 signal，但要 resample 成 44100
import librosa
sr = 48000
def make_harmonic_burst(f0=400, n_harmonics=20, duration_ms=1000):
    samples = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms/1000, samples)
    signal = np.zeros(samples)
    for n in range(1, n_harmonics + 1):
        if f0 * n > sr / 2:
            break
        signal += np.sin(2 * np.pi * f0 * n * t)
    
    # 只在頭尾 5ms 做短暫 fade，中間維持固定音量
    fade_samples = int(sr * 0.005)  # 5ms
    envelope = np.ones(samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    signal = signal * envelope
    return (signal / np.max(np.abs(signal))).astype(np.float32)
signal = make_harmonic_burst(f0=400, n_harmonics=20, duration_ms=1000)

signal_44k = librosa.resample(signal, orig_sr=48000, target_sr=44100)
test_positions = [
    (-45, 45,  'top_left'),    # 左上
    (0,   0,   'center'),      # 正中間
    (45, -45,  'bottom_right') # 右下
]

for az, el, label in test_positions:
    az_idx = cipic_az.index(az)
    el_idx = round((el + 45) / 5.625)
    
    hrir_l = data['hrir_l'][az_idx, el_idx, :]
    hrir_r = data['hrir_r'][az_idx, el_idx, :]
    
    left  = fftconvolve(signal_44k, hrir_l)
    right = fftconvolve(signal_44k, hrir_r)
    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    stereo = stereo / np.max(np.abs(stereo)) * 0.8
    sf.write(f'/mnt/c/Users/spt904/Desktop/cipic_{label}.wav', stereo.astype(np.float32), sr_cipic)
    print(f"生成 cipic_{label}.wav")
#for el_idx, label in [(0, 'down_-45'), (8, 'mid_0'), (16, 'up_45')]:
    # hrir_l = data['hrir_l'][12, el_idx, :]  # az index 12 = 0° 正前方
    # hrir_r = data['hrir_r'][12, el_idx, :]
    # left  = fftconvolve(signal_44k, hrir_l)
    # right = fftconvolve(signal_44k, hrir_r)
    # min_len = min(len(left), len(right))
    # stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    # stereo = stereo / np.max(np.abs(stereo)) * 0.8
    # sf.write(f'/mnt/c/Users/spt904/Desktop/cipic_{label}.wav', stereo.astype(np.float32), sr_cipic)
    # print(f"生成 cipic_{label}.wav")