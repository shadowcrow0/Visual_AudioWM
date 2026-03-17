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

# 訊號
sr = 48000
def make_pink_noise(duration_ms=500):
    """生成 pink noise (1/f noise) - 更自然的頻譜分佈，適合空間音訊測試"""
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
    """模擬空氣吸收：距離越遠，高頻衰減越多"""
    if distance <= 1.0:
        return signal
    # 距離越遠，截止頻率越低 (簡化的經驗模型)
    cutoff = max(2000, 20000 / (1 + 0.1 * (distance - 1)))
    b, a = butter(1, cutoff / (sr / 2), btype='low')
    return lfilter(b, a, signal)

signal = make_pink_noise()
signal_44k = librosa.resample(signal, orig_sr=48000, target_sr=44100)

# 9x9xN 格子
# x 軸：9 個 azimuth（左到右）
# y 軸：9 個距離（近到遠）
# z 軸：仰角（下到上）
az_values  = [-45, -35, -20, -10, 0, 10, 20, 35, 45]   # 左到右
dist_values = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]          # 近到遠（公尺）

# CIPIC 仰角索引對照：
# el_idx=0: -45°, el_idx=8: 0°, el_idx=16: +45°, el_idx=24: +90°
el_values = [
    (0, -45),   # 下方
    (8, 0),     # 水平
    (16, 45),   # 上方
]

# 計算基準音量（距離 1m 的音量）
ref_dist = 1.0

output_dir = '/mnt/c/Users/spt904/Desktop/stimuli/'
os.makedirs(output_dir, exist_ok=True)

n = 1
for el_idx, el_deg in el_values:
    for dist in dist_values:
        for az in az_values:
            # 取 HRTF
            az_idx = cipic_az.index(az)

            hrir_l = data['hrir_l'][az_idx, el_idx, :]
            hrir_r = data['hrir_r'][az_idx, el_idx, :]

            left  = fftconvolve(signal_44k, hrir_l)
            right = fftconvolve(signal_44k, hrir_r)

            # 空氣吸收（遠距離高頻衰減）
            left  = apply_air_absorption(left, dist, sr_cipic)
            right = apply_air_absorption(right, dist, sr_cipic)

            min_len = min(len(left), len(right))
            stereo = np.stack([left[:min_len], right[:min_len]], axis=1)

            # 距離衰減（稍微壓縮，讓遠處聲音別太小）
            gain = ref_dist / (dist ** 0.8)
            stereo = stereo * gain

            # 統一標準化到 -1dB 左右
            max_val = np.max(np.abs(stereo))
            if max_val > 0:
                stereo = stereo / max_val * 0.9

            sf.write(f'{output_dir}stimulus_{n}.wav', stereo.astype(np.float32), sr_cipic)
            print(f"生成 stimulus_{n}.wav  az={az}°  el={el_deg}°  dist={dist}m")
            n += 1