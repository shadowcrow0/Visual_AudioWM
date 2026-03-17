import numpy as np
from scipy.signal import fftconvolve
import netCDF4
import soundfile as sf
import pyroomacoustics as pra

room = pra.AnechoicRoom(3, fs=48000)

# ─────────────────────────────
# 1. 載入 HRTF 資料庫
# ─────────────────────────────
f = netCDF4.Dataset('/home/yyc/H3_48K_24bit_256tap_FIR_SOFA.sofa', 'r')
hrir = f.variables['Data.IR'][:]
positions = f.variables['SourcePosition'][:]
f.close()

# ─────────────────────────────
# 2. 找最接近目標方向的 HRIR
# ─────────────────────────────
def get_hrir(target_az, target_el):
    az = positions[:, 0]
    el = positions[:, 1]
    dist = np.sqrt((az - target_az)**2 + (el - target_el)**2)
    idx = np.argmin(dist)
    print(f"找到最近的點: az={az[idx]:.1f}°, el={el[idx]:.1f}°")
    hrir_right = hrir[idx, 0, :]  # 修正：index 0 是右耳
    hrir_left  = hrir[idx, 1, :]  # 修正：index 1 是左耳
    return hrir_left, hrir_right

# ─────────────────────────────
# 3. 產生音源訊號
# ─────────────────────────────
sr = 48000
t = np.linspace(0, 1, sr)
#signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
# 白噪音（寬頻，所有頻率都有）
#signal = np.random.randn(sr).astype(np.float32)
#signal = signal / np.max(np.abs(signal))
# 粉紅噪音（低頻多、高頻少，更像自然聲音）
white = np.random.randn(sr).astype(np.float32)
freqs = np.fft.rfftfreq(sr)
freqs[0] = 1  # 避免除以零
pink_filter = 1 / np.sqrt(freqs)
signal = np.fft.irfft(np.fft.rfft(white) * pink_filter).astype(np.float32)
signal = signal / np.max(np.abs(signal))
#room.add_source([3, 0, 0], signal=signal)  # 遠處
# vs
room.add_source([0.5, 0, 0], signal=signal)  # 近處
mic_positions = np.array([[0.085, 0, 0], [-0.085, 0, 0]]).T
room.add_microphone(mic_positions)
room.simulate()
signals = room.mic_array.signals

# ─────────────────────────────
# 4. 取 HRIR 並做卷積
# ─────────────────────────────
#hrir_l, hrir_r = get_hrir(target_az=90, target_el=0)  # 正右方

#hrir_l, hrir_r =get_hrir(target_az=0,   target_el=0)   # 正前方
#hrir_l, hrir_r =get_hrir(target_az=180, target_el=0)   # 正後方
hrir_l, hrir_r =get_hrir(target_az=-90, target_el=0)   # 正左方
#hrir_l, hrir_r =get_hrir(target_az=0,   target_el=90)  # 正上方
left  = fftconvolve(signal, hrir_l)
right = fftconvolve(signal, hrir_r)

# ─────────────────────────────
# 5. 組成立體聲輸出
# ─────────────────────────────
min_len = min(len(left), len(right))
stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
stereo = stereo / np.max(np.abs(stereo))

sf.write('output_left_close.wav', stereo.astype(np.float32), sr)
print("完成，聽 output_left_close.wav")
