screen_width_m  = 0.61   # 螢幕物理寬度，公尺
screen_height_m = 0.39   # 螢幕物理高度，公尺
screen_res_w = 2560
screen_res_h = 1440
listener_dist   = 0.70   # 受試者距離螢幕，公尺

scale_x = screen_width_m  / screen_res_w
scale_y = screen_height_m / screen_res_h

scale_factor = 20  # 調這個數字

def pixel_to_world(px, py):
    dx = px - screen_res_w / 2
    dy = py - screen_res_h / 2

    x = dx * scale_x * scale_factor + 5.0   # center in room
    y = listener_dist + 5.0                  # offset into room
    z = -dy * scale_y * scale_factor + 2.0  # center in room height

    return x, y, z

import numpy as np
from scipy.signal import fftconvolve
import netCDF4
import soundfile as sf
import pyroomacoustics as pra


# ─────────────────────────────
# 1. 載入 HRTF 資料庫
# ─────────────────────────────
f = netCDF4.Dataset('/home/yyc/H5_48K_24bit_256tap_FIR_SOFA.sofa', 'r')
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
sr = 48000
from scipy.signal import butter, lfilter

def boost_elevation_cues(signal, el, sr):
    # 上方：強化 8kHz 以上
    # 下方：強化 4kHz 附近
    if el > 0:
        b, a = butter(2, 8000/(sr/2), btype='high')
    else:
        b, a = butter(2, [3000/(sr/2), 6000/(sr/2)], btype='band')
    return lfilter(b, a, signal)
def make_harmonic_burst(f0=2000, n_harmonics=20, duration_ms=200):
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
signal = make_harmonic_burst(f0=2000, n_harmonics=20, duration_ms=200)
#print("signal shape:", signal.shape)
#print("signal max:", np.max(np.abs(signal)))

room_ref = pra.AnechoicRoom(3, fs=sr)



mic_center = np.array([5.0, 5.0, 2.0])
mic_positions = np.array([
    mic_center + [0.085, 0, 0],
    mic_center + [-0.085, 0, 0]
]).T

# 先跑一次找最近點的最大音量當基準

ref_x, ref_y, ref_z = pixel_to_world(960, 540)  # 正中心最近
room_ref = pra.AnechoicRoom(3, fs=sr)
room_ref.add_microphone(mic_positions)
room_ref.add_source([ref_x, ref_y, ref_z], signal=signal)
room_ref.simulate()
ref_max = np.max(np.abs(room_ref.mic_array.signals))
# 直接用角度定義，不用世界座標
az_list = [315, 326, 337, 348, 0, 12, 22, 34, 45]   # 左前到右前，9個
el_list = [45, 33, 22, 11, 0, -11, -22, -33, -45]   # 上到下，9個

stimuli_world = {}
n = 1
for el in el_list:
    for az in az_list:
        stimuli_world[str(n)] = (az, el)
        n += 1
# 只測上下極端值
for el in [45, 0, -45]:
    hrir_l, hrir_r = get_hrir(0, el)  # az=0 正前方，只改 el
    left  = fftconvolve(signal, hrir_l)
    right = fftconvolve(signal, hrir_r)
    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    stereo = stereo / np.max(np.abs(stereo)) * 0.8
    sf.write(f'/mnt/c/Users/spt904/Desktop/test_el_{el}.wav', stereo.astype(np.float32), sr)
# 只測第一個
name, (az, el) = list(stimuli_world.items())[0]
hrir_l, hrir_r = get_hrir(az, el)
print("hrir shape:", hrir_l.shape)
print("hrir max:", np.max(np.abs(hrir_l)))

left = fftconvolve(signal, hrir_l)
print("left shape:", left.shape)
print("left max:", np.max(np.abs(left)))

stereo = np.stack([left[:len(left)], left[:len(left)]], axis=1)
stereo = stereo / np.max(np.abs(stereo)) * 0.8

sf.write('/mnt/c/Users/spt904/Desktop/debug_test.wav', stereo.astype(np.float32), sr)
print("存到桌面了")
#sf.write('debug_test.wav', stereo.astype(np.float32), sr)
#print("存完，聽 debug_test.wav")
print("signal shape:", signal.shape)
print("signal max:", np.max(np.abs(signal)))
print("hrir shape:", hrir_l.shape)
print("hrir max:", np.max(np.abs(hrir_l)))
print("left max:", np.max(np.abs(left)))
# 然後批次產生所有刺激

# for name, (az, el) in stimuli_world.items():
#     hrir_l, hrir_r = get_hrir(az, el)
    
#     left  = fftconvolve(signal, hrir_l)
#     right = fftconvolve(signal, hrir_r)
    
#     min_len = min(len(left), len(right))
#     stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
#     stereo = stereo / np.max(np.abs(stereo)) * 0.8
#     output_dir = '/mnt/c/Users/spt904/Desktop/stimuli/'
#     import os
#     os.makedirs(output_dir, exist_ok=True)

#     sf.write(f'{output_dir}stimulus_{name}.wav', stereo.astype(np.float32), sr)

#     print(f"生成 stimulus_{name}.wav")
