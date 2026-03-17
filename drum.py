screen_width_m  = 0.61   # 螢幕物理寬度，公尺
screen_height_m = 0.39   # 螢幕物理高度，公尺
screen_res_w = 2560
screen_res_h = 1440
listener_dist   = 0.70   # 受試者距離螢幕，公尺

scale_x = screen_width_m  / screen_res_w
scale_y = screen_height_m / screen_res_h

scale_factor = 5  # 調這個數字

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
f = netCDF4.Dataset('/home/yyc/H5_96K_24bit_512tap_FIR_SOFA.sofa', 'r')
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
sr = 96000
duration = 10.0
def make_noise_burst(duration_ms=1500):
    samples = int(sr * duration_ms / 1000)
    noise = np.random.normal(0, 1, samples)
    # 加上 envelope 讓頭尾淡入淡出
    envelope = np.hanning(samples)
    burst = noise * envelope
    return (burst / np.max(np.abs(burst))).astype(np.float32)

signal = make_noise_burst(duration_ms=1500)
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
az_list  = [270, 292, 315, 337, 0, 22, 45, 67, 90]  # 9個水平角度
# 但我們只要5x5，用這5個
az_list  = [270, 315, 0, 45, 90]    # 左、左前、正前、右前、右
el_list  = [45, 22, 0, -22, -45]    # 上、上中、水平、下中、下

stimuli_world = {}
n = 1
for el in el_list:
    for az in az_list:
        stimuli_world[str(n)] = (az, el)
        n += 1


# 然後批次產生所有刺激
for name, (az, el) in stimuli_world.items():
    print(f"{name}: az={az}°, el={el}°")
    
    hrir_l, hrir_r = get_hrir(az, el)
    
    left  = fftconvolve(signal, hrir_l)
    right = fftconvolve(signal, hrir_r)
    
    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    stereo = stereo / np.max(np.abs(stereo)) * 0.8
    
    sf.write(f'stimulus_{name}.wav', stereo.astype(np.float32), sr)
    print(f"生成 stimulus_{name}.wav")
