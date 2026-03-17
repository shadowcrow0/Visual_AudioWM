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
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

def sph_to_cart(az_deg, el_deg):
    # 這個資料庫：az=0 正前方，az=90 右方
    # 轉成標準球座標：x=右，y=前，z=上
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    x = np.cos(el) * np.sin(az)   # 右方分量
    y = np.cos(el) * np.cos(az)   # 前方分量
    z = np.sin(el)                 # 上方分量
    return x, y, z
# 建立插值器（只跑一次，放在載入 HRTF 之後）
points = np.array([sph_to_cart(az, el)
                   for az, el in zip(positions[:,0], positions[:,1])])

interp_l = LinearNDInterpolator(points, hrir[:, 1, :])
interp_r = LinearNDInterpolator(points, hrir[:, 0, :])

def get_hrir(target_az, target_el):
    x, y, z = sph_to_cart(target_az, target_el)
    hrir_l = interp_l(x, y, z)[0]
    hrir_r = interp_r(x, y, z)[0]
    if np.any(np.isnan(hrir_l)):
        print(f"插值失敗，改用最近點")
        az = positions[:, 0]
        el = positions[:, 1]
        dist = np.sqrt((az - target_az)**2 + (el - target_el)**2)
        idx = np.argmin(dist)
        hrir_l = hrir[idx, 1, :]
        hrir_r = hrir[idx, 0, :]
    return hrir_l.flatten().astype(np.float32), hrir_r.flatten().astype(np.float32)
sr = 48000

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
