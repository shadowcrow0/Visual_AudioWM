screen_width_m  = 0.60   # 螢幕物理寬度，公尺
screen_height_m = 0.34   # 螢幕物理高度，公尺
screen_res_w    = 1920   # 水平解析度
screen_res_h    = 1080   # 垂直解析度
listener_dist   = 0.70   # 受試者距離螢幕，公尺

scale_x = screen_width_m  / screen_res_w
scale_y = screen_height_m / screen_res_h

scale_factor = 10  # 調這個數字

def pixel_to_world(px, py):
    dx = px - screen_res_w / 2
    dy = py - screen_res_h / 2
    
    x = dx * scale_x * scale_factor
    y = listener_dist
    z = -dy * scale_y * scale_factor
    
    return x, y, z

# 螢幕中心（正前方）
#x, y, z = pixel_to_world(960, 540)

# 螢幕左邊
#x, y, z = pixel_to_world(0, 540)

# 螢幕右邊
#x, y, z = pixel_to_world(1920, 540)

# 螢幕左上角
x, y, z = pixel_to_world(0, 0)

# 螢幕右下角
#x, y, z = pixel_to_world(1920, 1080)
print(x)
print(y)
print(z)

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



# ─────────────────────────────
# 3. 產生音源訊號
# ─────────────────────────────
sr = 48000
duration = 0.6
num_samples = int(sr * duration)
noise = np.random.normal(0, 1, num_samples)
signal = (noise / np.max(np.abs(noise))).astype(np.float32)

# 5. 組成立體聲輸出
# ─────────────────────────────
#min_len = min(len(left), len(right))
#stereo = np.stack([left[:min_len], right[:min_len]], axis=1)

# # 不要用這個
# # stereo = stereo / np.max(np.abs(stereo))

# # 改用這個
# # 先跑一次近處當基準，記住它的最大值
# ref_x, ref_y, ref_z = pixel_to_world(960, 540)  # 螢幕中心當基準
# room_ref = pra.AnechoicRoom(3, fs=sr)
# room_ref.add_microphone(mic_positions)
# room_ref.add_source([ref_x, ref_y, ref_z], signal=signal)
# room_ref.simulate()
# ref_max = np.max(np.abs(room_ref.mic_array.signals))

# # 然後所有輸出都除以這個基準值再乘以目標音量
# target_volume = 0.8
# stereo = stereo * (target_volume / ref_max)
# stereo = np.clip(stereo, -1.0, 1.0)

# sf.write('output_topleft.wav', stereo.astype(np.float32), sr)
# print("完成，聽 output_topleft.wav")
# print(f"音源座標: x={x:.2f}, y={y:.2f}, z={z:.2f}")
# print(f"距離聽者: {np.sqrt(x**2 + y**2 + z**2):.2f} 公尺")
# print(f"ref_max: {ref_max:.6f}")
# print(f"stereo max before clip: {np.max(np.abs(stereo)):.4f}")


# 所有你要用的點位
stimuli = [
    (0, 0),       # 左上角
    (1920, 1080), # 右下角
    (200, 540),   # 左邊靠中間
    (960, 200),   # 上面靠中間
    (960, 540),   # 正中心
]
mic_positions = np.array([[0.085, 0, 0], [-0.085, 0, 0]]).T
# 先跑一次找最近點的最大音量當基準
ref_x, ref_y, ref_z = pixel_to_world(960, 540)  # 正中心最近
room_ref = pra.AnechoicRoom(3, fs=sr)
room_ref.add_microphone(mic_positions)
room_ref.add_source([ref_x, ref_y, ref_z], signal=signal)
room_ref.simulate()
ref_max = np.max(np.abs(room_ref.mic_array.signals))

# 然後批次產生所有刺激
for px, py in stimuli:
    x, y, z = pixel_to_world(px, py)
    
    room = pra.AnechoicRoom(3, fs=sr)
    room.add_microphone(mic_positions)
    room.add_source([x, y, z], signal=signal)
    room.simulate()
    signals = room.mic_array.signals
    
    az = np.degrees(np.arctan2(x, y)) % 360
    el = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    hrir_l, hrir_r = get_hrir(az, el)
    
    left  = fftconvolve(signals[0], hrir_l)
    right = fftconvolve(signals[1], hrir_r)
    
    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    
    # 用正中心當基準，保留相對距離感
    target_volume = 0.8
    stereo = stereo * (target_volume / ref_max)
    stereo = np.clip(stereo, -1.0, 1.0)
    
    sf.write(f'stimulus_{px}_{py}.wav', stereo.astype(np.float32), sr)
    print(f"生成 stimulus_{px}_{py}.wav")