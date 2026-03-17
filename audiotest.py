import numpy as np
from scipy.signal import fftconvolve
import netCDF4
import pyroomacoustics as pra
import soundfile as sf

# ─────────────────────────────
# 載入 HRTF
# ─────────────────────────────
f = netCDF4.Dataset('/home/yyc/H3_48K_24bit_256tap_FIR_SOFA.sofa', 'r')
hrir = f.variables['Data.IR'][:]
positions = f.variables['SourcePosition'][:]
f.close()

sr = 48000

def get_hrir(target_az, target_el=0):
    az = positions[:, 0]
    el = positions[:, 1]
    dist = np.sqrt((az - target_az)**2 + (el - target_el)**2)
    idx = np.argmin(dist)
    print(f"找到最近的點: az={az[idx]:.1f}°, el={el[idx]:.1f}°")
    return hrir[idx, 1, :], hrir[idx, 0, :]  # left, right

# ─────────────────────────────
# 產生複合音
# ─────────────────────────────
def make_harmonic_complex(f0=440, n_harmonics=10, duration=1.0):
    """
    f0: 基頻
    n_harmonics: 諧波數量，越多頻譜越豐富
    duration: 秒
    """
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    
    for n in range(1, n_harmonics + 1):
        freq = f0 * n
        if freq > sr / 2:  # 超過 Nyquist 就停
            break
        signal += np.sin(2 * np.pi * freq * t)
    
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

# ─────────────────────────────
# 生成空間刺激
# ─────────────────────────────
def generate_stimulus(x, y, signal, filename):
    source_pos = [x, y, 0]
    
    # pyroomacoustics 處理距離衰減
    room = pra.AnechoicRoom(3, fs=sr)
    mic_positions = np.array([[0.085, 0, 0], [-0.085, 0, 0]]).T
    room.add_microphone(mic_positions)
    room.add_source(source_pos, signal=signal)
    room.simulate()
    signals = room.mic_array.signals

    # 換算方位角
    az = np.degrees(np.arctan2(x, y)) % 360

    # HRTF 卷積
    hrir_l, hrir_r = get_hrir(az)
    left  = fftconvolve(signals[0], hrir_l)
    right = fftconvolve(signals[1], hrir_r)

    min_len = min(len(left), len(right))
    stereo = np.stack([left[:min_len], right[:min_len]], axis=1)
    stereo = stereo / np.max(np.abs(stereo))
    sf.write(filename, stereo.astype(np.float32), sr)
    print(f"生成 {filename}")

# ─────────────────────────────
# 批次產生刺激
# ─────────────────────────────
signal = make_harmonic_complex(f0=440, n_harmonics=10, duration=1.0)

stimuli = [
    (-1,  1),   # 左前方
    ( 1,  1),   # 右前方
    (-1, -1),   # 左後方
    ( 1, -1),   # 右後方
    (-3,  0),   # 左方遠
    ( 3,  0),   # 右方遠
    ( 0,  3),   # 前方遠
    ( 0, -3),   # 後方遠
]

for x, y in stimuli:
    filename = f'stimulus_x{x}_y{y}.wav'
    generate_stimulus(x, y, signal, filename)