import numpy as np
import pyroomacoustics as pra
import soundfile as sf

sr = 44100
t = np.linspace(0, 1, sr)
signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# 建立無響室
room = pra.AnechoicRoom(3, fs=sr)  # 3 = 三維空間

# 放音源，座標單位是公尺
# 聽者在原點，音源在右前方 1 公尺
room.add_source([1, 1, 0], signal=signal)

# 放左右耳麥克風
mic_positions = np.array([
    [ 0.085, 0, 0],  # 右耳
    [-0.085, 0, 0]   # 左耳
]).T

room.add_microphone(mic_positions)
room.simulate()

# 取出訊號
stereo = room.mic_array.signals.T  # (samples, 2)
stereo = stereo / np.max(np.abs(stereo))
sf.write("anechoic_test.wav", stereo.astype(np.float32), sr)