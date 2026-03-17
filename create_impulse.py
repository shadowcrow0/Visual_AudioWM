import numpy as np
import scipy.io.wavfile as wav

# Parameters
fs = 44100          # Sampling rate (Hz)
duration = 0.6      # 100ms
num_samples = int(fs * duration)

# Generate Gaussian white noise (mean=0, std=1)
# Amplitude (0.0 to 1.0)
noise = np.random.normal(0, 1, num_samples)

# Normalize/Scale for 16-bit PCM WAV (int16)
audio_data = (noise * 32767 * 0.5).astype(np.int16)  

# Save or play the sound
wav.write("gaussian_impulse.wav", fs, audio_data)
print("Gaussian impulse generated and saved as 'gaussian_impulse.wav'")


def make_click(duration_ms=50):
    """
    短暫的高斯脈衝，聽起來像拍一下
    """
    samples = int(sr * duration_ms / 1000)
    t = np.linspace(-3, 3, samples)
    click = np.exp(-t**2)  # 高斯形狀
    return click.astype(np.float32)