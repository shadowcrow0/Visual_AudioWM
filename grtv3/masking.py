"""遮蔽：視覺 = 四格隨機紋理；聽覺 = 語音形狀噪音一小段，位準對齊 0 dB SNR 噪音的 RMS（GRTv3_ada_EN.py:883-924）。純 numpy。"""
import os

import numpy as np

import snr_audio
from . import config as C


def mask_reference(token=C.SND_TOKEN, n_seeds=16):
    """以 n_seeds 個種子估 0 dB SNR 混音裡噪音段的 RMS（去掉前導與尾巴），回傳 (sr, ref_rms)。"""
    v = []
    for s in range(n_seeds):
        sr, _sp, nz, lead = snr_audio.mix_components(token, 0.0, np.random.default_rng(s))
        tail = int(round(snr_audio.NOISE_TAIL_MS * sr / 1000.0))
        v.append(float(np.mean(nz[lead:len(nz) - tail] ** 2)))
    return sr, float(np.sqrt(np.mean(v)))


class MaskNoise:
    def __init__(self, outdir, sr, ref_rms, dur=C.MASK_DUR, db=C.MASK_NOISE_DB):
        self.outdir, self.sr, self.ref_rms, self.dur, self.db = outdir, sr, ref_rms, dur, db
        self.n_samp = int(round(dur * sr))

    def make(self, trial_idx):
        """新種子的噪音段寫成 wav；回傳 dict(seed, db, rms, path, summary)，summary 進資料檔的 snd_mask 欄。"""
        seed = snr_audio.new_seed()
        rms = self.ref_rms * 10.0 ** (self.db / 20.0)
        y = snr_audio.speech_shaped_noise(self.n_samp, np.random.default_rng(seed)) * rms
        name = f"mask_{trial_idx:04d}_{self.db:+.2f}dB.wav"
        path = os.path.join(self.outdir, name)
        scale = snr_audio.write_wav(path, self.sr, y)
        return dict(seed=seed, db=float(self.db), rms=rms, path=path,
                    summary=f"ssn|{self.db:+.2f}dB|seed={seed}|{name}|scale={scale:.3f}")


def visual_mask_textures(seed, n=4, tex_n=C.MASK_TEX_N):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, (tex_n, tex_n)) for _ in range(n)]
