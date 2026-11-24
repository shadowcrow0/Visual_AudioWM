"""聲學尺（一）：從頻譜圖算每一對子音在各 SNR 下「還剩多少可分辨的差別」。

讀：stimuli/T??_<子音>3.wav（16 個子音 × 12 位說話者；清單來自 consonant_distance.CONSONANTS / adaptive_vawm_EN.TALKERS）
算：每個音檔、每個 SNR（乾淨、+12、+6、0、-6 dB）、每個噪音種子（6 個）的 mel 頻譜圖
      32 個 mel 頻帶（100-7000 Hz）、25 ms 窗、10 ms 一格、dB；取母音起點前 300 ms 到後 250 ms。
      兩個音節對齊在「嘴巴打開進入母音」那一刻（500-1500 Hz 包絡第一次到峰值 -10 dB），共同的母音因此相消。
    每一對、每個 SNR 四個數：
      D        同一位說話者兩個音的頻譜圖差（-300 到 +100 ms：子音 + 進入母音的過渡）的 rms dB；兩個音用不同的噪音樣本
      floor    同一個音、兩段不同噪音的 D（噪音自己造成的差）；乾淨時為 0
      excess   √(D² − floor²)：扣掉噪音自己的貢獻後剩下的差別。這是企劃書說的「頻譜圖上的差別」
      dprime   留一位說話者的樣板觀察者：樣板 = 另外 11 位在該 SNR 的平均頻譜圖；
               決策量 = (到錯樣板的距離)² − (到對樣板的距離)²；d′ = 2 × 平均 ÷ SD；acc = 決策量 > 0 的比例
出：analyze/out/acoustic_scale_feats.npz（約 64 MB，全部頻譜圖）、acoustic_scale_pairs.csv（120 對 × 5 個 SNR）
對應：review/proposal_2026-11-24_language_independent_sound_scale.md §1、§3 (A)；跑一次約 25 秒。

為什麼這樣算：
- 噪音用實驗的做法（sft_noise 的語音形噪音與 SNR 定義），不是白噪音：語音形噪音先蓋低頻，
  同樣的 dB 與 Miller & Nicely 的白噪音不是同樣的遮蔽，所以這把尺要在「我們的噪音」下算。
- sft_noise.NOISE_LEAD_MS 在這裡（只在這個行程裡）改成 450 ms：實驗用 150 ms，但分析視窗要往前看 300 ms，
  前導噪音不夠長的話視窗前段會是靜音而不是噪音。噪音的種類與 SNR 的定義沒有動。
- 扣 floor：噪音本身就讓兩張頻譜圖不一樣；不扣的話 SNR 越低 D 反而可能越大。
- 樣板觀察者是「聲音裡有多少資訊」的上限，遠優於人（0 dB 仍有一百多對正確率 ≥ .90），只能看排序，不能當絕對難度。
- 所有時間—頻率格子等權重：不看聽者注意哪裡。這正是它「跟母語無關」的來源，也是它的限制。
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import _common as C
import adaptive_vawm_EN as A
import consonant_distance as cd
import sft_noise as sn
import snr_audio as sa

sn.NOISE_LEAD_MS = 450.0                       # 只影響這支分析；見檔頭「為什麼」
CONS = cd.CONSONANTS
TALKERS = A.TALKERS
SNRS = [None, 12.0, 6.0, 0.0, -6.0]
SNR_KEYS = ["clean", "12.0", "6.0", "0.0", "-6.0"]      # pairs.csv 的 snr 欄讀回來轉成字串後的樣子
NSEED = 6
HOP, WIN, NB = 0.010, 0.025, 32
T_PRE, T_POST = 30, 25                         # 母音起點前 / 後的格數（10 ms 一格）
W_CONS = slice(0, T_PRE + 10)                  # -300 .. +100 ms：子音 + 進入母音的過渡
W_VOW = slice(T_PRE + 12, T_PRE + 25)          # +120 .. +250 ms：母音本體


def mel_fb(sr, nfft):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(100), mel(7000), NB + 2))
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    fb = np.zeros((NB, len(fr)))
    for b in range(NB):
        lo, c, hi = edges[b:b + 3]
        fb[b] = np.clip(np.minimum((fr - lo) / (c - lo), (hi - fr) / (hi - c)), 0, None)
    return fb


def band_centres():
    """32 個 mel 頻帶的中心頻率（Hz），畫圖用。"""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    return imel(np.linspace(mel(100), mel(7000), NB + 2))[1:-1]


def vowel_onset(x, sr):
    """500-1500 Hz 包絡第一次到峰值 -10 dB 的樣本：嘴巴打開進入母音。"""
    y = sosfiltfilt(butter(4, [500, 1500], btype="band", fs=sr, output="sos"), x)
    n = int(.010 * sr)
    e = np.sqrt(np.convolve(y ** 2, np.ones(n) / n, mode="same"))
    return int(np.argmax(e > e.max() * 10 ** (-10 / 20)))


def spec(y, sr, centre):
    """以 centre + k × hop（k = -T_PRE .. T_POST-1）為中心的 mel dB 格；檔案外面的樣本當靜音。"""
    w = int(WIN * sr)
    nfft = 512
    fb = mel_fb(sr, nfft)
    pad = np.concatenate([np.zeros(sr), y, np.zeros(sr)])
    out = np.empty((T_PRE + T_POST, NB))
    for i, k in enumerate(range(-T_PRE, T_POST)):
        c = sr + centre + int(round(k * HOP * sr))
        seg = pad[c - w // 2: c - w // 2 + w] * np.hanning(w)
        out[i] = fb @ (np.abs(np.fft.rfft(seg, nfft)) ** 2)
    return 10 * np.log10(out + 1e-12)


def features():
    """F[si] : [子音, 說話者, 種子, 格, 頻帶]。"""
    F = {}
    for si, snr in enumerate(SNRS):
        X = np.empty((len(CONS), len(TALKERS), NSEED, T_PRE + T_POST, NB))
        for ci, c in enumerate(CONS):
            for ti, tk in enumerate(TALKERS):
                wav = A.sound_file(tk, c)                              # "stimuli/T01_p3.wav"：也是種子的一部分，所以用資料檔裡的相對路徑
                sr, x, on, off = sn._load(wav)
                von = vowel_onset(x, sr)
                for k in range(NSEED):
                    if snr is None:
                        y = x * (sa.OUTPUT_RMS / sa._rms(x[on:off]))
                        centre = von
                    else:
                        _, sp, nz, span = sn.mix_components(wav, snr, sn.seed_for("info", wav, snr, k))
                        y = sp + nz
                        centre = span.start - on + von
                    X[ci, ti, k] = spec(y, sr, centre)
        ref = np.median(X[:, :, :, W_VOW].max(-1))                    # 母音裡最強頻帶的典型位準
        F[si] = np.maximum(X, ref - 50)                               # 比母音低 50 dB 以上的不算（絕對下限；只影響乾淨條件）
        print("features", "clean" if snr is None else snr, X.shape, "ref %.1f dB" % ref, flush=True)
    return F


def pairs(F):
    rms = lambda d, w: np.sqrt(np.mean(d[..., w, :] ** 2, axis=(-1, -2)))
    rows = []
    for si, snr in enumerate(SNRS):
        X = F[si]
        Xn = X - X.mean(axis=(0, 2, 3), keepdims=True)                # 減掉每位說話者的平均頻譜（只給樣板觀察者用）
        fl = np.zeros((len(CONS), len(TALKERS), 2))                   # 每個音檔的噪音底：同一個音、兩段噪音
        if snr is not None:
            for a_, b_ in itertools.combinations(range(NSEED), 2):
                fl[..., 0] += rms(X[:, :, a_] - X[:, :, b_], W_CONS) ** 2
                fl[..., 1] += rms(X[:, :, a_] - X[:, :, b_], W_VOW) ** 2
            fl = np.sqrt(fl / (NSEED * (NSEED - 1) / 2))
        for i, j in itertools.combinations(range(len(CONS)), 2):
            # (a) 同一位說話者的頻譜圖差；一個音用噪音樣本 k、另一個用 k+1（絕不用同一段噪音）
            Dc = np.mean([rms(X[i, :, k] - X[j, :, (k + 1) % NSEED], W_CONS) for k in range(NSEED)], 0)
            Dv = np.mean([rms(X[i, :, k] - X[j, :, (k + 1) % NSEED], W_VOW) for k in range(NSEED)], 0)
            fc = np.sqrt((fl[i, :, 0] ** 2 + fl[j, :, 0] ** 2) / 2)
            fv = np.sqrt((fl[i, :, 1] ** 2 + fl[j, :, 1] ** 2) / 2)
            ex = np.sqrt(np.clip(Dc ** 2 - fc ** 2, 0, None))
            # (b) 留一位說話者的樣板觀察者
            dv = {i: [], j: []}
            for t in range(len(TALKERS)):
                others = [u for u in range(len(TALKERS)) if u != t]
                Ti = Xn[i, others].mean((0, 1))[W_CONS]
                Tj = Xn[j, others].mean((0, 1))[W_CONS]
                for c, right, wrong in ((i, Ti, Tj), (j, Tj, Ti)):
                    for k in range(NSEED if snr is not None else 1):
                        x_ = Xn[c, t, k][W_CONS]
                        dv[c].append(np.mean((x_ - wrong) ** 2) - np.mean((x_ - right) ** 2))
            allv = np.concatenate([np.array(dv[i]), np.array(dv[j])])
            # 決策量是「支持正確答案的證據」：等變異、判準在 0 時 d′ = 2 × 平均 ÷ SD
            rows.append(dict(a=CONS[i], b=CONS[j], snr="clean" if snr is None else snr, D=Dc.mean(), D_sd_talkers=Dc.std(ddof=1),
                             floor=fc.mean(), excess=ex.mean(), D_vowel=Dv.mean(), floor_vowel=fv.mean(),
                             dprime=2 * allv.mean() / allv.std(ddof=1), acc=float((allv > 0).mean()), n_trials=len(allv),
                             **{f"D_{tk}": v for tk, v in zip(TALKERS, Dc)}, **{f"ex_{tk}": v for tk, v in zip(TALKERS, ex)}))
        print("pairs", "clean" if snr is None else snr, flush=True)
    return pd.DataFrame(rows)


def load_pairs(out):
    """acoustic_scale_pairs.csv，加上無序對的鍵 pair（'b-p'）；snr 欄轉成字串（'clean'、'6.0'…）。"""
    P = pd.read_csv(os.path.join(out, "acoustic_scale_pairs.csv"))
    P["snr"] = P.snr.astype(str)
    P["pair"] = ["-".join(sorted([a, b])) for a, b in zip(P.a, P.b)]
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 analyze/out）")
    a = ap.parse_args()
    out = C.out_dir(a.out)
    F = features()
    np.savez_compressed(os.path.join(out, "acoustic_scale_feats.npz"), **{f"s{si}": F[si] for si in F},
                        cons=np.array(CONS), talkers=np.array(TALKERS))
    P = pairs(F)
    P.to_csv(os.path.join(out, "acoustic_scale_pairs.csv"), index=False)
    print(P.groupby("snr", sort=False)[["D", "floor", "excess", "D_vowel", "floor_vowel", "dprime", "acc"]].mean().round(2))
    print("寫出：", os.path.join(out, "acoustic_scale_feats.npz"), "、acoustic_scale_pairs.csv")


if __name__ == "__main__":
    main()
