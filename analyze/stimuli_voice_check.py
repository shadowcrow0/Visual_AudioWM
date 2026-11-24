"""音檔檢查：「12 位說話者」是幾個不同的聲音？母音是什麼？

    python analyze/stimuli_voice_check.py

讀：stimuli/T??_<子音>3.wav（12 × 16 = 192 個）；analyze/out/acoustic_scale_pairs.csv（有的話）
算：1. 每兩位說話者，同一個子音的音檔：10 ms 音量包絡的相關（先重取樣到同一個取樣率），對 16 個子音平均
    2. 每個音檔母音中段的音高（自相關，70–350 Hz）：同一位說話者 16 個音檔的中位數與全距
    3. 母音中段（母音起點後 80–160 ms）的 F1、F2、F3（Praat Burg，需要 praat-parselmouth；沒裝就略過）
    4. 聲學尺上同一對在 12 位說話者之間的差異，分成「同一組聲音內」與「三組之間」
出：analyze/out/stimuli_voice_check.json；印出摘要
對應：review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md §3。

為什麼要查：實驗把 T01–T12 當 12 個說話者抽樣（adaptive_vawm_EN.py 的 TALKERS），聲學尺的驗證也用「留一位說話者」。
如果其中幾位其實是同一段聲音換了音高，獨立的來源就比 12 少。這裡只報量到的相似度；音檔的來源與製作方式
這支程式不知道（review/論文規格書_聲音軸.md A2 也標為待填）。
"""
import json
import os
from fractions import Fraction

import numpy as np
from scipy.signal import correlate, resample_poly

import _common as C
import snr_audio as sa

TK = [f"T{i:02d}" for i in range(1, 13)]
CONS = ["p", "t", "k", "f", "theta", "s", "sh", "b", "d", "g", "v", "eth", "z", "zh", "m", "n"]
SR0 = 15000
GROUPS = (("T01-T06", range(0, 6)), ("T07-T09", range(6, 9)), ("T10-T12", range(9, 12)))     # 由下面第 1 項的相關矩陣看出來的分組


def load(t, c):
    sr, x = sa._read_wav(os.path.join(C.ROOT, "stimuli", f"{t}_{c}3.wav"))
    fr = Fraction(SR0, sr)
    return resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)


def envelope(x, w=0.010):
    n = int(w * SR0)
    return np.sqrt(np.convolve(x ** 2, np.ones(n) / n, "same"))[::n]


def env_corr(a, b):
    ea, eb = envelope(a), envelope(b)
    n = min(len(ea), len(eb))
    return float(np.corrcoef(ea[:n], eb[:n])[0, 1])


def pitch(x):
    seg = x[len(x) // 2: len(x) // 2 + int(0.06 * SR0)]
    seg = seg - seg.mean()
    ac = correlate(seg, seg, mode="full")[len(seg) - 1:]
    lo, hi = int(SR0 / 350), int(SR0 / 70)
    return SR0 / (lo + int(np.argmax(ac[lo:hi])))


def vowel_formants(x):
    """母音中段的 F1-F3：取檔案後半能量最高的 80 ms（子音之後），Praat Burg，五個上限取中位數。"""
    import parselmouth
    e = np.convolve(x ** 2, np.ones(int(0.08 * SR0)) / int(0.08 * SR0), "same")
    mid = len(x) // 3 + int(np.argmax(e[len(x) // 3:]))
    t = mid / SR0
    out = []
    for ceil in (4500, 5000, 5500):
        fm = parselmouth.Sound(x, SR0).to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceil, window_length=0.025)
        out.append([fm.get_value_at_time(k, t) for k in (1, 2, 3)])
    return np.nanmedian(np.array(out, float), axis=0)


if __name__ == "__main__":
    X = {(t, c): load(t, c) for t in TK for c in CONS}
    E = np.array([[np.mean([env_corr(X[a, c], X[b, c]) for c in CONS]) for b in TK] for a in TK])
    f0 = {t: [pitch(X[t, c]) for c in CONS] for t in TK}
    res = {"envelope_corr": E.tolist(), "pitch_hz": {t: dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v))) for t, v in f0.items()}}
    inside = [E[i, j] for _, g in GROUPS for i in g for j in g if i < j]
    across = [E[i, j] for i in range(12) for j in range(i + 1, 12) if not any(i in g and j in g for _, g in GROUPS)]
    res["envelope_corr_within"] = [float(min(inside)), float(max(inside))]
    res["envelope_corr_between"] = [float(min(across)), float(max(across))]
    print("1. 音量包絡的相關（16 個子音平均）")
    print(f"   同一組內 {min(inside):.2f}–{max(inside):.2f}；不同組之間 {min(across):.2f}–{max(across):.2f}")
    for name, g in GROUPS:
        print(f"   {name}：組內 " + " ".join(f"{E[i, j]:.2f}" for i in g for j in g if i < j))
    print("2. 母音中段的音高（Hz；16 個音檔的中位數，括號內是全距）")
    print("   " + "  ".join(f"{t} {res['pitch_hz'][t]['median']:.0f}({res['pitch_hz'][t]['max'] - res['pitch_hz'][t]['min']:.0f})" for t in TK))
    try:
        F = {t: np.array([vowel_formants(X[t, c]) for c in CONS]) for t in TK}
        res["vowel_formants_hz"] = {t: [float(v) for v in np.nanmedian(F[t], 0)] for t in TK}
        allf = np.vstack(list(F.values()))
        res["vowel_formants_all"] = dict(median=[float(v) for v in np.nanmedian(allf, 0)], n=int(len(allf)), f3_below_2200=int((allf[:, 2] < 2200).sum()))
        print("3. 母音中段的 F1 / F2 / F3（Hz，各說話者 16 個音檔的中位數）")
        for name, g in GROUPS:
            print(f"   {name}：" + "；".join(f"{TK[i]} {res['vowel_formants_hz'][TK[i]][0]:.0f}/{res['vowel_formants_hz'][TK[i]][1]:.0f}/{res['vowel_formants_hz'][TK[i]][2]:.0f}" for i in g))
        m = res["vowel_formants_all"]
        print(f"   全部 {m['n']} 個音檔的中位數 {m['median'][0]:.0f} / {m['median'][1]:.0f} / {m['median'][2]:.0f}；F3 低於 2200 Hz 的有 {m['f3_below_2200']} 個")
    except ImportError:
        print("3. 沒有 praat-parselmouth，略過母音共振峰")
    f = os.path.join(C.OUT, "acoustic_scale_pairs.csv")
    if os.path.exists(f):
        import pandas as pd
        P = pd.read_csv(f, dtype={"snr": str})
        print("4. 聲學尺上同一對在說話者之間的差異（dB，120 對的中位數）")
        res["acoustic_scale_talker_sd"] = {}
        for snr in ("12.0", "6.0", "0.0"):
            V = P[P.snr == snr][[f"ex_{t}" for t in TK]].values
            within = np.sqrt(np.mean([V[:, list(g)].var(1, ddof=1) for _, g in GROUPS], 0))
            between = np.c_[[V[:, list(g)].mean(1) for _, g in GROUPS]].T.std(1, ddof=1)
            r = dict(sd_12=float(np.median(V.std(1, ddof=1))), sd_within_group=float(np.median(within)), sd_between_groups=float(np.median(between)))
            res["acoustic_scale_talker_sd"][snr] = r
            print(f"   SNR {float(snr):+g} dB：12 位之間的 SD {r['sd_12']:.2f}；同一組內 {r['sd_within_group']:.2f}；三組平均之間 {r['sd_between_groups']:.2f}")
    json.dump(res, open(os.path.join(C.out_dir(), "stimuli_voice_check.json"), "w"), ensure_ascii=False, indent=1)
    print("寫出：", os.path.join(C.OUT, "stimuli_voice_check.json"))
