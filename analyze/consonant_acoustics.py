"""實驗音檔的聲學量測：子音段的低頻能量、高頻噪音強度與重心，以及成對的頻譜圖。

讀：stimuli/T??_<子音>3.wav（16 個子音 × 12 位說話者；清單來自 adaptive_vawm_EN.TALKERS / MN_FEATURES）
算：每個音檔
      dur   子音段長度 = 聲學起點（snr_audio._acoustic_onset）到母音起點（500-1500 Hz 包絡第一次到母音峰值 -10 dB）
      low   子音段 80-350 Hz 的 RMS，相對母音（母音起點後 30-130 ms 的全頻 RMS = 0 dB）—— 聲帶振動的那條黑帶
      hi    子音段 1.5-7 kHz 的 RMS，相對母音 —— 摩擦 / 送氣噪音的強度
      cent  子音段 1-7 kHz 內的頻譜重心（Hz）
    再對 12 位說話者取平均與 SD。
出：analyze/out/consonant_acoustics.csv（每檔一列）、consonant_acoustics_summary.csv（每子音一列）、
    acoustics_lowband_voicing.png、acoustics_frication_map.png、acoustics_consonant_spectra.png、
    acoustics_pair_<a>_<b>.png（--pairs 指定的每一對一張：波形、頻譜圖、80-350 Hz 能量）。
對應：2026-11-24 對話中的圖 S1-S4、N1-N3 與其中的 dB / Hz 數字。

為什麼量這三個頻帶：英語的清濁靠 80-350 Hz（有沒有聲帶振動），部位與方式靠高頻噪音的位置與強度，
鼻音與塞音的部位靠進入母音的過渡。前兩個可以用固定頻帶量；過渡這裡只畫圖、不量（要追共振峰，自動量不可靠）。
這些量測是「這批錄音長什麼樣」，不是「聽者用什麼線索」——後者要行為資料。
取樣率因說話者而異（15-17 kHz），所以帶通上限取 min(7 kHz, 0.98 × Nyquist)。
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, spectrogram

import _common as C
import adaptive_vawm_EN as A
import snr_audio as sa

FRICATIVE_PAIRS = (("f", "v"), ("theta", "eth"), ("s", "z"), ("sh", "zh"))    # 清 / 濁，部位依序：唇齒、齒間、齒齦、顎齦
PLACE = ("唇齒", "齒間", "齒齦", "顎齦")
MANDARIN_NEAR = {"f": "ㄈ", "s": "ㄙ", "sh": "ㄕ/ㄒ"}                          # 擦音裡華語有相近類別的三個；其餘五個沒有（音韻學判斷）
# 成對圖上的框：(x0 ms, x1 ms, y0 kHz, y1 kHz, 文字, 顏色)。只給回報過的那幾對；其他對照樣畫，只是沒有框。
ANNOT = {("m", "n"): [(50, 172, 0.0, 1.0, "鼻音低頻帶：兩個幾乎一樣", C.RED), (172, 245, 0.9, 3.0, "進入母音的這一小段不同", C.BLUE)],
         ("f", "theta"): [(30, 150, 1.5, 7.0, "弱而平坦的噪音：兩個幾乎一樣", C.RED)],
         ("s", "n"): [(30, 160, 3.5, 7.0, "/s/：高頻強噪音；/n/ 這裡是空的", C.BLUE), (30, 110, 0.0, 1.0, "/n/：低頻鼻音；/s/ 這裡是空的", C.RED)],
         ("s", "z"): [(30, 220, 3.5, 7.0, "3.5 kHz 以上的噪音：兩個音在同一個位置", C.BLUE), (30, 220, 0.0, 0.45, "0-0.45 kHz：/s/ 是空的，/z/ 有一條黑帶（聲帶在振動）", C.RED)]}


def wav(talker, cons):
    return sa._read_wav(os.path.join(C.ROOT, A.sound_file(talker, cons)))


def band_env(x, sr, lo, hi, win=0.010):
    """lo-hi Hz 帶通後的 10 ms 移動 RMS 包絡。"""
    sos = butter(4, [lo, min(hi, sr / 2 * 0.98)], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    n = int(win * sr)
    return np.sqrt(np.convolve(y ** 2, np.ones(n) / n, mode="same") + 1e-12)


def measure(talker, cons):
    sr, x = wav(talker, cons)
    x = x / np.abs(x).max()
    on = sa._acoustic_onset(x, sr)
    e_v = band_env(x, sr, 500, 1500)                                   # /ɑ/ 的 F1 區：用它標母音
    von = int(np.argmax(e_v > e_v.max() * 10 ** (-10 / 20)))
    vseg = x[von + int(.03 * sr): von + int(.13 * sr)]
    vrms = np.sqrt(np.mean(vseg ** 2))
    c0, c1 = on, max(von - int(.005 * sr), on + int(.01 * sr))         # 子音段至少 10 ms，避免塞音起點 = 母音起點時變空
    cseg = x[c0:c1]
    db = lambda e: 20 * np.log10(np.sqrt(np.mean(e[c0:c1] ** 2)) / vrms)
    P = np.abs(np.fft.rfft(cseg * np.hanning(len(cseg)), 4096)) ** 2
    fr = np.fft.rfftfreq(4096, 1 / sr)
    hb = (fr >= 1000) & (fr <= 7000)
    return dict(talker=talker, consonant=cons, sr=sr, dur_ms=(von - on) / sr * 1000, low_db=db(band_env(x, sr, 80, 350)),
                hi_db=db(band_env(x, sr, 1500, 7000)), centroid_hz=(fr[hb] * P[hb]).sum() / P[hb].sum())


def fig_lowband(plt, M, out):
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, pair in enumerate(FRICATIVE_PAIRS):
        for k, c in enumerate(pair):
            m, sd = M[c].low_db.mean(), M[c].low_db.std(); xx = i + (k - .5) * .36; near = c in MANDARIN_NEAR
            ax.bar(xx, m + 45, .32, bottom=-45, color=C.BLUE if near else "white", ec=C.BLUE if near else C.RED, lw=1.6, hatch=None if near else "///")
            ax.errorbar(xx, m, sd, color="k", lw=1, capsize=3)
            ax.text(xx, -46.2, f"/{C.ipa(c)}/", ha="center", va="top", fontsize=12, color=C.BLUE if near else C.RED)
        d = M[pair[1]].low_db.mean() - M[pair[0]].low_db.mean()
        ax.annotate(f"差 {d:.0f} dB", (i, max(M[pair[0]].low_db.mean(), M[pair[1]].low_db.mean()) + 7), ha="center", fontsize=10)
        ax.text(i, -49.6, PLACE[i], ha="center", va="top", fontsize=10, color=C.GREY)
    ax.set_xticks([]); ax.set_ylim(-45, 0); ax.set_ylabel("子音段 80-350 Hz 的能量 (dB，母音 = 0)")
    ax.legend(handles=[Patch(fc=C.BLUE, ec=C.BLUE, label="華語有相近的音"), Patch(fc="white", ec=C.RED, hatch="///", label="華語沒有這個音")],
              frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.set_title("英語四組擦音靠 80-350 Hz 分清濁\n（12 位說話者平均 ± SD）", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "acoustics_lowband_voicing.png")); plt.close(fig)


def fig_map(plt, M, out):
    off = {"f": (-.12, -1.6), "theta": (.10, -1.9), "v": (-.2, 1.1), "eth": (.2, 1.6), "s": (.42, 1.5), "z": (.2, -1.5), "sh": (-.5, 1.6), "zh": (-.36, -1.5)}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for c in [c for p in FRICATIVE_PAIRS for c in p]:
        cx, cy = M[c].centroid_hz.mean() / 1000, M[c].hi_db.mean(); near = c in MANDARIN_NEAR
        ax.errorbar(cx, cy, xerr=M[c].centroid_hz.std() / 1000, yerr=M[c].hi_db.std(), fmt="o", ms=11, mfc=C.BLUE if near else "white",
                    mec=C.BLUE if near else C.RED, mew=1.8, ecolor=C.GREY, elinewidth=1, capsize=2)
        ax.text(cx + off[c][0], cy + off[c][1], f"/{C.ipa(c)}/" + (f" ≈ {MANDARIN_NEAR[c]}" if near else ""), fontsize=11,
                color=C.BLUE if near else C.RED, ha="center", va="center")
    ax.plot([], [], "o", ms=9, mfc=C.BLUE, mec=C.BLUE, label="華語有相近的音"); ax.plot([], [], "o", ms=9, mfc="white", mec=C.RED, mew=1.8, label="華語沒有這個音")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_xlim(.8, 6.4); ax.set_ylim(-44, -10)
    ax.set_xlabel("噪音的重心頻率 (kHz，1-7 kHz 內)"); ax.set_ylabel("1.5-7 kHz 噪音的強度 (dB，母音 = 0)")
    ax.set_title("八個擦音的高頻噪音：重心頻率 × 強度（12 位說話者平均 ± SD）", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "acoustics_frication_map.png")); plt.close(fig)


def fig_spectra(plt, cons, talkers, out):
    """子音起點後 80 ms 的平均頻譜（各自最大值 = 0 dB），看頻譜「形狀」像不像。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); grid = np.linspace(200, 6500, 130)
    styles = [(C.RED, "-"), (C.ORANGE, "--"), (C.BLUE, "-"), ("#5dade2", "--"), (C.GREEN, "-"), (C.GREY, "--"), ("k", "-"), ("k", ":")]
    for c, (col, ls) in zip(cons, styles):
        acc = []
        for tk in talkers:
            sr, x = wav(tk, c); on = sa._acoustic_onset(x, sr); seg = x[on: on + int(0.08 * sr)]
            P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 4096)) ** 2
            acc.append(10 * np.log10(np.interp(grid, np.fft.rfftfreq(4096, 1 / sr), P) + 1e-12))
        m = np.convolve(np.pad(np.mean(acc, 0), 3, mode="edge"), np.ones(7) / 7, mode="valid")
        ax.plot(grid / 1000, m - m.max(), color=col, ls=ls, lw=2, label=f"/{C.ipa(c)}/")
    ax.set_xlabel("頻率 (kHz)"); ax.set_ylabel("相對能量 (dB，各自最大值 = 0)"); ax.set_ylim(-48, 3); ax.set_xlim(0.2, 6.5)
    ax.legend(ncol=len(cons), fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title(f"子音段（起始後 80 ms）的平均頻譜，{len(talkers)} 位說話者平均", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "acoustics_consonant_spectra.png")); plt.close(fig)


def fig_pair(plt, a, b, talker, out):
    """一對子音並排：波形、寬頻頻譜圖（6 ms 窗）、80-350 Hz 能量隨時間。從聲學起點前 30 ms 開始取 450 ms。"""
    fig = plt.figure(figsize=(9, 6.6)); gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 3, 1.5], hspace=0.12, wspace=0.07)
    notes = ANNOT.get((a, b), [])
    for j, c in enumerate((a, b)):
        sr, x = wav(talker, c); on = sa._acoustic_onset(x, sr)
        x = x[max(on - int(.03 * sr), 0):][: int(.45 * sr)]; x = x / np.abs(x).max(); t = np.arange(len(x)) / sr * 1000
        a0 = fig.add_subplot(gs[0, j]); a0.plot(t, x, "k", lw=.5); a0.set_xlim(0, 450); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"/{C.ipa(c)}ɑ/", loc="left", fontsize=12)
        for s in ("top", "right"):
            a0.spines[s].set_visible(True)
        a1 = fig.add_subplot(gs[1, j]); nper = int(.006 * sr)
        f, tt, S = spectrogram(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - int(.001 * sr), nfft=512)
        D = 10 * np.log10(S + 1e-12); D -= D.max()
        a1.pcolormesh(tt * 1000, f / 1000, D, vmin=-55, vmax=0, cmap="Greys", shading="auto"); a1.set_ylim(0, 7); a1.set_xlim(0, 450); a1.set_xticklabels([])
        for (x0, x1, y0, y1, txt, col) in notes:
            a1.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.8))
            if j == 0:
                inside = y1 >= 6.5
                a1.text(x0 + 4, y1 - 0.15 if inside else y1 + 0.12, txt, color=col, fontsize=9, va="top" if inside else "bottom",
                        bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
        a2 = fig.add_subplot(gs[2, j]); vr = np.sqrt(np.mean(x[int(.3 * sr):int(.4 * sr)] ** 2))      # 300-400 ms 已在母音裡
        a2.plot(t, 20 * np.log10(band_env(x, sr, 80, 350) / vr), color=C.RED, lw=1.6); a2.set_xlim(0, 450); a2.set_ylim(-50, 5); a2.set_xlabel("時間 (ms)")
        if j == 0:
            a1.set_ylabel("頻率 (kHz)"); a2.set_ylabel("80-350 Hz 能量\n(dB，母音 = 0)")
        else:
            a1.set_yticklabels([]); a2.set_yticklabels([])
    fig.suptitle(f"/{C.ipa(a)}/ 與 /{C.ipa(b)}/（說話者 {talker}；上：波形，中：頻譜圖，下：80-350 Hz 能量）", fontsize=12, y=0.96)
    fig.savefig(os.path.join(out, f"acoustics_pair_{a}_{b}.png"), bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="*", default=["m,n", "f,theta", "s,n", "s,z"], help="要畫頻譜圖的對，例如 m,n s,z（子音名用檔名寫法：theta eth sh zh）")
    ap.add_argument("--talker", default="T01", help="成對頻譜圖用哪位說話者")
    ap.add_argument("--spectra", nargs="*", default=["m", "n", "f", "theta", "s"], help="疊在一張平均頻譜圖上的子音")
    ap.add_argument("--spectra-talkers", nargs="*", default=["T01", "T02", "T05", "T07", "T12"])
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = C.out_dir(a.out)
    rows = [measure(tk, c) for c in A.MN_FEATURES for tk in A.TALKERS if os.path.isfile(os.path.join(C.ROOT, A.sound_file(tk, c)))]
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "consonant_acoustics.csv"), index=False)
    S = T.groupby("consonant", sort=False).agg(n=("talker", "count"), dur_ms=("dur_ms", "mean"), dur_sd=("dur_ms", "std"), low_db=("low_db", "mean"),
                                               low_sd=("low_db", "std"), hi_db=("hi_db", "mean"), hi_sd=("hi_db", "std"),
                                               centroid_hz=("centroid_hz", "mean"), centroid_sd=("centroid_hz", "std"))
    S.to_csv(os.path.join(out, "consonant_acoustics_summary.csv"))
    print("子音段量測，各說話者平均（SD 是說話者之間的）")
    print(S.round(1).rename(index=C.ipa).to_string())
    print("\n清濁擦音對，濁 - 清：")
    for v0, v1 in FRICATIVE_PAIRS:
        print(f"   {C.ipa(v0)}-{C.ipa(v1)}：80-350 Hz {S.low_db[v1] - S.low_db[v0]:+.0f} dB，1.5-7 kHz {S.hi_db[v1] - S.hi_db[v0]:+.1f} dB，重心 {S.centroid_hz[v1] - S.centroid_hz[v0]:+.0f} Hz")
    if not a.no_figures:
        plt = C.setup_plot()
        M = {c: g for c, g in T.groupby("consonant")}
        fig_lowband(plt, M, out); fig_map(plt, M, out); fig_spectra(plt, a.spectra, a.spectra_talkers, out)
        for p in a.pairs:
            x, y = p.split(","); fig_pair(plt, x, y, a.talker, out)
    print(f"\n寫出：{out}/consonant_acoustics.csv、consonant_acoustics_summary.csv" + ("" if a.no_figures else "、acoustics_*.png"))


if __name__ == "__main__":
    main()
