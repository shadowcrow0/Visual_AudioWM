"""Step 3. Find the target syllable (the last voiced stretch after the last silent gap = stop closure) in each
verified clip and measure it. t = 0 is where voicing starts after the consonant.

    python measure.py --out out       (reads out/verified.json, writes out/measures.json, prints a table)

Works for targets that begin with a stop or fricative. Nasal- or vowel-initial targets (ㄇㄛ, ㄨㄛ) have no
silent gap before them and are not segmented correctly, so they are not in the default item list.
"""
import json, argparse, collections, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt
def env(x, sr, lo, hi, win=0.010):
    y = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x); n = int(win*sr)
    return np.sqrt(np.convolve(y**2, np.ones(n)/n, mode="same") + 1e-12)
def landmarks(f):
    """samples, rate, index where the final syllable's voicing starts, index where it ends"""
    x, sr = sf.read(f); x = x/np.abs(x).max(); e = env(x, sr, 150, 1200); on = e > 0.12*e[int(-.6*sr):].max()
    i = len(x) - 1
    while i > 0 and not on[i]: i -= 1
    end = i
    while i > 0 and (on[i] or on[max(i - int(.015*sr), 0): i].any()): i -= 1          # bridge dips shorter than 15 ms
    return x, sr, i + 1, end
def band_centroid(x, sr, t, lo=200, hi=1500, win=0.030):
    """centre of gravity (Hz) of the power spectrum between lo and hi in a 30 ms window centred at t (s).
    It rises as the mouth opens ([u] -> [o]) and needs no formant tracking."""
    a = int((t - win/2)*sr); seg = x[max(a, 0): a + int(win*sr)]; seg = seg*np.hanning(len(seg))
    P = np.abs(np.fft.rfft(seg, 8192))**2; fr = np.fft.rfftfreq(8192, 1/sr); k = (fr >= lo) & (fr <= hi)
    return float((fr[k]*P[k]).sum()/P[k].sum())
def formants(x, sr, times):
    """F1, F2 (Hz): Praat Burg, median over five ceilings (settings for a male voice)"""
    import parselmouth
    snd = parselmouth.Sound(x, sr); out = []
    for ceil in (4500, 4750, 5000, 5250, 5500):
        fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceil, window_length=0.025)
        out.append([[fm.get_value_at_time(k, t) for k in (1, 2)] for t in times])
    return np.nanmedian(np.array(out, float), axis=0)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); a = ap.parse_args(); rows = []
    for r in [r for r in json.load(open(f"{a.out}/verified.json")) if r["ok"]]:
        x, sr, i0, i1 = landmarks(f"{a.out}/{r['file']}"); dur = (i1 - i0)/sr; tt = i0/sr + np.array([.15, .5, .85])*dur; F = formants(x, sr, tt)
        rows.append(dict(label=r["label"], seed=r["seed"], file=r["file"], on=i0/sr, off=i1/sr, dur=dur,
                         F1=[float(v) for v in F[:, 0]], F2=[float(v) for v in F[:, 1]], cog=[band_centroid(x, sr, t) for t in tt]))
    json.dump(rows, open(f"{a.out}/measures.json", "w"), ensure_ascii=False, indent=1)
    G = collections.defaultdict(list)
    for r in rows: G[r["label"].split("_")[0]].append(r)
    print("syll   n  dur(ms)  | F1 at 15/50/85%  | F2 at 15/50/85%    | centroid 0.2-1.5 kHz at 15/50/85%")
    for k, v in G.items():
        A = lambda key: np.nanmean([r[key] for r in v], 0)
        print(f"{k:5s} {len(v):2d}  {1000*np.mean([r['dur'] for r in v]):4.0f}     | " + " ".join(f"{q:4.0f}" for q in A("F1")) + "   | " + " ".join(f"{q:5.0f}" for q in A("F2")) + "   | " + " ".join(f"{q:4.0f}" for q in A("cog")))
