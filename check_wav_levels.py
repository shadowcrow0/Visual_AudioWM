"""Level check of the 192 consonant tokens (stimuli/T01..T12_<cons>3.wav): is a consonant systematically quieter?

    python check_wav_levels.py            # writes review/data/wav_levels.csv and prints the per-consonant table

Per file: duration, whole-file RMS, voiced-segment RMS (snr_audio._voiced_rms: periodic windows only), RMS of the first
80 ms after the acoustic onset (the consonant portion for a CV token), peak. All in dB re full scale.
Written 2026-11-10 after the pilot (review/pilot_2026-11-10_sound_pairs_perception.md) missed mostly pairs involving
f / v / θ / ʃ / ʒ / m / n — the question is whether those tokens are quieter, not more similar.
"""
import csv
import glob
import os
import re

import numpy as np

import snr_audio as sa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review", "data", "wav_levels.csv")
IPA = {"theta": "θ", "eth": "ð", "sh": "ʃ", "zh": "ʒ"}


def db(v):
    return 20 * np.log10(max(float(v), 1e-9))


def measure(path):
    sr, x = sa._read_wav(path)
    on = sa._acoustic_onset(x, sr)
    seg = x[on:on + int(0.08 * sr)]
    try:
        voiced = sa._voiced_rms(x, sr)
    except ValueError:
        voiced = np.nan
    return dict(duration_ms=round(1000 * len(x) / sr), onset_ms=round(1000 * on / sr),
                rms_db=round(db(sa._rms(x)), 1), voiced_db=round(db(voiced), 1) if np.isfinite(voiced) else "",
                onset80_db=round(db(sa._rms(seg)), 1), peak_db=round(db(np.abs(x).max()), 1))


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "stimuli", "T??_*3.wav"))):
        m = re.match(r"(T\d\d)_(.+)3\.wav", os.path.basename(p))
        rows.append(dict(file=os.path.relpath(p, HERE), talker=m.group(1), cons=m.group(2), **measure(p)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cons = sorted({r["cons"] for r in rows}, key=lambda c: np.mean([r["onset80_db"] for r in rows if r["cons"] == c]))
    print(f"{'cons':>5} {'n':>2} {'file RMS':>9} {'voiced':>7} {'onset 80ms':>11} {'peak':>6} {'dur ms':>7}   (mean over talkers, dB re FS)")
    for c in cons:
        rr = [r for r in rows if r["cons"] == c]
        v = [r["voiced_db"] for r in rr if r["voiced_db"] != ""]
        print(f"{IPA.get(c, c):>5} {len(rr):>2} {np.mean([r['rms_db'] for r in rr]):>9.1f} {np.mean(v) if v else float('nan'):>7.1f} "
              f"{np.mean([r['onset80_db'] for r in rr]):>11.1f} {np.mean([r['peak_db'] for r in rr]):>6.1f} {np.mean([r['duration_ms'] for r in rr]):>7.0f}")
    print(f"\nwrote {OUT}: {len(rows)} files")
    return rows


if __name__ == "__main__":
    main()
