"""Build the ONLINE SNR pilot (online/snr_pilot/): a fixed 108-trial plan of the d' consonant test, every syllable
pre-mixed in speech-shaped noise at each pilot SNR, plus the headphone-screening tones.

    python online/make_snr_pilot_stimuli.py              # writes online/snr_pilot/stim/*.wav and conditions.json

Why pre-mixed: the browser cannot run sft_noise. The seeds are the same deterministic hash the lab version uses
(sft_noise.seed_for("perc", trial, 1|2)), so the online listener hears exactly the file the lab listener would.
The PLAN (which pair, which talker, which level on each row) is fixed with PLAN_SEED, like stimuli/sound_pairs.csv:
the set is identical for everyone, only the presentation order is shuffled in the browser.

Headphone screening (Woods, Siegel, Bishop & Dakin, 2017, Atten Percept Psychophys): three 200 Hz tones, one of
them 6 dB quieter (the answer on headphones) and one phase-inverted between the two channels (sounds quietest over
loudspeakers because left and right cancel). Six trials; pass = 5 or more correct. Stereo files, 1 s each.
"""
import csv
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import sft_noise  # noqa: E402
from adaptive_vawm_EN import LEVELS_AUDIO, TALKERS, audio_x, feature_distance, foil_for, load_dprime, sound_file  # noqa: E402

SNRS = (0.0, 6.0, 12.0)
N_PER_LEVEL = 12
P_SAME = 1 / 3
PLAN_SEED = 20261124
OUT = os.path.join(HERE, "snr_pilot")
STIM = os.path.join(OUT, "stim")


def make_plan(rng, table):
    consonants = sorted(table)
    plan = [(x, False) for x in LEVELS_AUDIO for _ in range(N_PER_LEVEL)]
    plan += [(None, True)] * int(round(len(plan) * P_SAME / (1 - P_SAME)))
    rng.shuffle(plan)
    rows = []
    for i, (x, same) in enumerate(plan, 1):
        cons = consonants[int(rng.integers(len(consonants)))]
        talker = TALKERS[int(rng.integers(len(TALKERS)))]
        if same:
            foil, x_actual = cons, None
        else:
            foil, dprime = foil_for(table, cons, x)
            x_actual = audio_x(dprime)
        rows.append(dict(row=i, is_same=int(same), x_planned=x, x=x_actual, consonant=cons, foil=foil,
                         feature_distance=None if same else feature_distance(cons, foil), talker=talker,
                         snd1=sound_file(talker, cons), snd2=sound_file(talker, foil),
                         seed1=sft_noise.seed_for("perc", i, 1), seed2=sft_noise.seed_for("perc", i, 2)))
    return rows


def write_stereo(path, sr, left, right):
    y = np.stack([left, right], axis=1)
    y = np.clip(y, -0.95, 0.95)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def headphone_check(rng, n_trials=6, sr=44100, dur=1.0, f=200.0):
    """Returns the trial list; writes 3 tones per trial. Positions shuffled per trial; the file records the answer."""
    t = np.arange(int(sr * dur)) / sr
    env = np.minimum(1.0, np.minimum(t, dur - t) / 0.05)          # 50 ms ramps
    tone = 0.3 * np.sin(2 * np.pi * f * t) * env
    trials = []
    for k in range(1, n_trials + 1):
        kinds = ["normal", "quiet", "antiphase"]
        rng.shuffle(kinds)
        files = []
        for pos, kind in enumerate(kinds, 1):
            if kind == "normal":
                L, R = tone, tone
            elif kind == "quiet":
                L, R = tone * 10 ** (-6 / 20), tone * 10 ** (-6 / 20)
            else:
                L, R = tone, -tone
            fn = f"hp{k}_{pos}.wav"
            write_stereo(os.path.join(STIM, fn), sr, L, R)
            files.append(fn)
        trials.append(dict(trial=k, files=files, answer=kinds.index("quiet") + 1, kinds=kinds))
    return trials


def main():
    os.makedirs(STIM, exist_ok=True)
    table = load_dprime()
    rng = np.random.default_rng(PLAN_SEED)
    rows = make_plan(rng, table)
    n = 0
    for r in rows:
        for snr in SNRS:
            for slot, (wav, seed) in enumerate(((r["snd1"], r["seed1"]), (r["snd2"], r["seed2"])), 1):
                fn = f"r{r['row']:03d}_s{slot}_snr{snr:g}.wav"
                sft_noise.rebuild(wav, snr, seed, os.path.join(STIM, fn))
                r[f"file{slot}_snr{snr:g}"] = "stim/" + fn
                n += 1
    hp = headphone_check(rng)
    cond = dict(snrs=list(SNRS), levels=list(LEVELS_AUDIO), n_per_level=N_PER_LEVEL, plan_seed=PLAN_SEED,
                trials=rows, headphone_check=hp, built="2026-11-24")
    with open(os.path.join(OUT, "conditions.json"), "w", encoding="utf-8") as f:
        json.dump(cond, f, indent=1)
    with open(os.path.join(OUT, "conditions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    mb = sum(os.path.getsize(os.path.join(STIM, x)) for x in os.listdir(STIM)) / 1e6
    print(f"{len(rows)} rows x {len(SNRS)} SNRs = {n} syllable files + {len(hp) * 3} headphone tones, {mb:.1f} MB in {STIM}")


if __name__ == "__main__":
    main()
