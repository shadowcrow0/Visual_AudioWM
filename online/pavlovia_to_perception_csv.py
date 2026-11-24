"""Convert a Pavlovia / jsPsych data file of the online SNR pilot into the lab's sound-perception format.

    python online/pavlovia_to_perception_csv.py <id>_snr_pilot.csv [more files...]

Writes, for every SNR block in the file, data/<participant>_sound_perception_dprime_snr<dB>.csv with the columns of
sound_perception_EN.py (participant, trial, axis, is_same, x_planned, x, consonant, foil, count, feature_distance,
talker, snd1, snd2, snd_snr_db, snd_seed_1, snd_seed_2, key, rt, correct) and the matching .json summary (hit rate
and d' per level, false-alarm rate, headphone-check result), so the pilot analyses run unchanged.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COLS = ["participant", "trial", "axis", "is_same", "x_planned", "x", "consonant", "foil", "count", "feature_distance",
        "talker", "snd1", "snd2", "snd_snr_db", "snd_seed_1", "snd_seed_2", "key", "rt", "correct"]


def val(v):
    """jsPsych writes null as the string 'null' (and sometimes '') in its CSV."""
    return None if v in (None, "", "null") else v


def z(p, n):
    from statistics import NormalDist
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    return NormalDist().inv_cdf(p)


def convert(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [r for r in rows if r.get("task") == "pair"]
    if not pairs:
        print(f"{path}: no pair trials"); return
    pid = pairs[0]["participant"]
    hp = [r for r in rows if r.get("task") == "hpcheck"]
    hp_correct = sum(int(float(val(r["correct"]) or 0)) for r in hp)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for snr in sorted({float(r["snr_db"]) for r in pairs}):
        blk = sorted([r for r in pairs if float(r["snr_db"]) == snr], key=lambda r: int(r["trial"]))
        out = []
        for r in blk:
            rt = val(r["rt"]); key = val(r["key"]); corr = val(r["correct"])
            out.append(dict(participant=pid, trial=int(r["trial"]), axis="dprime", is_same=int(r["is_same"]),
                            x_planned=val(r["x_planned"]), x=val(r["x"]), consonant=r["consonant"], foil=r["foil"], count=None,
                            feature_distance=val(r["feature_distance"]), talker=r["talker"], snd1=r["snd1"], snd2=r["snd2"],
                            snd_snr_db=snr, snd_seed_1=r["seed1"], snd_seed_2=r["seed2"], key=key,
                            rt=float(rt) / 1000.0 if rt is not None else None, correct=int(float(corr)) if corr is not None else None))
        base = os.path.join(ROOT, "data", f"{pid}_sound_perception_dprime_snr{snr:g}")
        with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(out)
        same = [r for r in out if r["is_same"] and r["key"] is not None]
        fa = sum(1 for r in same if r["key"] == "n") / max(len(same), 1)
        summary = dict(participant=pid, axis="dprime", snr_db=snr, source="pavlovia", n_trials=len(out),
                       n_no_response=sum(1 for r in out if r["key"] is None), same_trials=len(same), false_alarm=fa,
                       headphone_check=dict(correct=hp_correct, n=len(hp)), levels={})
        for x in sorted({float(r["x_planned"]) for r in out if r["x_planned"]}):
            sub = [r for r in out if not r["is_same"] and r["x_planned"] and float(r["x_planned"]) == x and r["key"] is not None]
            if not sub:
                continue
            hit = sum(r["correct"] for r in sub) / len(sub)
            rts = sorted(r["rt"] for r in sub)
            summary["levels"][f"{x:g}"] = dict(n=len(sub), hit=hit, dprime=z(hit, len(sub)) - z(fa, max(len(same), 1)), median_rt=rts[len(rts) // 2])
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"{pid} SNR {snr:+g} dB: {len(out)} trials, FA {fa:.2f}, hits " +
              " ".join(f"{k}:{v['hit']:.2f}" for k, v in summary["levels"].items()) + f"  -> {os.path.relpath(base, ROOT)}.csv/.json")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
