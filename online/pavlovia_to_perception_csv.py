"""Convert a Pavlovia / jsPsych data file of the online SNR pilot into the lab's sound-perception format.

    python online/pavlovia_to_perception_csv.py <id>_snr_pilot.csv [more files...] [--id NEWID] [--force]

Handles both online designs (the data file says which in its `design` column; absent = snr_pilot):
online/snr_pilot/   (three SNR blocks, levels on the d' table) and online/calib_pilot/ (one SNR, trials spread along
the acoustic axis). For calib_pilot the csv gets extra columns after the lab's (row, group, pair_type, voicing_only,
x_acoustic, x_acoustic_pair, bin), `axis` reads "acoustic", and the json keeps the questionnaire and the order seed.
The file name pattern is the same for both so that analyze/ reads either; an existing file is never replaced by
different content unless --force is given (use --id for a second session of the same person).

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


EXTRA = ["row", "group", "pair_type", "voicing_only", "x_acoustic", "x_acoustic_pair", "bin"]     # calib_pilot only, appended after COLS
PLACEHOLDER_IDS = ("", "test", "anon")


def write_if_same_or_new(path, text, force):
    """Never silently replace another session's data: an existing file with different content needs --force."""
    if os.path.exists(path) and not force:
        if open(path, encoding="utf-8", newline="").read() == text:          # newline="": compare the bytes as written
            return
        raise SystemExit(f"{os.path.relpath(path, ROOT)} already exists with different content. If this is a second session of the same "
                         f"person, convert it under another id (--id NEWID); to replace the file, add --force.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)


def convert(path, new_id=None, force=False):
    import io
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pairs = [r for r in rows if r.get("task") == "pair"]; stored = any(r.get("trial_type") == "pavlovia" for r in rows)
    if not pairs:
        print(f"{path}: no pair trials"); return
    design = val(pairs[0].get("design")) or "snr_pilot"           # online/snr_pilot/ predates the design column
    pid = new_id or pairs[0]["participant"]
    if pid.strip().lower() in PLACEHOLDER_IDS:
        raise SystemExit(f"{path}: the participant id is the placeholder {pid!r}; give the real one with --id")
    hp = [r for r in rows if r.get("task") == "hpcheck"]
    hp_correct = sum(int(float(val(r["correct"]) or 0)) for r in hp)
    # The browser stores 64-bit noise seeds as doubles, so snr_pilot data files have them rounded (…9237 -> …9200) and a
    # rounded seed cannot rebuild the stimulus. The exact seeds are in the plan the page was built from: take them by row.
    # calib_pilot writes its seeds as strings; they are still checked against its plan.
    plan = os.path.join(HERE, design, "conditions.json")
    exact = {int(t["row"]): (str(t["seed1"]), str(t["seed2"])) for t in json.load(open(plan, encoding="utf-8"))["trials"]} if os.path.exists(plan) else {}
    calib = design == "calib_pilot"
    cols = COLS + EXTRA if calib else COLS
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for snr in sorted({float(r["snr_db"]) for r in pairs}):
        blk = sorted([r for r in pairs if float(r["snr_db"]) == snr], key=lambda r: int(r["trial"]))
        out, repaired = [], 0
        for r in blk:
            rt = val(r["rt"]); key = val(r["key"]); corr = val(r["correct"])
            s1, s2 = exact.get(int(r["row"]), (r["seed1"], r["seed2"]))
            repaired += (str(r["seed1"]), str(r["seed2"])) != (s1, s2)
            # x / count = the pair's value in the d' table: the lab script writes it there and the pair analyses read it.
            mn = val(r["mn_dprime"]) if calib else val(r["x"])
            row = dict(participant=pid, trial=int(r["trial"]), axis="acoustic" if calib else "dprime", is_same=int(r["is_same"]),
                       x_planned=None if calib else val(r["x_planned"]), x=mn, consonant=r["consonant"], foil=r["foil"], count=mn,
                       feature_distance=val(r["feature_distance"]), talker=r["talker"], snd1=r["snd1"], snd2=r["snd2"],
                       snd_snr_db=snr, snd_seed_1=s1, snd_seed_2=s2, key=key,
                       rt=float(rt) / 1000.0 if rt is not None else None, correct=int(float(corr)) if corr is not None else None)
            if calib:
                row.update({k: val(r[k]) for k in EXTRA})
            out.append(row)
        base = os.path.join(ROOT, "data", f"{pid}_sound_perception_dprime_snr{snr:g}")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\r\n"); w.writeheader(); w.writerows(out)
        write_if_same_or_new(base + ".csv", buf.getvalue(), force)
        same = [r for r in out if r["is_same"] and r["key"] is not None]
        fa = sum(1 for r in same if r["key"] == "n") / max(len(same), 1)
        summary = dict(participant=pid, axis="acoustic" if calib else "dprime", snr_db=snr,
                       source="pavlovia" if stored else "local download",   # no rows of the Pavlovia bridge = the page's own "Download the data file" link (run without Pavlovia)
                       n_trials=len(out), n_no_response=sum(1 for r in out if r["key"] is None), same_trials=len(same), false_alarm=fa, headphone_check=dict(correct=hp_correct, n=len(hp)), levels={})
        if calib:
            # the questionnaire and the order seed travel with the data; the levels are the 1-dB bins of the acoustic axis
            first = pairs[0]
            summary.update(design=design, plan_seed=val(first.get("plan_seed")), order_seed=val(first.get("order_seed")),
                           native_language=val(first.get("native_language")), english_at_home=val(first.get("english_at_home")),
                           english_start_age=val(first.get("english_start_age")), user_agent=val(first.get("user_agent")),
                           test_maxtrials=val(first.get("test_maxtrials")), seeds_repaired_from_plan=repaired, source_file=os.path.basename(path))
            for b in sorted({int(r["bin"]) for r in out if r["bin"] is not None}):
                sub = [r for r in out if not r["is_same"] and r["bin"] is not None and int(r["bin"]) == b and r["key"] is not None]
                if not sub:
                    continue
                hit = sum(r["correct"] for r in sub) / len(sub); rts = sorted(r["rt"] for r in sub)
                summary["levels"][f"bin{b}"] = dict(n=len(sub), hit=hit, x_acoustic_mean=sum(float(r["x_acoustic"]) for r in sub) / len(sub),
                                                    dprime=z(hit, len(sub)) - z(fa, max(len(same), 1)), median_rt=rts[len(rts) // 2])
        else:
            for x in sorted({float(r["x_planned"]) for r in out if r["x_planned"]}):
                sub = [r for r in out if not r["is_same"] and r["x_planned"] and float(r["x_planned"]) == x and r["key"] is not None]
                if not sub:
                    continue
                hit = sum(r["correct"] for r in sub) / len(sub)
                rts = sorted(r["rt"] for r in sub)
                summary["levels"][f"{x:g}"] = dict(n=len(sub), hit=hit, dprime=z(hit, len(sub)) - z(fa, max(len(same), 1)), median_rt=rts[len(rts) // 2])
        write_if_same_or_new(base + ".json", json.dumps(summary, indent=2, ensure_ascii=not calib), force)
        print(f"{pid} SNR {snr:+g} dB: {len(out)} trials, FA {fa:.2f}, hits " +
              " ".join(f"{k}:{v['hit']:.2f}" for k, v in summary["levels"].items()) + f"  -> {os.path.relpath(base, ROOT)}.csv/.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pavlovia / jsPsych data file(s) of an online pilot -> the lab's sound-perception format.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--id", default=None, help="store the session under this participant id (required when the file carries a placeholder)")
    ap.add_argument("--force", action="store_true", help="replace existing output files whose content differs")
    a = ap.parse_args()
    for p in a.files:
        convert(p, a.id, a.force)
