"""Pack online/<pilot>/stim/*.wav into a few base64 JSON bundles so the experiment can be uploaded through
GitLab's web UI (one file at a time, < 10 MB each) instead of git.

    python online/pack_stimuli.py                # snr_pilot (default): writes snr_pilot/bundles/stim_<group>.json
    python online/pack_stimuli.py calib_pilot    # calib_pilot/bundles/stim_hp.json + stim_p1..p6.json

Groups, snr_pilot: hp (headphone tones), and <snr>_s<slot> for the pair syllables (6 bundles of 108 files, ~4 MB each).
Groups, calib_pilot (one SNR): hp, and p1..p6 = both syllables of 54 consecutive plan rows each (108 files, ~4 MB).
experiment.js loads the bundles it needs, turns each entry into a Blob URL, and plays that; nothing else changes.
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = sys.argv[1] if len(sys.argv) > 1 else "snr_pilot"
STIM = os.path.join(HERE, PILOT, "stim")
OUT = os.path.join(HERE, PILOT, "bundles")
ROWS_PER_PART = 54


def main():
    os.makedirs(OUT, exist_ok=True)
    cond = json.load(open(os.path.join(HERE, PILOT, "conditions.json"), encoding="utf-8"))
    groups = {"hp": [f for t in cond["headphone_check"] for f in t["files"]]}
    if cond.get("design") == "calib_pilot":
        for k in range(0, len(cond["trials"]), ROWS_PER_PART):
            groups[f"p{k // ROWS_PER_PART + 1}"] = [os.path.basename(t[f"file{slot}"]) for t in cond["trials"][k:k + ROWS_PER_PART] for slot in (1, 2)]
    else:
        for snr in cond["snrs"]:
            for slot in (1, 2):
                groups[f"snr{snr:g}_s{slot}"] = [os.path.basename(t[f"file{slot}_snr{snr:g}"]) for t in cond["trials"]]
    index = {}
    for g, files in groups.items():
        data = {f: base64.b64encode(open(os.path.join(STIM, f), "rb").read()).decode("ascii") for f in files}
        path = os.path.join(OUT, f"stim_{g}.json")
        json.dump(data, open(path, "w"), separators=(",", ":"))
        index[g] = dict(file=f"bundles/stim_{g}.json", n=len(files), mb=round(os.path.getsize(path) / 1e6, 2))
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    for g, v in index.items():
        print(f"{v['file']:28s} {v['n']:4d} files  {v['mb']:5.2f} MB")


if __name__ == "__main__":
    main()
