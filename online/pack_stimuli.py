"""Pack online/snr_pilot/stim/*.wav into a few base64 JSON bundles so the experiment can be uploaded through
GitLab's web UI (one file at a time, < 10 MB each) instead of git.

    python online/pack_stimuli.py        # writes snr_pilot/bundles/stim_<group>.json

Groups: hp (headphone tones), and <snr>_s<slot> for the pair syllables (6 bundles of 108 files, ~4.6 MB each).
experiment.js loads the bundles it needs, turns each entry into a Blob URL, and plays that; nothing else changes.
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "snr_pilot", "stim")
OUT = os.path.join(HERE, "snr_pilot", "bundles")


def main():
    os.makedirs(OUT, exist_ok=True)
    cond = json.load(open(os.path.join(HERE, "snr_pilot", "conditions.json"), encoding="utf-8"))
    groups = {"hp": [f for t in cond["headphone_check"] for f in t["files"]]}
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
