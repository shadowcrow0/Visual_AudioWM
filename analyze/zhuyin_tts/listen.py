"""Optional. A listening test in the terminal: the recogniser is a crude machine listener, this collects yours.

    python listen.py --out out        (plays the spliced 撥 clips in random order, writes out/splice/listen_responses.csv)

Each clip is the sentence 「這個字唸，X。」. You are asked what X sounded like. Mixed together, unlabelled: the
untouched clip, clips with 40-120 ms of ㄨ inserted ("bo"), and clips made longer with ㄛ instead ("boctl").
Answers are appended after every trial, so quitting early loses nothing.
"""
import os, csv, json, time, random, shutil, argparse, subprocess
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--pairs", type=int, default=4, help="source pairs per condition")
ap.add_argument("--kinds", nargs="*", default=["bo", "boctl"], help="bo, boctl, duo (see splice.py)"); ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--no-audio", action="store_true", help="do not play anything (only for testing the script)")
a = ap.parse_args()
PLAYERS = (["pw-play"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"])
player = next((p for p in PLAYERS if shutil.which(p[0])), None)
if player is None and not a.no_audio:
    raise SystemExit("no audio player found. Install one of: pipewire (pw-play), libpulse (paplay), alsa-utils (aplay), ffmpeg (ffplay)")
log = [r for r in json.load(open(f"{a.out}/splice/splice_log.json")) if r["kind"] in a.kinds]
trials = []
for key in dict.fromkeys((r["kind"], r["L"]) for r in log):     # the same number of clips for every condition
    if key == ("boctl", 0): continue                            # identical to ("bo", 0): the untouched clip
    v = [r for r in log if (r["kind"], r["L"]) == key]; k = min(a.pairs, len(v))
    trials += [v[round(i*(len(v) - 1)/max(k - 1, 1))] for i in range(k)]*a.reps     # spread over the list so that different source clips are used
random.Random(a.seed).shuffle(trials)
CHOICES = {"1": "one syllable, like 撥 (ㄅㄛ)", "2": "one syllable with an extra ㄨ before ㄛ (ㄅㄨㄛ)", "3": "two syllables (like 不喔)", "4": "something else"} if "duo" not in a.kinds else \
          {"1": "one syllable, like 多 (ㄉㄨㄛ)", "2": "one syllable with a longer ㄨ", "3": "two syllables (like 督喔)", "4": "something else"}
path = f"{a.out}/splice/listen_responses.csv"; new = not os.path.exists(path)
print(f"{len(trials)} trials. After each clip type 1-4 and Enter;  r = play again,  q = stop.\n" + "\n".join(f"  {k} = {v}" for k, v in CHOICES.items()))
with open(path, "a", newline="") as fh:
    w = csv.writer(fh)
    if new: w.writerow(["time", "trial", "file", "kind", "L", "response", "response_label", "replays"])
    for i, r in enumerate(trials, 1):
        replays = -1
        while True:
            replays += 1
            if not a.no_audio: subprocess.run(player + [f"{a.out}/splice/{r['file']}"], check=False)
            try: ans = input(f"[{i}/{len(trials)}] ").strip().lower()
            except EOFError: ans = "q"
            if ans in CHOICES or ans == "q": break
        if ans == "q": break
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), i, r["file"], r["kind"], r["L"], ans, CHOICES[ans], replays]); fh.flush()
print("answers saved in", path)
