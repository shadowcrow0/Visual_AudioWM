"""Step 2. Check every clip with a speech recogniser (faster-whisper "small"): the last syllable of the
transcript must have the target's toneless pinyin. Clips that fail are not used later.

    python verify.py --out out        (reads out/log.json, writes out/verified.json)
"""
import re, json, argparse, collections, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from pypinyin import pinyin, Style
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); a = ap.parse_args()
log = json.load(open(f"{a.out}/log.json")); m = WhisperModel("small", device="cpu", compute_type="int8")
for r in log:
    x, sr = sf.read(f"{a.out}/{r['file']}"); y = resample_poly(x, 16000, sr).astype(np.float32)        # pass samples, not a path: avoids the audio-decoder dependency
    segs, _ = m.transcribe(y, language="zh", beam_size=5); txt = re.sub(r"[^一-鿿]", "", "".join(s.text for s in segs))
    cands = set(sum(pinyin(txt[-1:], style=Style.NORMAL, heteronym=True), [])) if txt else set()
    r.update(asr=txt, ok=r["label"].split("_")[0] in cands); print(r["file"], txt, "OK" if r["ok"] else "--", flush=True)
json.dump(log, open(f"{a.out}/verified.json", "w"), ensure_ascii=False, indent=1)
c = collections.Counter((r["label"], r["ok"]) for r in log)
for lab in dict.fromkeys(r["label"] for r in log): print(lab, "verified", c[(lab, True)], "/", c[(lab, True)] + c[(lab, False)])
