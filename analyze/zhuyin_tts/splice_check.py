"""Step 6. Look at every spliced clip two ways: what a speech recogniser hears, and how the sound moves.

    python splice_check.py --out out  (reads out/splice/splice_log.json, writes out/splice/splice_results.json)

"heard" is whatever the recogniser (faster-whisper "small") writes after 「這個字唸」. It is a crude machine
listener, not a person: listen.py collects human answers. The curve is the 0.2-1.5 kHz centre of gravity every
5 ms from voicing onset: about 360 Hz while the lips are round and nearly closed (ㄨ), 450-550 Hz for ㄛ.
"""
import re, json, argparse, collections, numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from pypinyin import lazy_pinyin
from measure import landmarks, band_centroid
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); a = ap.parse_args()
log = json.load(open(f"{a.out}/splice/splice_log.json")); asr = WhisperModel("small", device="cpu", compute_type="int8")
for r in log:
    x, sr, i0, i1 = landmarks(f"{a.out}/splice/{r['file']}")
    segs, _ = asr.transcribe(resample_poly(x, 16000, sr).astype(np.float32), language="zh", beam_size=5)
    txt = re.sub(r"[^一-鿿]", "", "".join(s.text for s in segs)); m = re.search(r"[唸念](.*)$", txt)
    heard = m.group(1) if m else txt[-1:]                              # what came after the carrier; may be more than one character
    ts = np.arange(10, (i1 - i0)/sr*1000 - 5, 5)
    r.update(asr=txt, heard=heard, heard_pinyin=" ".join(lazy_pinyin(heard)), dur_ms=(i1 - i0)/sr*1000, onset_s=i0/sr,
             t_ms=[float(t) for t in ts], cog=[band_centroid(x, sr, i0/sr + t/1000) for t in ts])
    r["cog_first60"] = float(np.mean([c for t, c in zip(ts, r["cog"]) if t <= 60]))              # mean over the first 60 ms after the consonant
    print(r["file"], txt, "->", r["heard_pinyin"], flush=True)
json.dump(log, open(f"{a.out}/splice/splice_results.json", "w"), ensure_ascii=False, indent=1)
print("\nkind  L     n  length(ms)  centre 10-60 ms (Hz)  heard as (count)")
for kind in dict.fromkeys(r["kind"] for r in log):
    for L in dict.fromkeys(r["L"] for r in log if r["kind"] == kind):
        v = [r for r in log if r["kind"] == kind and r["L"] == L]; c = collections.Counter(r["heard_pinyin"] or "(nothing)" for r in v)
        print(f"{kind:4s}  {str(L):5s} {len(v):2d}  {np.mean([r['dur_ms'] for r in v]):5.0f}±{np.std([r['dur_ms'] for r in v]):3.0f}   {np.mean([r['cog_first60'] for r in v]):5.0f}±{np.std([r['cog_first60'] for r in v]):3.0f}             "
              + ", ".join(f"{k} {n}" for k, n in c.most_common()))
