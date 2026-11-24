"""Step 5. Build "ㄅㄨㄛ" by splicing: nobody (and no model) says it, so cut a real ㄨ in between ㄅ and ㄛ.

    python splice.py --out out        (needs out/measures.json; writes out/splice/*.wav and splice_log.json)

    result(L) = B[sentence + consonant, up to where the vowel starts] + D[L ms of steady vowel] + B[its own vowel]

kind "bo"     B = 撥 (ㄅㄛ),   D = ㄨ from a tone-1 ㄨ syllable (督 姑 夫 撲)   -> ㄅ + ㄨ + ㄛ
kind "boctl"  B = 撥,          D = ㄛ from another ㄅㄛ clip                     -> same extra length and same two joins,
                                                                                  but no ㄨ: separates "longer" from "has a ㄨ"
kind "duo"    B = 多 (ㄉㄨㄛ), D = ㄨ as in "bo"                                -> the same extra ㄨ after a non-labial
L = 0 is B untouched. 不 / 布 (ㄅㄨˋ) would be the natural donor but tone 4 is too short (under 100 ms) to give 120 ms of ㄨ.
These clips are constructed: they are not how any speaker or the model says anything.
"""
import os, json, argparse, numpy as np, soundfile as sf, parselmouth
from measure import landmarks
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--n", type=int, default=4, help="clips per role (n x n pairs for every L)")
ap.add_argument("--L", type=int, nargs="*", default=[40, 80, 120], help="ms of vowel inserted")
a = ap.parse_args(); os.makedirs(f"{a.out}/splice", exist_ok=True)
M = [r for r in json.load(open(f"{a.out}/measures.json")) if 0.10 <= r["dur"] <= 0.32]      # same filter as figs.py: a badly segmented clip cannot be cut
syl = lambda r: r["label"].split("_")[0]
FADE, SKIP = 0.010, 0.030          # s: cross-fade at each join; start of the donor vowel skipped (its own consonant still colours it)

def f0(x, sr, t0, t1):
    """median pitch (Hz) between t0 and t1 seconds"""
    p = parselmouth.Sound(x, sr).to_pitch(time_step=0.005, pitch_floor=70, pitch_ceiling=350); f = p.selected_array["frequency"]; k = (p.xs() >= t0) & (p.xs() <= t1) & (f > 0)
    return float(np.median(f[k])) if k.any() else float("nan")

def insert(b, d, L):
    """b, d = (samples, rate, vowel start, vowel end) from landmarks(). Returns (samples, pitch of the insert, pitch of B's vowel start) or None."""
    xb, sr, ob, eb = b; xd, _, od, ed = d; w, n, P = int(FADE*sr), int(L/1000*sr), int(sr/110); s0 = od + int(SKIP*sr)
    if s0 + P + n + w > ed: return None                                    # donor vowel shorter than L
    # Two voiced sounds joined out of phase give a click and a false notch in the spectrum, so each cut is slid by up
    # to one pitch period (P) to where the two waveforms line up best.
    d0 = max(range(s0, s0 + P), key=lambda c: float(np.dot(xb[ob:ob + w], xd[c:c + w]))); d1 = d0 + n
    cb = max(range(ob, ob + P), key=lambda c: float(np.dot(xd[d1:d1 + w], xb[c:c + w])))
    up = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, w)); dn = 1 - up
    y = np.concatenate([xb[:ob], xb[ob:ob + w]*dn + xd[d0:d0 + w]*up, xd[d0 + w:d1], xd[d1:d1 + w]*dn + xb[cb:cb + w]*up, xb[cb + w:]])
    return y, f0(xd, sr, d0/sr, d1/sr), f0(xb, sr, ob/sr, ob/sr + 0.05)

longest = lambda rows: sorted(rows, key=lambda r: -r["dur"])[:a.n]
U = longest([r for r in M if syl(r) in ("du", "gu", "fu", "pu") and r["cog"][1] < 430])     # the donor must really sit low (about 360 Hz), else it is no ㄨ
O = longest([r for r in M if syl(r) == "bo"])
log = []
for kind, B, D in (("bo", [r for r in M if r["label"] == "bo_撥"][:a.n], U), ("boctl", [r for r in M if r["label"] == "bo_撥"][:a.n], O), ("duo", [r for r in M if r["label"] == "duo_多"][:a.n], U)):
    if not B or not D: print(f"skip {kind}: missing source clips"); continue
    for rb in B:
        b = landmarks(f"{a.out}/{rb['file']}"); name = f"{kind}_L000_{rb['label']}s{rb['seed']}.wav"
        sf.write(f"{a.out}/splice/{name}", 0.9*b[0], b[1]); log.append(dict(file=name, kind=kind, L=0, b_file=rb["file"], d_file=None))
        for rd in D:
            if rd["file"] == rb["file"]: continue                           # never lengthen a clip with itself
            d = landmarks(f"{a.out}/{rd['file']}")
            for L in a.L:
                out = insert(b, d, L)
                if out is None: continue
                y, f_ins, f_b = out; name = f"{kind}_L{L:03d}_{rb['label']}s{rb['seed']}_{rd['label']}s{rd['seed']}.wav"
                sf.write(f"{a.out}/splice/{name}", 0.9*y/np.abs(y).max(), b[1])
                log.append(dict(file=name, kind=kind, L=L, b_file=rb["file"], d_file=rd["file"], insert_start_s=b[2]/b[1], f0_insert=f_ins, f0_b=f_b))
json.dump(log, open(f"{a.out}/splice/splice_log.json", "w"), ensure_ascii=False, indent=1)
print(len(log), "clips in", f"{a.out}/splice")
for kind in dict.fromkeys(r["kind"] for r in log):
    d = np.array([[r["f0_insert"], r["f0_b"]] for r in log if r["kind"] == kind and r["L"] > 0], float)
    if len(d): print(f"{kind}: {len(d)} spliced clips; pitch of the insert {np.nanmedian(d[:, 0]):.0f} Hz, of the vowel it joins {np.nanmedian(d[:, 1]):.0f} Hz, median |difference| {np.nanmedian(np.abs(d[:, 0] - d[:, 1])):.0f} Hz")
