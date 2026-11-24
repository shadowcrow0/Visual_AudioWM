"""Step 1. Synthesize single Mandarin syllables with BlueMagpie-TTS (OpenFormosa), one target per carrier sentence.

    python synth.py --out out --seeds 8

Each target character X is spoken as 「這個字唸，X。」 (a bare character or bare Bopomofo makes the model babble).
Every sample uses torch.manual_seed(1000 + k), so any clip can be rebuilt. Speaker: the bundled, authorised
hung_yi_lee voice. The first run downloads the model (about 8 GB) from Hugging Face.
Existing wav files are kept, so an interrupted run can simply be started again.
"""
import os, json, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--model-dir", default=os.environ.get("BLUEMAGPIE_DIR"), help="local copy of OpenFormosa/BlueMagpie-TTS (default: download)")
ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
ap.add_argument("--items", nargs="*", default="bo_撥 bo_玻 bo_菠 po_坡 po_潑 duo_多 tuo_拖 guo_鍋 guo_郭 pu_撲 fu_夫 du_督 gu_姑".split(),
                help="pinyin_character pairs; all tone 1 so that pitch is the same across items")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import torch, soundfile as sf
from transformers import PreTrainedTokenizerFast
if a.model_dir is None:
    from huggingface_hub import snapshot_download
    a.model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS")
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
# The 7.8 GB checkpoint is read with torch.load and then copied into the model: two full copies in RAM.
# mmap=True keeps the first copy on disk, which is what lets this run on a 16 GB machine.
_load = torch.load
def _mmap_load(f, *args, **kw):
    if isinstance(f, str) and os.path.getsize(f) > 1e9: kw.setdefault("mmap", True)
    return _load(f, *args, **kw)
torch.load = _mmap_load
from bluemagpie import BlueMagpieModel
t0 = time.time(); tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(a.model_dir, tokenizer=tok, training=False, device=device)
table = _load(os.path.join(a.model_dir, "checkpoints", "speaker_centroids.pt"), map_location="cpu", weights_only=True)
centroid = table["centroids"][table["speaker_ids"].index("hung_yi_lee")]
print(f"model loaded on {device} in {time.time() - t0:.0f} s", flush=True)
log = []
for label in a.items:
    text = f"這個字唸，{label.split('_')[1]}。"
    for k in range(a.seeds):
        seed = 1000 + k; name = f"{label}_s{seed}.wav"; f = os.path.join(a.out, name)
        if not os.path.exists(f):
            torch.manual_seed(seed); t1 = time.time()
            audio = model.generate(target_text=text, speaker_centroid=centroid, cfg_value=2.8, inference_timesteps=9, max_len=2000, retry_badcase=True)
            sf.write(f, audio.squeeze().cpu().float().numpy(), model.sample_rate); print(f"{name}  {time.time() - t1:.0f} s", flush=True)
        log.append(dict(label=label, text=text, seed=seed, file=name))
json.dump(log, open(os.path.join(a.out, "log.json"), "w"), ensure_ascii=False, indent=1)
print("done:", len(log), "clips in", a.out)
