"""Last step. Collect the small result files (no audio) into out/send_back/.

    python send_back.py --out out
"""
import os, glob, shutil, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); a = ap.parse_args()
dst = f"{a.out}/send_back"; os.makedirs(dst, exist_ok=True)
files = [f"{a.out}/{n}" for n in ("verified.json", "measures.json", "splice/splice_log.json", "splice/splice_results.json", "splice/vowel_side.json", "splice/listen_responses.csv")] + glob.glob(f"{a.out}/figs/*.png")
for f in files:
    if os.path.exists(f): shutil.copy2(f, dst)
print(f"\n{len(os.listdir(dst))} files in {os.path.abspath(dst)}\n把這個資料夾的檔案傳回來（不含音檔）。")
