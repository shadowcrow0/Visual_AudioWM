#!/usr/bin/env bash
# Everything for the "why ㄅㄛ and not ㄅㄨㄛ" question, in order. Usage: ./run_splice.sh [output folder] [seeds per character]
# Clips that already exist in the output folder are not synthesized again.
set -euo pipefail
OUT="${1:-out}"; SEEDS="${2:-8}"; HERE="$(cd "$(dirname "$0")" && pwd)"
# the 13 characters of run_all.sh plus 歌 / 哥 (ㄍㄜ), the unrounded partner of 鍋 / 郭 (ㄍㄨㄛ)
ITEMS="bo_撥 bo_玻 bo_菠 po_坡 po_潑 duo_多 tuo_拖 guo_鍋 guo_郭 pu_撲 fu_夫 du_督 gu_姑 ge_歌 ge_哥"
python "$HERE/synth.py"        --out "$OUT" --seeds "$SEEDS" --items $ITEMS
python "$HERE/verify.py"       --out "$OUT"
python "$HERE/measure.py"      --out "$OUT"
python "$HERE/figs.py"         --out "$OUT"
python "$HERE/splice.py"       --out "$OUT"
python "$HERE/splice_check.py" --out "$OUT"
python "$HERE/figs_splice.py"  --out "$OUT"
python "$HERE/send_back.py"    --out "$OUT"
echo "optional listening test:  python $HERE/listen.py --out $OUT   (then run send_back.py again)"
