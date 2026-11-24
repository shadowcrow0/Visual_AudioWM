#!/usr/bin/env bash
# Run the four steps in order. Usage: ./run_all.sh [output folder] [seeds per character]
set -euo pipefail
OUT="${1:-out}"; SEEDS="${2:-8}"; HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/synth.py"   --out "$OUT" --seeds "$SEEDS"
python "$HERE/verify.py"  --out "$OUT"
python "$HERE/measure.py" --out "$OUT"
python "$HERE/figs.py"    --out "$OUT"
echo "figures: $OUT/figs/"
