"""
Generate /aCa/ stimuli using espeak-ng phoneme input.
espeak-ng has precise phoneme control for accurate /ɑCɑ/ production.

Usage:
    sudo apt install espeak-ng   # Linux
    brew install espeak-ng       # macOS
    python generate_aCa_stimuli.py
"""

import subprocess
import wave
from pathlib import Path

# ── espeak-ng phoneme mapping for /aCa/ context ───────────────────────────
# espeak phoneme notation: [[...]] for phoneme input
# A: = /ɑ/ (open back unrounded vowel)
# See: https://github.com/espeak-ng/espeak-ng/blob/master/docs/phonemes.md
CONSONANT_MAP = {
    "p":     ("ɑpɑ",  "[[A:pA:]]"),
    "t":     ("ɑtɑ",  "[[A:tA:]]"),
    "k":     ("ɑkɑ",  "[[A:kA:]]"),
    "f":     ("ɑfɑ",  "[[A:fA:]]"),
    "theta": ("ɑθɑ",  "[[A:TA:]]"),      # T = θ (voiceless dental fricative)
    "s":     ("ɑsɑ",  "[[A:sA:]]"),
    "sh":    ("ɑʃɑ",  "[[A:SA:]]"),      # S = ʃ (voiceless postalveolar fricative)
    "b":     ("ɑbɑ",  "[[A:bA:]]"),
    "d":     ("ɑdɑ",  "[[A:dA:]]"),
    "g":     ("ɑgɑ",  "[[A:gA:]]"),
    "v":     ("ɑvɑ",  "[[A:vA:]]"),
    "eth":   ("ɑðɑ",  "[[A:DA:]]"),      # D = ð (voiced dental fricative)
    "z":     ("ɑzɑ",  "[[A:zA:]]"),
    "zh":    ("ɑʒɑ",  "[[A:ZA:]]"),      # Z = ʒ (voiced postalveolar fricative)
    "m":     ("ɑmɑ",  "[[A:mA:]]"),
    "n":     ("ɑnɑ",  "[[A:nA:]]"),
}

# espeak-ng voices
VOICES = [
    ("en-us", "female", "f1"),   # American English female
    ("en-us", "male",   "m1"),   # American English male
    ("en",    "female", "f2"),   # British English female
    ("en",    "male",   "m2"),   # British English male
]

OUTPUT_DIR = Path("aCa_stimuli")


def generate_with_espeak(phonemes: str, voice: str, variant: str, out_path: Path,
                         speed: int = 120, pitch: int = 50):
    """Generate audio using espeak-ng phoneme input."""
    cmd = [
        "espeak-ng",
        "-v", f"{voice}+{variant}",
        "-s", str(speed),       # speed (words per minute)
        "-p", str(pitch),       # pitch (0-99)
        "-w", str(out_path),    # output wav file
        phonemes
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng error: {result.stderr}")
    return out_path


def generate_stimuli():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating {len(CONSONANT_MAP)} consonants × {len(VOICES)} voices")
    print(f"= {len(CONSONANT_MAP) * len(VOICES)} files\n")

    results = []

    for voice_lang, voice_gender, voice_id in VOICES:
        voice_name = f"{voice_lang}_{voice_gender}"
        voice_dir = OUTPUT_DIR / voice_name
        voice_dir.mkdir(exist_ok=True)

        for label, (ipa_display, phonemes) in CONSONANT_MAP.items():
            out_path = voice_dir / f"{label}.wav"

            try:
                # Use different variant for gender
                variant = "f3" if voice_gender == "female" else "m3"
                generate_with_espeak(phonemes, voice_lang, variant, out_path)

                print(f"  [{voice_name}] {label:6s}  {ipa_display}  {phonemes:15s} → {out_path.name}")
                results.append({
                    "label": label,
                    "voice": voice_name,
                    "ipa": ipa_display,
                    "phonemes": phonemes,
                    "file": str(out_path)
                })

            except Exception as e:
                print(f"  [{voice_name}] {label}: ERROR - {e}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("label,voice,ipa,phonemes,file\n")
        for r in results:
            f.write(f"{r['label']},{r['voice']},{r['ipa']},{r['phonemes']},{r['file']}\n")

    print(f"\nDone. {len(results)} files saved to '{OUTPUT_DIR}/'")
    print(f"Manifest: {manifest_path}")
    return results


if __name__ == "__main__":
    generate_stimuli()