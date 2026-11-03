"""Pack the files needed to run the SFT experiment on a lab PC into VAWM_run.zip.

    python make_run_bundle.py            # writes VAWM_run.zip next to this file

Unzipped layout (the two folders must stay side by side: adaptive_vawm_EN.py looks for ../adaptiveSFT):

    VAWM_run/      SFT_WM_nochange_EN.py, sound_perception_EN.py, adaptive_vawm_EN.py, audio_pick.py,
                   debug_adaptivesft_EN.py, generate_trials.py (consonant confusion table), setup_pymc_coder.bat,
                   stimuli/ (wav), data/ (empty; results go here), README_RUN.md
    adaptiveSFT/   the adaptivesft package + pyproject.toml, so `pip install -e ..\\adaptiveSFT` works
"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(os.path.dirname(HERE), "adaptiveSFT")
OUT = os.path.join(HERE, "VAWM_run.zip")

FILES = ["SFT_WM_nochange_EN.py", "sound_perception_EN.py", "adaptive_vawm_EN.py", "audio_pick.py",
         "debug_adaptivesft_EN.py", "generate_trials.py", "setup_pymc_coder.bat"]
README = """# VAWM_run — how to run the experiment on a new PC

Keep `VAWM_run/` and `adaptiveSFT/` side by side (as unzipped).

1. Once per PC (installs PyMC + nutpie into PsychoPy's own Python and the adaptivesft package):

       setup_pymc_coder.bat ..\\adaptiveSFT

2. Find the headphones (prints every output device, opens them, plays a tone on the chosen one):

       python audio_pick.py --probe --beep
       python audio_pick.py --set "Headphones"        <- the name you saw

3. Check the whole chain without a participant (imports, a short fit, device):

       python debug_adaptivesft_EN.py

4. Pure perception test of the consonants (no memory load, ~6 min; run this before the experiment):

       python sound_perception_EN.py

5. Run (PsychoPy Coder: open the file and press Run; or from a terminal):

       python SFT_WM_nochange_EN.py
       python SFT_WM_nochange_EN.py --robot --quick    <- simulated participant, no window, a few minutes

Data land in `data/`: `<id>_simple_calib_trials.csv`, `<id>_simple_calib.json`, `<id>_simple_main.csv`,
plus PsychoPy's `<id>_SFT_WM_<date>.csv / .psydat / .log`. Copy the whole `data/` folder back.

Settings are the constants at the top of `SFT_WM_nochange_EN.py` (targets 8.0 / 1.3, colour levels dE 2–90,
calibration 16 warm-up + 40 fitted + 8 filler per dimension, main task 6 blocks x 14 sets x 16 probes).
"""


def main():
    stage = os.path.join(HERE, "_bundle")
    shutil.rmtree(stage, ignore_errors=True)
    run = os.path.join(stage, "VAWM_run")
    os.makedirs(os.path.join(run, "data"))
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), run)
    shutil.copytree(os.path.join(HERE, "stimuli"), os.path.join(run, "stimuli"))
    open(os.path.join(run, "data", ".keep"), "w").close()
    open(os.path.join(run, "README_RUN.md"), "w", encoding="utf-8").write(README)
    pkg = os.path.join(stage, "adaptiveSFT")
    os.makedirs(pkg)
    for f in ["pyproject.toml", "README.md", "requirements.txt", "LICENSE"]:
        if os.path.isfile(os.path.join(SFT, f)):
            shutil.copy2(os.path.join(SFT, f), pkg)
    shutil.copytree(os.path.join(SFT, "adaptivesft"), os.path.join(pkg, "adaptivesft"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, stage))
    shutil.rmtree(stage)
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
