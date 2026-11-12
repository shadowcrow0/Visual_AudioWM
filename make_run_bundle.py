"""Pack the files needed to run the SFT experiment on a lab PC into VAWM_run.zip, or refresh the public run repo.

    python make_run_bundle.py                          # writes VAWM_run.zip next to this file
    python make_run_bundle.py --repo ../vawm_run       # refresh a checkout of github.com/shadowcrow0/VAWM_run

Unzipped layout (the two folders must stay side by side: adaptive_vawm_EN.py looks for ../adaptiveSFT):

    VAWM_run/      SFT_WM_nochange_EN.py, sound_perception_EN.py, adaptive_vawm_EN.py, audio_pick.py,
                   debug_adaptivesft_EN.py, generate_trials.py (consonant confusion table), setup_pymc_coder.bat,
                   GRTv3_ada_EN.py + AGRT.py + snr_audio.py + snr_runtime.py + agrt_colour_lut.json (the GRT experiment),
                   stimuli/ (wav), data/ (empty; results go here), README_RUN.md
    adaptiveSFT/   the adaptivesft package + pyproject.toml, so `pip install -e ..\\adaptiveSFT` works
"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(os.path.dirname(HERE), "adaptiveSFT")
OUT = os.path.join(HERE, "VAWM_run.zip")

FILES = ["SFT_WM_nochange_EN.py", "sound_perception_EN.py", "adaptive_vawm_EN.py", "audio_pick.py", "consonant_distance.py",
         "debug_adaptivesft_EN.py", "generate_trials.py", "setup_pymc_coder.bat", "RUN_ON_LINUX.md",
         # GRTv3_ada (the GRT experiment; same audio_pick, no PyMC): script + Psi + SNR mixing + colour LUT + Linux note
         "GRTv3_ada_EN.py", "AGRT.py", "snr_audio.py", "snr_runtime.py", "agrt_colour_lut.json", "RUN_GRTv3_ON_LINUX.md", "RUN_ON_MAC.md"]
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


def write_tree(stage, run_name="VAWM_run"):
    """Write the run folder (run_name/) and adaptiveSFT/ side by side under stage."""
    run = os.path.join(stage, run_name)
    os.makedirs(os.path.join(run, "data"), exist_ok=True)
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), run)
    shutil.rmtree(os.path.join(run, "stimuli"), ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "stimuli"), os.path.join(run, "stimuli"))
    open(os.path.join(run, "data", ".keep"), "w").close()
    open(os.path.join(run, "README_RUN.md"), "w", encoding="utf-8").write(README)
    pkg = os.path.join(stage, "adaptiveSFT")
    shutil.rmtree(pkg, ignore_errors=True)
    os.makedirs(pkg)
    for f in ["pyproject.toml", "README.md", "requirements.txt", "LICENSE"]:
        if os.path.isfile(os.path.join(SFT, f)):
            shutil.copy2(os.path.join(SFT, f), pkg)
    shutil.copytree(os.path.join(SFT, "adaptivesft"), os.path.join(pkg, "adaptivesft"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", metavar="DIR", help="instead of a zip, refresh a checkout of github.com/shadowcrow0/VAWM_run "
                                                 "(writes DIR/Visual_AudioWM and DIR/adaptiveSFT; the .bat files and README there are kept)")
    a = ap.parse_args()
    if a.repo:
        # the public repo keeps the run folder under the name Visual_AudioWM (the .bat files cd into it)
        run = os.path.join(a.repo, "Visual_AudioWM")
        keep_data = os.path.join(run, "data")
        for name in os.listdir(run) if os.path.isdir(run) else []:
            if name != "data":
                p = os.path.join(run, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        write_tree(a.repo, run_name="Visual_AudioWM")
        os.remove(os.path.join(run, "README_RUN.md"))          # the repo has its own README at the top level
        print("refreshed", a.repo)
        return
    stage = os.path.join(HERE, "_bundle")
    shutil.rmtree(stage, ignore_errors=True)
    write_tree(stage)
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
