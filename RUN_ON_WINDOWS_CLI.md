# Running the experiments on Windows from the command line

For someone comfortable with a terminal. Same scripts as the `.bat` files; this makes every step explicit so
nothing depends on a hard-coded path. PowerShell throughout.

Updated 2026-11-26. **Not executed on a Windows machine by the author** (the development container is Linux).
What was checked from there:

- the packages in step 3 resolve on Windows for Python 3.10, 3.11 and 3.12, all as wheels (nothing is compiled);
- the same install, the diagnostics and a simulated session were run on Linux with Python 3.12 and with Python 3.10,
  the two stacks in the table of step 3 (`RUN_ON_LINUX.md`, table at the top).

Route A below is what `setup_pymc_coder.bat` does and what the lab machine uses. Route B is new and untested on
Windows. Report what breaks, with the exact line.

```
 VAWM_run\
    Visual_AudioWM\     SFT_WM_nochange_EN.py, sound_perception_EN.py, adaptive_vawm_EN.py, sft_noise.py,
                        audio_pick.py, debug_adaptivesft_EN.py, GRTv3_ada_EN.py, stimuli\, data\ (output)
    adaptiveSFT\        the model-fitting package (PyMC race model) the SFT calibration imports
```

## 0. What has to be on the machine

| | why | check |
|---|---|---|
| a Python with PsychoPy (route A or B in step 2) | everything is run with that one `python.exe`, stored in `$py` | step 2 |
| git (optional) | to get and update this folder; without it, download the zip (step 1b) | `git --version` |
| headphones | over speakers the delivered SNR is not the configured SNR, and since 2026-11-26 every SFT sound is in noise | plugged in before step 4 |

## 1. Get the code

**1a. With git**

```powershell
git clone https://github.com/shadowcrow0/VAWM_run.git
cd VAWM_run
```

Later updates: `git pull` in the same folder.

**1b. Without git** (nothing to install; PowerShell downloads and unpacks the current snapshot)

```powershell
cd $HOME\Documents
Invoke-WebRequest -Uri https://github.com/shadowcrow0/VAWM_run/archive/refs/heads/main.zip -OutFile VAWM_run.zip
Expand-Archive -Path VAWM_run.zip -DestinationPath . -Force
Rename-Item VAWM_run-main VAWM_run        # the zip unpacks as VAWM_run-main
cd VAWM_run
```

To update later, repeat 1b into a fresh folder and copy your `Visual_AudioWM\data\`, `psychopy_path.txt` and
`Visual_AudioWM\audio_device.txt` across (or just download again and redo steps 2 to 4; they take a few minutes).

If you would rather have git: `winget install --id Git.Git -e` in an administrator PowerShell, then open a new
window and use 1a.

Participant data in `Visual_AudioWM\data\` are never pushed from here; send them to the private repository.

## 2. Choose the Python: `$py`

Do **one** of the two. Both end with `$py` pointing at a `python.exe`; every later step is the same.

### Route A — Standalone PsychoPy (the lab machine's setup)

Install Standalone PsychoPy 2024.2 or newer from psychopy.org. Its own `python.exe` already has PsychoPy, numpy,
scipy, pandas and psychtoolbox; step 3 adds PyMC, nutpie and `adaptivesft` next to it.

Newer builds do not all put `python.exe` in the same place. Let PowerShell look:

```powershell
$py = (Get-ChildItem "C:\Program Files\PsychoPy*", "$env:LOCALAPPDATA\Programs\PsychoPy*", "${env:ProgramFiles(x86)}\PsychoPy*" `
        -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
$py
```

If that prints nothing: open PsychoPy Coder, click the **Shell** tab at the bottom, run
`import sys; print(sys.executable)`, and copy the path:

```powershell
$py = "C:\...\python.exe"      # the path Coder printed
```

PsychoPy adds `<its preferences folder>\packages` to `sys.path`. Installing with `--user` while `PYTHONUSERBASE`
points there puts the packages where both the command line and Coder / Builder see them:

```powershell
& $py -c "from psychopy.preferences import prefs; print(prefs.paths['userPrefsDir'])"
# usually C:\Users\<you>\AppData\Roaming\psychopy3
$env:PYTHONUSERBASE = "$env:APPDATA\psychopy3\packages"     # use <the printed folder>\packages if it differs
$pipargs = @("-m", "pip", "install", "--user")
```

### Route B — a separate Python 3.12 environment (no Standalone PsychoPy, no Coder / Builder)

Use this on a machine that has no PsychoPy yet and will only run the scripts from the terminal. It is the same
environment as the Linux and Mac guides build.

```powershell
winget install --id astral-sh.uv -e          # then open a new PowerShell window
uv venv --seed --python 3.12 $HOME\venv-vawm      # --seed puts pip into the venv; step 3 uses it
$py = "$HOME\venv-vawm\Scripts\python.exe"
uv pip install --python $py psychopy psychtoolbox
$pipargs = @("-m", "pip", "install")
```

Always pass `--python 3.12`: PsychoPy 2026.2 declares support for 3.10 to 3.12 only, and uv does not stop you from
using a newer one.

### Both routes: confirm and remember

```powershell
& $py --version                                              # 3.10, 3.11 or 3.12
& $py -c "import psychopy; print(psychopy.__version__)"
Set-Content -Path psychopy_path.txt -Value $py               # the .bat files read this, so both ways stay in sync
```

`$py`, `$pipargs` and `$env:` settings last for the current PowerShell window only. In a new window, set `$py`
again (`$py = Get-Content psychopy_path.txt`) and, on route A, `$env:PYTHONUSERBASE` as above.

## 3. Install the fitting stack (SFT only; the GRT experiment does not need it)

```powershell
& $py @pipargs --upgrade pip
& $py @pipargs "pymc>=5.20" nutpie arviz numba pytest
& $py @pipargs --no-deps -e .\adaptiveSFT
& $py -c "import sys, pymc, nutpie, arviz, numba, adaptivesft; print('OK python', sys.version.split()[0], 'pymc', pymc.__version__, 'nutpie', nutpie.__version__)"
```

Takes 3–10 minutes the first time. pip picks the newest versions that support the Python it runs under, so what the
`OK` line shows depends on `& $py --version` (resolved on 2026-11-26):

| Python | PyMC | nutpie | ArviZ |
|---|---|---|---|
| 3.12 | 6.4.0 | 0.16.11 | 1.3.0 |
| 3.11 | 5.28.5 | 0.16.8 | 0.23.4 |
| 3.10 | 5.25.1 | 0.15.2 | 0.23.4 |

All three rows work with the scripts (the 3.12 and 3.10 stacks were run on Linux). Write the versions from the `OK`
line into the lab log. If the last line fails on `nutpie`, send the output of `& $py --version`; the fallback is
`FIT_KW = dict(..., nuts_sampler="pymc")` in `SFT_WM_nochange_EN.py` (slower fits, same model).

## 4. Audio: find and pin the headphones

```powershell
cd Visual_AudioWM
& $py audio_pick.py                       # table of output devices, and the one that would be chosen
& $py audio_pick.py --probe --beep        # also open each device, then play a 1 s tone on the chosen one
& $py audio_pick.py --set "Headphones"    # pin by (part of) the device name -> audio_device.txt
```

- The automatic choice looks for a name containing `headphone`, `headset`, `earphone`, `earbud`, `airpods` or `buds`
  (`audio_pick.py:42`). `Speakers (USB Audio Device)` or a brand-only name does **not** match.
- A device you name yourself is trusted as it is (`audio_pick.py:203`): pin it with `--set`, or pass
  `--audio-device "NAME"`. `--require-headphones` and the GRT script's built-in headphone requirement then pass,
  so there is no reason to drop the flag.

The device name is written into every data row (`audio_device`). Set the system volume once with the tone and do
not touch it afterwards. Turn off Windows audio enhancements and spatial sound for that device (Sound settings →
device properties); they change the noise.

## 5. Check the whole chain

```powershell
& $py debug_adaptivesft_EN.py
$LASTEXITCODE                              # 0 = everything passed
& $py -m pytest -q tests                   # 36 tests, about half a minute
```

Nine numbered groups `[0]`–`[8]`: environment and packages, `adaptiveSFT` found, imports, which calibration methods
are available, a simulated calibration without a window, a short real PyMC fit, a window test, the audio device and
calibration settings, and the noise mixer. Every line should start with `✔`; a `✘` line prints the fix. Group `[8]`
must show `SND_SNR_DB = 3.0 (every sound mixed at that SNR)`.

## 6. Run

```powershell
# self-test with a simulated participant, no window, under a minute
& $py SFT_WM_nochange_EN.py --robot --quick

# pure perception, no memory load
& $py sound_perception_EN.py --require-headphones                          # ~7 min, fixed 80 pairs + 40 same, clean speech
& $py sound_perception_EN.py --axis dprime --snr 3 --require-headphones    # the experiment's axis, in +3 dB noise

# SFT experiment: colour calibration -> sound calibration -> practice -> 6 blocks
& $py SFT_WM_nochange_EN.py --require-headphones

# GRT experiment
& $py snr_runtime.py                                 # noise-mixer self-check
& $py GRTv3_ada_EN.py
```

`--robot --quick` overwrites `data\robot_simple_*`; in a git checkout run `git checkout -- data/` afterwards.

**Noise.** Since 2026-11-26 the SFT experiment mixes every speech sound into speech-shaped noise at +3 dB
(`SND_SNR_DB = 3.0`, `SFT_WM_nochange_EN.py:51`; reasons and what has not been measured: `DECISIONS.md` 27 in the
source repository). `sound_perception_EN.py` is clean unless `--snr` is given.

Output lands in `Visual_AudioWM\data\`:

```
 <id>_sound_perception.csv / .json                     default run (clean speech, fixed pairs)
 <id>_sound_perception_dprime_snr3.csv / .json         with --axis dprime --snr 3
 <id>_simple_calib_trials.csv                          every calibration trial (snd_snr_db, snd_seed_* columns)
 <id>_simple_calib.json                                fitted L / H per dimension, 50 % point and plateau with intervals, warnings
 <id>_simple_main.csv                                  main-task probes
 <id>_SFT_WM_<date>.csv / .psydat / .log               PsychoPy's standard files
 <id>_snr\                                             the mixed wavs; delete after copying the csv files
                                                       (each is rebuildable from the SNR and seed logged in its row)
```

Send the `data\` files to the private repository (Visual_AudioWM `data/`), never to VAWM_run.

## 7. The knobs, and where they live

| what | file | line / name |
|---|---|---|
| colour levels, calibration model (ogival + b0), plateau fraction | `adaptive_vawm_EN.py` | `LEVELS_COLOUR`, `CALIB_LINK`, `CALIB_BIAS`, `CALIB_PLATEAU` (lines 109–115) |
| sound levels | `adaptive_vawm_EN.py` | `LEVELS_AUDIO` (line 110) |
| session SNR for the SFT sounds (3.0 dB; None = clean) | `SFT_WM_nochange_EN.py` | `SND_SNR_DB` (line 51) |
| trial counts, response window, fit settings | `SFT_WM_nochange_EN.py` | header, lines 36–55 |

The reasons behind every value are in `DECISIONS.md` of the source repository; the single-source table is
`ARCHITECTURE.md` §3 there. Change a value in one place only.

## 8. If something fails

| symptom | cause | fix |
|---|---|---|
| `pip` SSL / proxy errors | institutional network | add `--proxy http://...` to the pip line, or install from a hotspot once |
| `ModuleNotFoundError: pymc` when run from Coder but not from the terminal (route A) | `PYTHONUSERBASE` differs | step 2 route A: use the folder Coder prints, redo step 3 |
| `No module named pip` (route B) | the venv was made without `--seed` | `& $py -m ensurepip --upgrade` |
| every `& $py ...` line fails at the `&` | new PowerShell window, `$py` is empty | `$py = Get-Content psychopy_path.txt` |
| `audio_pick.py` lists only `Microsoft Sound Mapper` / `Primary Sound Driver` | no PortAudio device named after the headphones | select the headphones in Windows sound settings, pin that generic device with `--set`, confirm with `--beep` |
| window opens then closes at once | an exception before the first frame | run again from the terminal; the traceback is in the window you ran it from |
| `PsychoPy python not found` from a `.bat` | step 2 not done | `psychopy_path.txt` with the path, or run the `.bat` and paste it when asked |
