# Online SNR pilot (jsPsych 7 on Pavlovia)

The consonant same/different test (`sound_perception_EN.py --axis dprime --snr <dB>`) as a browser experiment, for
choosing the session-wide SNR of the SFT task (DECISIONS 24) without a lab visit. Same 108-trial design per SNR,
same stimuli (pre-mixed with the lab's own seeds), plus a headphone screen.

```
 online/snr_pilot/
    index.html, experiment.js      the experiment (jsPsych 7.3.4; lib/ vendored, no CDN)
    conditions.json / .csv         the fixed 108-row plan: pair, talker, level, seeds, file names per SNR
    stim/                          648 pre-mixed syllables (r<row>_s<1|2>_snr<dB>.wav) + 18 headphone-check tones, 21 MB
 online/make_snr_pilot_stimuli.py  rebuilds stim/ and conditions.* from stimuli/*.wav with sft_noise (deterministic)
 online/pavlovia_to_perception_csv.py   turns a Pavlovia data file into data/<id>_sound_perception_dprime_snr<dB>.csv + .json
```

## Put it on Pavlovia

1. pavlovia.org → Dashboard → **Experiments** → New: pick *jsPsych*, give it a name. Pavlovia creates a GitLab repo.
2. Upload this whole folder (`index.html`, `experiment.js`, `conditions.json`, `lib/`, `stim/`) into that repo
   (GitLab web upload, or `git clone` the repo, copy the files in, `git push`).
3. Set the experiment to **Piloting** to try it (free), **Running** to collect (needs credits or a licence).
4. Participant link: `https://run.pavlovia.org/<user>/<experiment>/?participant=XX` — add `&snr=0,6,12` to set the
   block order (default), `&snr=6` for one block, `&skipcheck=1` to skip the headphone screen (testing only).
5. Data: Dashboard → the experiment → Download results; one CSV per participant in `data/`.

The Pavlovia bridge is loaded from `https://lib.pavlovia.org/jspsych-7-pavlovia-2022.1.1.js` (the version Pavlovia
documents for jsPsych 7). If Pavlovia's docs name a newer file, change that one line in `index.html`.

## Test locally

```bash
cd online/snr_pilot
python -m http.server 8000          # fetch() needs http, not file://
# open http://localhost:8000/?participant=test&snr=6&skipcheck=1
```

Without the Pavlovia bridge the page ends with a "Download the data file" link.

## Back into the lab format

```bash
python online/pavlovia_to_perception_csv.py path/to/<id>_snr_pilot.csv
# -> data/<id>_sound_perception_dprime_snr0.csv, _snr6.csv, _snr12.csv (+ .json summaries), same columns as the lab run
```

## What is and is not controlled online

- The SNR is inside each file (speech and noise mixed), so it survives any playback volume. Absolute level is not
  controlled; the headphone screen (Woods et al., 2017; 6 trials, pass ≥ 5) rejects loudspeakers.
- RT precision in a browser is ±10–20 ms; the pilot is scored on accuracy, so this is fine. Do not use this page for
  the SFT main task.
- The plan is fixed (`PLAN_SEED`), only the trial order is shuffled per participant; seeds are logged per row.
