# CLAUDE.md

## What this repo is

A **PsychoPy** experiment on **audiovisual working memory**, analysed with
**GRT** (General Recognition Theory). The question: when you remember a colour
paired with a speech sound, are the two features stored independently, or does
one interfere with the other?

Each item is a **colour × sound** pair, encoded as 2 bits:

```
item = 0     1     2     3
color  C1    C2    C1    C2      colour = i & 1
sound  S1    S1    S2    S2      sound  = (i >> 1) & 1

so:  target ^ 1 = differs in colour only
     target ^ 2 = differs in sound only
     target ^ 3 = differs in both
```

That XOR trick is used throughout the trial code — `relation` in the data files
is exactly this value.

## One trial

```
study  ~4.3s   four colour+sound items, one per corner, 1s each
               onsets 0.3 / 1.3 / 2.3 / 3.3s, order reshuffled per trial
               all four items appear every trial (one of each)

cue     2.0s   0-1s  empty white frame at one corner
               1-2s  a colour+sound appears inside that frame
               ⚠ the frame sits where cued_item was studied,
                 but the content shown is target_item

task     ---   four options (colour + label), one per corner, reshuffled
               keys:  g = upper left    j = upper right
                      f = lower left    h = lower right

rest     ---   only after practice and after each block; skipped otherwise
```

On **valid** trials (2/3 of them) cued_item == target_item, so there is no
conflict. On **invalid** trials the cue points one way and the probe shows
another; picking the cued item is scored `intrusion`.

**Because the probe displays the answer, valid trials are an identification
task, not a memory task.** Memory only enters through the invalid trials'
intrusions. See `review/voicing×顏色互動_資料怎麼量到.md` §1.3 — that wording
matters for any write-up.

Counts: 24 practice + 576 main (4 blocks × 144) = 600 trials, valid:invalid
held at 2:1 within every block. Practice deliberately uses the same ratio.

## The two experiment scripts

Both are PsychoPy Builder output from `GRTv2.psyexp`, hand-edited afterwards.
**Do not regenerate from Builder** — that would wipe the hand-written sections.

| | `GRTv3_a.py` | `GRTv3.py` |
|---|---|---|
| `_a` = | adaptive | — |
| Sound dimension | SNR on `be.wav` (/bi/) | same |
| SNR levels | **calibrated per participant** by the AGRT phase | **fixed** +6 / −6 dB |
| Calibration | 60-trial AGRT phase (colour + SNR together) | none |
| Colour axis | calibrated per participant | fixed ±3.0 ΔE00 |
| Use it for | real data | checking the flow runs |

Both auditory dimensions are now **SNR**, i.e. **audibility** — not consonant
identity. The 9-step b/p continuum that `GRTv3_a.py` used to run is in git
history at `165d823` and earlier; `stimuli/kutlu_mcmurray_2024/` is its stimulus
set and is currently unused. The trade-off behind that switch: SNR is a
continuous knob, so the adaptive procedure's arbitrary-real proposals land
exactly (no rounding to one of 9 files), but what gets measured is audibility.
See `snr_vs_grt_dimension.md` and `review/聽覺維度_嘗試與放棄紀錄.md` §2.6.

The two scripts' `snr_only` levels are **not the same numbers** — one is
per-participant, one is fixed — so do not pool their CSVs without accounting
for that.

⚠️ **Open decision in `GRTv3_a.py`: `SND_FEEDBACK`.** The b/p version could
score the sound judgement against a real phonetic category boundary. "Clear vs
noisy" has no such ground truth — the criterion is subjective. The current
setting gives feedback against 0 dB (speech power = noise power), which anchors
an otherwise drifting criterion but means the estimated α is partly "where the
feedback pushed them". Set `SND_FEEDBACK = False` to score nothing on that
dimension instead.

## Supporting modules

```
AGRT.py            Psi adaptive procedure (Kontsevich & Tyler 1999),
                   GRT variant by Glavan 2022. GPL, third-party.
                   Locally patched for a removed scipy API.
                   Two independent Psi objects, one per dimension.

agrt_setup.py      builds the perceptually-uniform colour axis (ΔE00 arc
   +               length, not CIELAB hue angle — hue degrees are not
agrt_colour_lut    perceptually uniform, which would silently break AGRT's
   .json           constant-variance assumption).
                   ⚠ Needs `colour-science`, which PsychoPy's bundled Python
                     does NOT have. So the LUT is generated OFFLINE and the
                     experiment only does a numpy table lookup at runtime.
                     To change the colour gamut, re-run export_lut().

snr_audio.py       mixes speech into speech-shaped noise at an exact SNR.
                   Aligns onset + voiced-segment RMS across tokens, fixes
                   output level so loudness is not a cue, logs the noise
                   seed so any sample can be rebuilt bit-for-bit.

snr_runtime.py     the thin layer between snr_audio and the experiment:
                   mixes a stimulus ON DEMAND at any real dB (~3 ms each),
                   writes a wav, and logs dB + noise seed for every one.
                   Fresh seed per call, so running noise is automatic.
                   `SNRStimulus.rebuild(seed, db)` reconstructs any
                   stimulus that was actually presented.
                   Run `python snr_runtime.py` for its self-check.

audio_device.py    resolves the output device BY NAME, never by index
                   (indices shift between machines and across replugging).
```

## Gotchas that have already bitten

- **`thisExp.nextEntry()` is written by hand** at the end of the trial loop.
  The loop's handler is built with `isTrials=False`, so PsychoPy does not
  emit it. Without that line all 600 trials overwrite one row and the saved
  file has a single line of data.
- **`GRTv3.py` requires headphones** (`require_headphones=True`) and errors
  out if none are found. Over speakers the delivered SNR is not the
  configured SNR, so the data is worthless.
- **`dim2steps` in `AGRTHandler` sets three grids at once** — the stimulus
  grid, the α grid and the β grid. It is 9 for the b/p continuum because only
  9 audio files exist, which leaves α and β on a 9-point search grid too.
- **PsychoPy is not installed in the dev container.** Scripts can be
  `py_compile`d and the pure-numpy modules can be exercised, but nothing
  visual or audio-related can be run here. Say so rather than implying a
  script was executed.

## Open design question

The instructions say *"report which item was there"* but scoring credits the
**probe's** content. On invalid trials a participant following the instructions
literally is recorded as `intrusion`, and there is no trial-by-trial feedback
to teach them otherwise. Relative comparisons across `relation` levels survive
this; absolute rates do not. Fixing it means changing the instruction wording
or the scoring — a design decision for the researcher, not a cleanup.

## `review/`

Chinese-language decision and audit notes: why the auditory dimension went
duration → VOT → SNR, why this consonant pair, statistical plan, what was tried
and abandoned. `決策脈絡_索引.md` is the index.

These are **dated records**. They cite line numbers against the older
`GRTv2.py` / `GRTv2_demo.py` filenames — that is intentional, not staleness to
fix.

## Response style

**Be concise.** Lead with the answer. Cut preamble, restatements of the
question, and closing summaries of what was just said. Length should track the
complexity of the question, not the effort spent on it.

Keep what is load-bearing: caveats that change a decision, what was verified
versus assumed, and the numbers behind a claim. Trim commentary, not evidence.

**Use ASCII to visualize content when explaining concepts.** When something has
structure — a timeline, a data flow, a grid, a state change, a before/after —
draw it instead of describing it in prose. Label the boxes, keep it under ~15
lines, put it in a fenced block so the alignment survives. Skip it when there
is no structure to show (a yes/no answer, a single number).

## Conventions

- Comments and docs in this repo are written in Chinese; match that.
- Comments explain **why**, especially where a simpler-looking approach was
  tried and failed. Keep that reasoning when editing nearby code.
- Generated per-participant stimuli (`data/*_snr/`, ~130 MB per session) are
  gitignored — they are rebuildable from the dB + seed logged in each trial's
  `snd_*` columns.
