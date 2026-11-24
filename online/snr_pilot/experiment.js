/* Online SNR pilot of the consonant same/different test (jsPsych 7.3.4).
 * Mirrors sound_perception_EN.py --axis dprime --snr <dB>: 108 trials per SNR (72 different + 36 same), two
 * syllables 0.5 s apart, y = same / n = different, answer from the onset of syllable 2 within 3.7 s, 0.6 s feedback.
 * Stimuli are pre-mixed (online/make_snr_pilot_stimuli.py); the browser only plays files and records keys.
 * URL parameters:  ?participant=ID   ?snr=0,6,12 (default; order = block order)   ?skipcheck=1 (no headphone test)
 * Data columns per pair trial: participant, task=pair, snr_db, block, trial, row, is_same, x_planned, x, consonant, foil,
 * feature_distance, talker, snd1, snd2, seed1, seed2, key, rt, correct.  Headphone check: task=hpcheck, answer, response, correct. */

const params = new URLSearchParams(window.location.search);
const PID = params.get("participant") || window.prompt("Participant id:") || "anon";
const SNR_ORDER = (params.get("snr") || "0,6,12").split(",").map(Number);
const SKIP_CHECK = params.get("skipcheck") === "1";
const ISI_MS = 500, RESP_WINDOW_MS = 3000, FEEDBACK_MS = 600, HP_PASS = 5;
const onPavlovia = typeof jsPsychPavlovia !== "undefined";

const jsPsych = initJsPsych({
  on_finish: () => {
    if (!onPavlovia) {
      const csv = jsPsych.data.get().csv();
      const pre = document.createElement("pre"); pre.id = "jspsych-data"; pre.textContent = csv; document.body.appendChild(pre);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = `${PID}_snr_pilot.csv`;
      a.textContent = "Download the data file"; a.style.cssText = "display:block;color:#9cf;font-size:24px;margin:40px";
      document.body.appendChild(a);
    }
  }
});
jsPsych.data.addProperties({ participant: PID, snr_order: SNR_ORDER.join("|"), user_agent: navigator.userAgent });

// The sounds come as base64 bundles (online/pack_stimuli.py) so the whole experiment is a dozen files that can be
// uploaded one by one through GitLab's web UI. Each wav becomes a Blob URL; "stim/<name>" is mapped to that URL.
const SRC = {};
const b64url = (b64) => { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return URL.createObjectURL(new Blob([u], { type: "audio/wav" })); };
const snd = (name) => SRC[name.replace(/^stim\//, "")] || name;
document.body.innerHTML = "<p style='font-size:22px;margin:40px'>Loading the sounds (about 20 MB)…</p>";
const needed = [SKIP_CHECK ? null : "hp"].filter(Boolean).concat(SNR_ORDER.flatMap(s => [`snr${s}_s1`, `snr${s}_s2`]));
Promise.all([fetch("conditions.json").then(r => r.json())].concat(needed.map(g => fetch(`bundles/stim_${g}.json`).then(r => r.json()))))
.then(([cond, ...bundles]) => {
  bundles.forEach(b => Object.entries(b).forEach(([name, b64]) => { SRC[name] = b64url(b64); }));
  document.body.innerHTML = "";
  window.__hp = cond.headphone_check;   // test hook: lets an automated driver answer the headphone check
  const timeline = [];
  if (onPavlovia) timeline.push({ type: jsPsychPavlovia, command: "init" });

  const files = [];
  cond.headphone_check.forEach(t => t.files.forEach(f => files.push(snd(f))));
  cond.trials.forEach(t => SNR_ORDER.forEach(s => { files.push(snd(t[`file1_snr${s}`])); files.push(snd(t[`file2_snr${s}`])); }));
  timeline.push({ type: jsPsychPreload, audio: files, show_progress_bar: true, max_load_time: 600000,
                  message: "<p class='center'>Preparing the sounds…</p>" });

  const page = (html) => ({ type: jsPsychHtmlKeyboardResponse, stimulus: html, choices: [" "], post_trial_gap: 300 });
  const fix = (ms, cls = "fix") => ({ type: jsPsychHtmlKeyboardResponse, stimulus: `<div class='${cls}'>+</div>`, choices: "NO_KEYS", trial_duration: ms });

  timeline.push(page(`<p>Welcome. This test takes about ${7 * SNR_ORDER.length + 3} minutes.</p>
    <p><b>Please use headphones</b>, sit somewhere quiet, and set the volume to a comfortable level now (it will not change during the test).</p>
    <p>Press the space bar to continue.</p>`));

  // ---- headphone screening (Woods et al., 2017) ----
  if (!SKIP_CHECK) {
    timeline.push(page(`<p>First a short headphone check (${cond.headphone_check.length} trials).</p>
      <p>You will hear three tones. One of them is <b>quieter</b> than the other two. Press <b>1</b>, <b>2</b> or <b>3</b> for the quietest tone.</p>
      <p>Press the space bar to begin.</p>`));
    cond.headphone_check.forEach(t => {
      t.files.forEach((f, i) => {
        timeline.push({ type: jsPsychAudioKeyboardResponse, stimulus: snd(f), choices: "NO_KEYS", trial_ends_after_audio: true,
                        prompt: `<div class='fix'>${i + 1}</div>`, post_trial_gap: 300 });
      });
      timeline.push({ type: jsPsychHtmlKeyboardResponse, stimulus: "<p class='center'>Which tone was the quietest?  1 / 2 / 3</p>", choices: ["1", "2", "3"],
                      data: { task: "hpcheck", trial: t.trial, answer: t.answer, kinds: t.kinds.join("|") },
                      on_finish: d => { d.correct = (parseInt(d.response) === d.answer) ? 1 : 0; }, post_trial_gap: 400 });
    });
    timeline.push({ type: jsPsychHtmlKeyboardResponse, choices: [" "],
      stimulus: () => {
        const n = jsPsych.data.get().filter({ task: "hpcheck" }).select("correct").sum();
        jsPsych.data.addProperties({ hp_correct: n, hp_pass: n >= HP_PASS ? 1 : 0 });
        return n >= HP_PASS ? "<p class='center'>Headphone check passed. Press the space bar to continue.</p>"
          : `<p class='center'>The headphone check did not pass (${n}/${cond.headphone_check.length}).</p><p class='center'>Please put on headphones, check that both sides work, and reload the page to try again. Press the space bar to end.</p>`;
      },
      on_finish: () => { if (jsPsych.data.get().last(1).values()[0].hp_pass === 0) jsPsych.endExperiment("<p class='center'>Thank you.</p>"); } });
  }

  // ---- the pair test, one block per SNR ----
  timeline.push(page(`<p>You will hear two syllables, one after the other, in background noise.</p>
    <p><b>Is the CONSONANT the same in both?</b></p>
    <p class='center'>Y = same &nbsp;&nbsp;&nbsp;&nbsp; N = different</p>
    <p>The background noise is different every time; listen to the consonant only. Answer after the second syllable, as quickly as you can, on every trial. You will see whether you were right.</p>
    <p>Press the space bar to begin.</p>`));

  SNR_ORDER.forEach((snr, bi) => {
    if (bi > 0) timeline.push(page(`<p>Block ${bi + 1} of ${SNR_ORDER.length}. The noise level changes now. Take a short break if you like.</p><p>Press the space bar to continue.</p>`));
    const order = jsPsych.randomization.shuffle(cond.trials.map((_, i) => i));
    order.forEach((idx, k) => {
      const t = cond.trials[idx];
      const base = { task: "pair", snr_db: snr, block: bi + 1, trial: k + 1, row: t.row, is_same: t.is_same, x_planned: t.x_planned, x: t.x,
                     consonant: t.consonant, foil: t.foil, feature_distance: t.feature_distance, talker: t.talker,
                     snd1: t.snd1, snd2: t.snd2, seed1: String(t.seed1), seed2: String(t.seed2) };
      timeline.push(fix(300));
      timeline.push({ type: jsPsychAudioKeyboardResponse, stimulus: snd(t[`file1_snr${snr}`]), choices: "NO_KEYS", trial_ends_after_audio: true,
                      prompt: "<div class='fix'>+</div>", post_trial_gap: ISI_MS });
      timeline.push({ type: jsPsychAudioKeyboardResponse, stimulus: snd(t[`file2_snr${snr}`]), choices: ["y", "n"], response_allowed_while_playing: true,
                      trial_duration: RESP_WINDOW_MS + 1000, response_ends_trial: true, prompt: "<div class='fix'>+</div>", data: base,
                      on_finish: d => { d.key = d.response; d.correct = d.response === null ? null : ((d.response === "y") === (d.is_same === 1) ? 1 : 0); } });
      timeline.push({ type: jsPsychHtmlKeyboardResponse, choices: "NO_KEYS", trial_duration: FEEDBACK_MS,
                      stimulus: () => { const d = jsPsych.data.get().last(1).values()[0]; return `<p class='center'>${d.response === null ? "No response" : d.correct ? "Correct" : "Wrong"}</p>`; } });
    });
  });

  if (onPavlovia) timeline.push({ type: jsPsychPavlovia, command: "finish" });
  timeline.push({ type: jsPsychHtmlKeyboardResponse, choices: "NO_KEYS", trial_duration: 1500,
                  stimulus: "<p class='center'>That is the end of the test. Thank you very much!</p>" });
  jsPsych.run(timeline);
});
