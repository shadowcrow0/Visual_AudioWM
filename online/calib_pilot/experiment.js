/* Online sound-calibration pilot (jsPsych 7.3.4): the consonant same/different test at ONE SNR, 216 different + 108 same
 * trials spread along the acoustic axis (online/make_calib_pilot_stimuli.py). Same trial timing as online/snr_pilot:
 * two syllables 0.5 s apart, y = same / n = different, answer from the onset of syllable 2 within 4 s, 0.6 s feedback.
 * Stimuli are pre-mixed; the browser only plays files and records keys.
 *
 * Before anything loads the page asks for: participant id (never the placeholder "test"), native language, whether
 * English was spoken at home in childhood, and the age English was first learned. All four are required and are written
 * on every data row.
 *
 * URL parameters:  ?participant=ID          prefills the id
 *                  ?skipcheck=1             no headphone test (testing only)
 *                  ?order_seed=N            fixes the trial order (default: drawn at random and logged)
 *                  ?maxtrials=N             only the first N trials of the order (testing only; logged as test_maxtrials)
 * Data columns per pair trial: participant, design, snr_db, order_seed, native_language, english_at_home, english_start_age,
 * task=pair, trial, row, is_same, consonant, foil, talker, group, pair, pair_type, voicing_only, x_acoustic, x_acoustic_pair,
 * bin, mn_dprime, feature_distance, snd1, snd2, seed1, seed2, key, rt, correct.
 * seed1 / seed2 are STRINGS: a 64-bit integer stored as a JS number is rounded and can no longer rebuild the stimulus. */

const params = new URLSearchParams(window.location.search);
const SKIP_CHECK = params.get("skipcheck") === "1";
const MAX_TRIALS = params.get("maxtrials") ? parseInt(params.get("maxtrials")) : null;
const ORDER_SEED = params.get("order_seed") ? (parseInt(params.get("order_seed")) >>> 0) : (Math.floor(Math.random() * 4294967296) >>> 0);
const ISI_MS = 500, RESP_WINDOW_MS = 3000, FEEDBACK_MS = 600, HP_PASS = 5, BREAK_EVERY = 81;
const BAD_IDS = ["", "test", "anon", "participant", "id", "xx"];
const onPavlovia = typeof jsPsychPavlovia !== "undefined";

// Seeded shuffle (mulberry32 + Fisher-Yates) so the order can be rebuilt from order_seed.
function shuffled(n, seed) {
  let a = seed >>> 0;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx;
}

// ---- intake form: plain DOM, so no extra jsPsych plugin has to be uploaded ----
function intake() {
  return new Promise(resolve => {
    document.body.innerHTML = `
      <form id="intake" class="intake" autocomplete="off">
        <h2>Consonant listening test</h2>
        <p>About 20 minutes. Please use headphones and sit somewhere quiet.</p>
        <label>Participant code (given to you by the researcher)<br><input id="pid" type="text" maxlength="24"></label>
        <label>What is your native language (the first language you learned)?<br><input id="lang" type="text" maxlength="40"></label>
        <label>Was English spoken in your home when you were a child?<br>
          <select id="home"><option value="">choose…</option><option value="yes">yes</option><option value="no">no</option></select></label>
        <label>At what age did you start learning English? (0 if from birth)<br><input id="age" type="number" min="0" max="99" step="1"></label>
        <p id="err" class="err"></p>
        <button type="submit">Continue</button>
      </form>`;
    const $ = id => document.getElementById(id);
    $("pid").value = params.get("participant") || "";
    $("intake").addEventListener("submit", ev => {
      ev.preventDefault();
      const pid = $("pid").value.trim(), lang = $("lang").value.trim(), home = $("home").value, age = $("age").value.trim();
      const bad = [];
      if (BAD_IDS.includes(pid.toLowerCase()) || !/^[A-Za-z0-9_-]+$/.test(pid)) bad.push("a participant code (letters, digits, - or _; not a placeholder such as “test”)");
      if (lang.length < 2) bad.push("your native language");
      if (!home) bad.push("whether English was spoken at home");
      if (age === "" || isNaN(parseInt(age)) || parseInt(age) < 0) bad.push("the age you started learning English");
      if (bad.length) { $("err").textContent = "Please fill in: " + bad.join("; ") + "."; return; }
      resolve({ participant: pid, native_language: lang, english_at_home: home, english_start_age: parseInt(age) });
    });
  });
}

intake().then(who => {
  const PID = who.participant;
  const jsPsych = initJsPsych({
    on_finish: () => {
      if (!onPavlovia) {
        const csv = jsPsych.data.get().csv();
        const pre = document.createElement("pre"); pre.id = "jspsych-data"; pre.textContent = csv; document.body.appendChild(pre);
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = `${PID}_calib_pilot.csv`;
        a.textContent = "Download the data file"; a.style.cssText = "display:block;color:#9cf;font-size:24px;margin:40px";
        document.body.appendChild(a);
      }
    }
  });
  window.jsPsychInstance = jsPsych;      // test hook

  // The sounds come as base64 bundles (online/pack_stimuli.py calib_pilot); each wav becomes a Blob URL.
  const SRC = {};
  const b64url = (b64) => { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return URL.createObjectURL(new Blob([u], { type: "audio/wav" })); };
  const snd = (name) => SRC[name.replace(/^stim\//, "")] || name;
  document.body.innerHTML = "<p style='font-size:22px;margin:40px'>Loading the sounds (about 25 MB)…</p>";
  const PARTS = ["p1", "p2", "p3", "p4", "p5", "p6"];
  const needed = (SKIP_CHECK ? [] : ["hp"]).concat(PARTS);
  Promise.all([fetch("conditions.json").then(r => r.json())].concat(needed.map(g => fetch(`bundles/stim_${g}.json`).then(r => r.json()))))
  .then(([cond, ...bundles]) => {
    bundles.forEach(b => Object.entries(b).forEach(([name, b64]) => { SRC[name] = b64url(b64); }));
    document.body.innerHTML = "";
    window.__hp = cond.headphone_check;   // test hook: lets an automated driver answer the headphone check
    jsPsych.data.addProperties({ participant: PID, design: cond.design, plan_seed: cond.plan_seed, snr_db: cond.snr_db, order_seed: String(ORDER_SEED),
                                 native_language: who.native_language, english_at_home: who.english_at_home, english_start_age: who.english_start_age,
                                 test_maxtrials: MAX_TRIALS === null ? "" : MAX_TRIALS, user_agent: navigator.userAgent });
    let order = shuffled(cond.trials.length, ORDER_SEED);
    if (MAX_TRIALS !== null) order = order.slice(0, MAX_TRIALS);
    const timeline = [];
    if (onPavlovia) timeline.push({ type: jsPsychPavlovia, command: "init" });

    const files = [];
    if (!SKIP_CHECK) cond.headphone_check.forEach(t => t.files.forEach(f => files.push(snd(f))));
    order.forEach(i => { files.push(snd(cond.trials[i].file1)); files.push(snd(cond.trials[i].file2)); });
    timeline.push({ type: jsPsychPreload, audio: files, show_progress_bar: true, max_load_time: 600000,
                    message: "<p class='center'>Preparing the sounds…</p>" });

    const page = (html) => ({ type: jsPsychHtmlKeyboardResponse, stimulus: html, choices: [" "], post_trial_gap: 300 });
    const fix = (ms) => ({ type: jsPsychHtmlKeyboardResponse, stimulus: "<div class='fix'>+</div>", choices: "NO_KEYS", trial_duration: ms });

    timeline.push(page(`<p>Thank you. Set the volume to a comfortable level now (it will not change during the test).</p>
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

    // ---- the pair test ----
    const nBlocks = Math.ceil(order.length / BREAK_EVERY);
    timeline.push(page(`<p>You will hear two syllables, one after the other, in background noise.</p>
      <p><b>Is the CONSONANT the same in both?</b></p>
      <p class='center'>Y = same &nbsp;&nbsp;&nbsp;&nbsp; N = different</p>
      <p>The background noise is different every time; listen to the consonant only. Answer after the second syllable, as quickly as you can, on every trial. You will see whether you were right.</p>
      <p>There are ${order.length} trials in ${nBlocks} parts, with a short break between parts.</p>
      <p>Press the space bar to begin.</p>`));

    order.forEach((idx, k) => {
      if (k > 0 && k % BREAK_EVERY === 0)
        timeline.push(page(`<p>Part ${k / BREAK_EVERY} of ${nBlocks} is done. Take a short break if you like.</p><p>Press the space bar to continue.</p>`));
      const t = cond.trials[idx];
      const base = { task: "pair", trial: k + 1, row: t.row, is_same: t.is_same, consonant: t.consonant, foil: t.foil, talker: t.talker, group: t.group,
                     pair: t.pair, pair_type: t.pair_type, voicing_only: t.voicing_only, x_acoustic: t.x_acoustic, x_acoustic_pair: t.x_acoustic_pair,
                     bin: t.bin, mn_dprime: t.mn_dprime, feature_distance: t.feature_distance,
                     snd1: t.snd1, snd2: t.snd2, seed1: String(t.seed1), seed2: String(t.seed2) };
      timeline.push(fix(300));
      timeline.push({ type: jsPsychAudioKeyboardResponse, stimulus: snd(t.file1), choices: "NO_KEYS", trial_ends_after_audio: true,
                      prompt: "<div class='fix'>+</div>", post_trial_gap: ISI_MS });
      timeline.push({ type: jsPsychAudioKeyboardResponse, stimulus: snd(t.file2), choices: ["y", "n"], response_allowed_while_playing: true,
                      trial_duration: RESP_WINDOW_MS + 1000, response_ends_trial: true, prompt: "<div class='fix'>+</div>", data: base,
                      on_start: () => { window.__pair = { trial: k + 1, is_same: t.is_same, x: t.x_acoustic }; },     // test hook
                      on_finish: d => { window.__pair = null; d.key = d.response; d.correct = d.response === null ? null : ((d.response === "y") === (d.is_same === 1) ? 1 : 0); } });
      timeline.push({ type: jsPsychHtmlKeyboardResponse, choices: "NO_KEYS", trial_duration: FEEDBACK_MS,
                      stimulus: () => { const d = jsPsych.data.get().last(1).values()[0]; return `<p class='center'>${d.response === null ? "No response" : d.correct ? "Correct" : "Wrong"}</p>`; } });
    });

    if (onPavlovia) timeline.push({ type: jsPsychPavlovia, command: "finish" });
    timeline.push({ type: jsPsychHtmlKeyboardResponse, choices: "NO_KEYS", trial_duration: 1500,
                    stimulus: "<p class='center'>That is the end of the test. Thank you very much!</p>" });
    jsPsych.run(timeline);
  });
});
