/* Drives online/calib_pilot/ in headless Chromium from start to finish and saves the data file the page offers.
 *
 *   node online/calib_pilot_browser_test.js [out.csv] [extra query, e.g. "maxtrials=12&skipcheck=1"]
 *
 * It serves the folder on localhost, fills the intake form, answers the headphone check from the plan, and answers
 * every pair trial about 300 ms after the second syllable starts: correctly, except that every 7th trial is answered
 * wrongly and every 50th is left unanswered (so "Wrong" and "No response" are exercised too).
 * Needs the Playwright that ships with the dev container (require('playwright')); not part of the experiment. */
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs"), path = require("path");
const out = process.argv[2] || "calib_pilot_browser_test.csv";
const extra = process.argv[3] ? "&" + process.argv[3] : "";
const PORT = 8765, dir = path.join(__dirname, "calib_pilot");

(async () => {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: dir, stdio: "ignore" });
  await new Promise(r => setTimeout(r, 1200));
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const page = await browser.newPage();
  page.on("pageerror", e => console.log("PAGE ERROR:", e.message));
  await page.route("**/lib.pavlovia.org/**", r => r.abort());                // no Pavlovia here: the page falls back to local mode
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${PORT}/index.html?participant=test${extra}`);
  // the placeholder id must be refused
  await page.fill("#lang", "English"); await page.selectOption("#home", "yes"); await page.fill("#age", "0");
  await page.click("button[type=submit]");
  const refused = await page.textContent("#err");
  console.log("placeholder id refused:", /participant code/.test(refused));
  await page.fill("#pid", "BROWSERTEST"); await page.click("button[type=submit]");

  let last = -1, pairs = 0;
  for (;;) {
    if (await page.$("#jspsych-data")) break;
    const st = await page.evaluate(() => {
      const j = window.jsPsychInstance; if (!j) return null;
      const cur = j.getCurrentTrial(); if (!cur) return null;
      return { i: j.getProgress().current_trial_global, choices: Array.isArray(cur.choices) ? cur.choices : null, data: cur.data || {} };
    }).catch(() => null);
    if (st && st.choices && st.i !== last) {
      last = st.i;
      if (st.choices.includes(" ")) await page.keyboard.press("Space");
      else if (st.choices.includes("1")) await page.keyboard.press(String(st.data.answer));
      else if (st.choices.includes("y")) {
        pairs++;
        await page.waitForTimeout(300);
        const right = st.data.is_same === 1 ? "y" : "n", wrong = right === "y" ? "n" : "y";
        if (pairs % 50 !== 0) await page.keyboard.press(pairs % 7 === 0 ? wrong : right);
        if (pairs % 40 === 0) console.log(`  ${pairs} pair trials, ${((Date.now() - t0) / 60000).toFixed(1)} min`);
      }
    }
    await page.waitForTimeout(60);
  }
  const csv = await page.textContent("#jspsych-data");
  fs.writeFileSync(out, csv);
  console.log(`done: ${pairs} pair trials answered or skipped, ${((Date.now() - t0) / 60000).toFixed(1)} min, ${csv.length} bytes -> ${out}`);
  await browser.close(); server.kill();
})().catch(e => { console.error(e); process.exit(1); });
