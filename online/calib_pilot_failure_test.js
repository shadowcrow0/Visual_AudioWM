/* What does the calibration pilot do when an uploaded file is broken? Headless Chromium, no Pavlovia.
 *
 *   node online/calib_pilot_failure_test.js
 *
 * Each case serves a copy of online/calib_pilot/ with one file damaged and reports what the participant would see.
 *   1  js/jspsych.js cut to 20000 bytes (the real incident), CDN reachable  -> the library comes from the CDN, the test starts
 *   2  the same, CDN unreachable                                            -> a message naming js/jspsych.js
 *   3  one plugin cut short, CDN unreachable                                -> a message naming that plugin
 *   4  a stimulus bundle missing                                            -> a message naming the bundle
 *   5  conditions.json cut short                                            -> a message naming conditions.json
 *   6  a bundle swapped for another one (valid JSON, wrong sounds)          -> a message counting the missing sounds
 * The CDN is not reachable from the dev container, so in case 1 the CDN address is answered from the repository's own
 * copy of the file (which is byte-identical to the npm package); the browser still verifies the integrity hash.
 * "Started" means jsPsych has built its display (#jspsych-content); getCurrentTrial() is a truthy {} before that.
 * Exit status 0 only if every case behaves as listed. Needs the Playwright that ships with the dev container. */
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs"), os = require("os"), path = require("path");
const SRC = path.join(__dirname, "calib_pilot"), PORT = 8766;
const CDN = { "jspsych@7.3.4": "js/jspsych.js", "plugin-html-keyboard-response@1.1.3": "js/plugin-html-keyboard-response.js",
              "plugin-audio-keyboard-response@1.1.3": "js/plugin-audio-keyboard-response.js", "plugin-preload@1.1.3": "js/plugin-preload.js" };

function stage(damage) {                       // a copy made of links, except the files a case damages
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "calibfail-"));
  for (const d of ["js", "bundles"]) {
    fs.mkdirSync(path.join(dir, d));
    for (const f of fs.readdirSync(path.join(SRC, d))) fs.symlinkSync(path.join(SRC, d, f), path.join(dir, d, f));
  }
  for (const f of ["index.html", "experiment.js", "conditions.json"]) fs.symlinkSync(path.join(SRC, f), path.join(dir, f));
  damage((rel, fn) => { const p = path.join(dir, rel); const b = fs.readFileSync(path.join(SRC, rel)); fs.unlinkSync(p); const out = fn(b); if (out !== null) fs.writeFileSync(p, out); });
  return dir;
}

const CASES = [
  ["1 jspsych.js cut short, CDN reachable", e => e("js/jspsych.js", b => b.subarray(0, 20000)), true, s => s.started && !s.fatal],
  ["2 jspsych.js cut short, CDN unreachable", e => e("js/jspsych.js", b => b.subarray(0, 20000)), false, s => !s.started && /js\/jspsych\.js/.test(s.fatal)],
  ["3 a plugin cut short, CDN unreachable", e => e("js/plugin-preload.js", b => b.subarray(0, 3000)), false, s => !s.started && /plugin-preload\.js/.test(s.fatal) && !/jspsych\.js/.test(s.fatal)],
  ["4 a bundle missing", e => e("bundles/stim_p3.json", () => null), false, s => !s.started && /stim_p3\.json/.test(s.fatal)],
  ["5 conditions.json cut short", e => e("conditions.json", b => b.subarray(0, 20000)), false, s => !s.started && /conditions\.json/.test(s.fatal)],
  ["6 a bundle swapped for another", e => e("bundles/stim_p3.json", () => fs.readFileSync(path.join(SRC, "bundles/stim_p2.json"))), false, s => !s.started && /108 of the/.test(s.fatal)],
];

(async () => {
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  let failed = 0;
  for (const [name, damage, cdnUp, expect] of CASES) {
    const dir = stage(damage);
    const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: dir, stdio: "ignore" });
    await new Promise(r => setTimeout(r, 1200));
    const page = await browser.newPage(); const cdnHits = [];
    await page.route("**/lib.pavlovia.org/**", r => r.abort());
    await page.route("**/unpkg.com/**", r => {
      const key = Object.keys(CDN).find(k => r.request().url().includes(k)); cdnHits.push(key || r.request().url());
      if (!cdnUp || !key) return r.abort();
      return r.fulfill({ status: 200, contentType: "application/javascript", headers: { "access-control-allow-origin": "*" }, body: fs.readFileSync(path.join(SRC, CDN[key])) });
    });
    await page.goto(`http://127.0.0.1:${PORT}/index.html?participant=FAILTEST&skipcheck=1&maxtrials=2`).catch(() => {});
    await page.waitForFunction(() => document.querySelector(".fatal") || document.querySelector("#jspsych-content"), null, { timeout: 90000 }).catch(() => {});
    const s = await page.evaluate(() => ({ fatal: document.querySelector(".fatal") ? document.querySelector(".fatal").innerText.replace(/\s+/g, " ") : "",
                                           started: !!document.querySelector("#jspsych-content"), body: document.body.innerText.slice(0, 120) }));
    const ok = expect(s); failed += !ok;
    console.log(`${ok ? "ok  " : "FAIL"} ${name}\n     started: ${s.started} | CDN requests: ${cdnHits.length ? cdnHits.join(", ") : "none"}\n     shown: ${s.fatal || s.body.replace(/\s+/g, " ")}`);
    await page.close(); server.kill(); await new Promise(r => setTimeout(r, 300)); fs.rmSync(dir, { recursive: true, force: true });
  }
  // the undamaged folder must not touch the CDN at all
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: SRC, stdio: "ignore" });
  await new Promise(r => setTimeout(r, 1200));
  const page = await browser.newPage(); let cdn = 0;
  await page.route("**/lib.pavlovia.org/**", r => r.abort()); await page.route("**/unpkg.com/**", r => { cdn++; r.abort(); });
  await page.goto(`http://127.0.0.1:${PORT}/index.html?participant=FAILTEST&skipcheck=1&maxtrials=2`);
  await page.waitForFunction(() => document.querySelector("#jspsych-content"), null, { timeout: 90000 }).catch(() => {});
  const started = await page.evaluate(() => !!document.querySelector("#jspsych-content"));
  const ok = started && cdn === 0; failed += !ok;
  console.log(`${ok ? "ok  " : "FAIL"} 0 undamaged folder\n     started: ${started} | CDN requests: ${cdn}`);
  server.kill(); await browser.close();
  console.log(failed ? `${failed} case(s) did not behave as listed` : "all cases behave as listed");
  process.exit(failed ? 1 : 0);
})();
