"""Check that what is on Pavlovia is byte-for-byte what is in this repository.

    python online/check_pavlovia_deploy.py <account> <experiment>            # e.g.  LoraC choose
    python online/check_pavlovia_deploy.py <account> <experiment> --pilot snr_pilot
    python online/check_pavlovia_deploy.py --write-manifest                  # rewrite online/calib_pilot/MANIFEST.sha256
    python online/check_pavlovia_deploy.py --local                           # only: do the local files match the manifest?

Downloads every file the experiment needs from https://run.pavlovia.org/<account>/<experiment>/ and compares size
and SHA-256 with the local copy. Exit status 0 only if all of them are identical. Standard library only.

Why this exists: a participant could not start the test because js/jspsych.js on Pavlovia was the first 20000 bytes
of the real file (98520). Nothing on the page or in the upload step said so. Files that are created by pasting text
(GitLab's "New file", a terminal editor) can be cut short or re-indented; files that are uploaded or pushed with git
cannot. Run this after every upload.

MANIFEST.sha256 lists the same files with their sizes and hashes, in the format `sha256sum -c` reads, so the folder
can also be checked on a machine that has no Python (`cd online/calib_pilot && sha256sum -c MANIFEST.sha256`).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = ["index.html", "experiment.js", "conditions.json", "js/jspsych.js", "js/jspsych.css",
         "js/plugin-html-keyboard-response.js", "js/plugin-audio-keyboard-response.js", "js/plugin-preload.js"]


def upload_list(pilot):
    """The files that have to be on Pavlovia: the fixed ones plus every stimulus bundle (bundles/index.json is only a
    packing log and is not loaded by the page)."""
    d = os.path.join(HERE, pilot, "bundles")
    bundles = sorted(f"bundles/{f}" for f in os.listdir(d) if f.startswith("stim_") and f.endswith(".json"))
    return FIXED + bundles


def digest(b):
    return hashlib.sha256(b).hexdigest()


def read_local(pilot, rel):
    with open(os.path.join(HERE, pilot, rel), "rb") as f:
        return f.read()


def manifest_path(pilot):
    return os.path.join(HERE, pilot, "MANIFEST.sha256")


def write_manifest(pilot):
    lines = [f"# {pilot}: the files to upload. `sha256sum -c MANIFEST.sha256` checks them; sizes in bytes are in the comment column.",
             "# Rewrite with: python online/check_pavlovia_deploy.py --write-manifest"]
    for rel in upload_list(pilot):
        b = read_local(pilot, rel)
        lines.append(f"{digest(b)}  {rel}")
        lines.append(f"# size {len(b)}  {rel}")
    with open(manifest_path(pilot), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(manifest_path(pilot), os.path.dirname(HERE))}: {len(upload_list(pilot))} files")


def read_manifest(pilot):
    out = {}
    with open(manifest_path(pilot), encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                h, rel = line.rstrip("\n").split("  ", 1)
                out[rel] = h
    return out


def check_local(pilot):
    """Local files against the manifest: catches a manifest that was not rewritten after an edit."""
    man = read_manifest(pilot); files = upload_list(pilot); bad = []
    for rel in files:
        if man.get(rel) != digest(read_local(pilot, rel)):
            bad.append(rel)
    bad += [f"{rel} (in the manifest, not in the folder)" for rel in man if rel not in files]
    return bad


def fetch(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "deploy-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def describe(local, remote):
    """One line on HOW the two differ, because the cause decides the fix."""
    if local.startswith(remote):
        return f"cut short: the first {len(remote)} of {len(local)} bytes (created by pasting text? upload the file instead)"
    strip = lambda b: [ln.strip() for ln in b.decode("utf-8", "replace").strip().split("\n")]
    try:
        if strip(local) == strip(remote):
            return "same text, different indentation or line ends (pasted into an editor that re-indents; harmless for .js / .html, but upload the file to be sure)"
    except Exception:
        pass
    n = next((i for i, (x, y) in enumerate(zip(local, remote)) if x != y), min(len(local), len(remote)))
    return f"different content from byte {n} on (an older version?)"


def main():
    ap = argparse.ArgumentParser(description="Compare the experiment on Pavlovia with the local folder.")
    ap.add_argument("account", nargs="?"); ap.add_argument("experiment", nargs="?")
    ap.add_argument("--pilot", default="calib_pilot", help="folder under online/ (default calib_pilot)")
    ap.add_argument("--write-manifest", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0); ap.add_argument("--json", default=None, help="also write the comparison here")
    a = ap.parse_args()
    if a.write_manifest:
        write_manifest(a.pilot); return 0
    stale = check_local(a.pilot) if os.path.exists(manifest_path(a.pilot)) else ["MANIFEST.sha256 is missing"]
    if stale:
        print("local files do not match MANIFEST.sha256 (rewrite it with --write-manifest):\n  " + "\n  ".join(stale))
    if a.local:
        print("local files match the manifest" if not stale else ""); return 1 if stale else 0
    if not (a.account and a.experiment):
        ap.error("give the Pavlovia account and the experiment name")
    base = f"https://run.pavlovia.org/{a.account}/{a.experiment}/"
    rows, bad = [], 0
    print(f"{base}\n{'file':40s} {'local':>9s} {'deployed':>9s}  result")
    for rel in upload_list(a.pilot):
        local = read_local(a.pilot, rel)
        try:
            remote = fetch(base + rel, a.timeout); note = "identical" if remote == local else describe(local, remote)
        except urllib.error.HTTPError as e:
            remote, note = None, f"not on the server (HTTP {e.code})"
        except Exception as e:                                             # network trouble is not a verdict on the upload
            remote, note = None, f"could not be downloaded ({type(e).__name__}: {e})"
        ok = remote == local; bad += not ok
        rows.append(dict(file=rel, local_size=len(local), deployed_size=None if remote is None else len(remote), identical=ok, note=note,
                         local_sha256=digest(local), deployed_sha256=None if remote is None else digest(remote)))
        print(f"{rel:40s} {len(local):9d} {('-' if remote is None else str(len(remote))):>9s}  {'OK' if ok else 'DIFFERENT: ' + note}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(base=base, files=rows), f, indent=1)
    print(f"\n{len(rows) - bad} of {len(rows)} files identical." + ("" if not bad else f"  {bad} to upload again (as files, not pasted text)."))
    return 1 if (bad or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
