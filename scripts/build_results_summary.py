"""Build outputs/results_summary.json — the machine-readable source of
truth for every headline number in the README, with a full run manifest —
and verify that nothing has drifted.

History: round-2 review caught README figures drifting from rebuilt
artifacts; round-3 review caught the first version of THIS file giving
false assurance (it only substring-matched seven figures, so a stale
manifest still reported "in sync"). The verify mode now checks:

  * every file hash recorded in the manifest against the working tree
    (the audit record is hashed EXCLUDING its LLM narrative fields, so
    regenerating the report does not invalidate the manifest);
  * that `git_commit` was recorded (generation on a git-less sandbox is
    allowed, but verification then fails until regenerated in-repo);
  * that all quoted figures appear in the README;
  * that known-stale figures do NOT appear in the README.

All numeric results are recomputed here from committed inputs at
CANONICAL parameters — including the Taitung raw run (previously a
hard-coded constant, flagged in round 3).

Usage:
    python scripts/build_results_summary.py              # compute + write
    python scripts/build_results_summary.py --verify     # full drift check
    python scripts/build_results_summary.py --check-readme   # alias
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TRUTH_MAG = {"0403": 7.2, "dapu": 6.4, "eq2026053": 4.7}
REPLAYS = {"0403": "outputs/replay_0403.json",
           "dapu": "outputs/replay_dapu.json",
           "eq2026053": "outputs/replay_eq2026053.json"}
AUDIT = "outputs/audit_archive/audit_eq2026053.json"
NARRATIVE_FIELDS = ("report_md", "report_model")


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_content_sha256(p: Path) -> str | None:
    """Hash of the audit record MINUS the LLM narrative — regenerating the
    report must not invalidate the numeric manifest."""
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    for k in NARRATIVE_FIELDS:
        rec.pop(k, None)
    return hashlib.sha256(
        json.dumps(rec, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    """True if the working tree differs from HEAD in any NON-derived path.

    Round-4 review: a summary generated on a dirty tree cannot claim HEAD
    as its source. But the two-phase protocol REGENERATES the derived
    artifacts on the clean code tree, which necessarily makes the
    DERIVED_PATHS dirty before the second commit — so those paths are
    exempt. What this flag guarantees is exactly that the CODE that
    produced the numbers is the recorded commit, nothing more."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:          # rename: "old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"').replace("\\", "/")
        if not any(path.startswith(d) for d in DERIVED_PATHS):
            return True
    return False


# files whose content determines the numeric results — hashed into the
# manifest so provenance holds even if the commit bookkeeping is imperfect
SOURCE_FILES = [
    "scripts/build_results_summary.py",
    "scripts/demo_convergence.py",   # round-5: load_replay_json() lives
                                     # here and feeds every computation
    "src/edgequake/location/replay_sim.py",
    "src/edgequake/location/locator.py",
    "src/edgequake/location/magnitude.py",
    "src/edgequake/location/site.py",
]

# the ONLY paths allowed to differ between the recorded commit and HEAD
# (two-phase protocol: commit code first, generate on the clean tree,
# commit derived artifacts second). Keep entries as exact files where
# possible; "outputs/audit_archive/" (per-event records/reports/logs) and
# "outputs/replay_eq" (per-event CWA replay artifacts) are deliberate
# prefixes because the audit workflow creates new per-event files.
DERIVED_PATHS = ("outputs/results_summary.json",
                 "outputs/reproduction_report.json",
                 "outputs/audit_archive/", "docs/audit.json",
                 "vercel/audit.json", "docs/index.html",
                 "vercel/index.html", "outputs/replay_eq")


def verify() -> None:
    out_path = ROOT / "outputs" / "results_summary.json"
    s = json.loads(out_path.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    fails = []

    if not s.get("git_commit"):
        fails.append("git_commit is null — regenerate the summary inside "
                     "the git repo (python scripts/build_results_summary.py)")
    if s.get("working_tree_dirty") is not False:
        fails.append("summary was generated on a dirty (or unknown) "
                     "working tree — commit code first, then regenerate "
                     "on the clean tree (two-phase protocol)")
    # round-5: the recorded flag only proves the tree was clean at
    # GENERATION time — also re-check the tree NOW, so uncommitted edits
    # to any non-derived file fail verification even if that file is not
    # individually hashed below
    if git_dirty() is not False:
        fails.append("working tree is dirty in a non-derived path RIGHT "
                     "NOW (or git is unavailable) — commit or revert "
                     "before verifying")
    rec_commit = s.get("git_commit")
    if rec_commit:
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT,
                stderr=subprocess.DEVNULL).decode().strip()
            if rec_commit != head:
                anc = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", rec_commit,
                     "HEAD"], cwd=ROOT).returncode == 0
                if not anc:
                    fails.append(f"recorded commit {rec_commit[:10]} is "
                                 "not an ancestor of HEAD")
                else:
                    diff = subprocess.check_output(
                        ["git", "diff", "--name-only",
                         f"{rec_commit}..HEAD"], cwd=ROOT).decode().split()
                    bad = [f for f in diff
                           if not any(f.startswith(d)
                                      for d in DERIVED_PATHS)]
                    if bad:
                        fails.append(
                            "commits since the recorded commit touch "
                            "NON-derived files (source drifted): "
                            + ", ".join(bad[:8]))
        except Exception as e:
            fails.append(f"git provenance check unavailable: {e}")
    for path, want in s.get("source_file_sha256", {}).items():
        got = sha256(ROOT / path)
        if got != want:
            fails.append(f"source file drifted since generation: {path}")
    for path, want in s.get("file_sha256", {}).items():
        got = sha256(ROOT / path)
        if want is None or got is None:
            # a missing file must never verify as "unchanged"
            fails.append(f"file absent (manifest or working tree): {path}")
        elif got != want:
            fails.append(f"hash drift: {path}\n    manifest {want}\n"
                         f"    actual   {got}")
    want = s.get("audit_content_sha256")
    got = audit_content_sha256(ROOT / AUDIT)
    if want != got:
        fails.append(f"audit numeric content drifted: {AUDIT}")

    # semantic checks on the reproduction report (round 4: verifying the
    # file's hash proves it did not change, not that its content holds)
    rp = ROOT / "outputs" / "reproduction_report.json"
    if not rp.exists():
        fails.append("outputs/reproduction_report.json missing")
    else:
        rep = json.loads(rp.read_text(encoding="utf-8"))
        if rep.get("verdict") != "identical_canonical_json":
            fails.append("reproduction report verdict is "
                         f"{rep.get('verdict')!r}, not "
                         "'identical_canonical_json'")
        # round-5: require the report to cover EXACTLY the full event
        # set — a subset or superset (stray/unknown events) must fail,
        # not just a missing one
        want_ev = ["0403", "dapu"]
        if sorted(rep.get("events_requested") or []) != want_ev:
            fails.append("reproduction report events_requested is "
                         f"{rep.get('events_requested')!r}, expected "
                         f"{want_ev}")
        if sorted(rep.get("events", {})) != want_ev:
            fails.append("reproduction report events are "
                         f"{sorted(rep.get('events', {}))}, expected "
                         f"exactly {want_ev}")
        v3 = sha256(ROOT / "outputs" / "v3_verify_x83.pt")
        ft = sha256(ROOT / "outputs" / "phasenet_cwa_ft.pt")
        if rep.get("checkpoint", {}).get("sha256") != v3:
            fails.append("reproduction report checkpoint hash != current "
                         "outputs/v3_verify_x83.pt")
        if v3 != ft:
            fails.append("checkpoint identity broken: v3_verify_x83.pt "
                         "!= phasenet_cwa_ft.pt")
        for e, ev in rep.get("events", {}).items():
            cur = sha256(ROOT / ev["tracked_artifact"])
            if ev.get("tracked_artifact_sha256") != cur:
                fails.append(f"replay artifact changed since "
                             f"reproduction: {ev['tracked_artifact']}")
    for q in s.get("readme_quotes", []):
        if q not in readme:
            fails.append(f"README missing quoted figure: {q!r}")
    for q in s.get("forbidden_quotes", []):
        if q in readme:
            fails.append(f"README still contains stale figure: {q!r}")

    if fails:
        print(f"[summary] VERIFY FAILED ({len(fails)} problems):")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"[summary] verify OK — {len(s.get('file_sha256', {}))} file "
          f"hashes, {len(s.get('readme_quotes', []))} quoted figures, "
          f"{len(s.get('forbidden_quotes', []))} stale figures absent, "
          f"git_commit {s['git_commit'][:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", "--check-readme", dest="verify",
                    action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
        return

    from demo_convergence import load_replay_json
    from edgequake.location.replay_sim import CANONICAL, simulate
    from edgequake.location.site import load_site_terms

    site_path = ROOT / "outputs" / "site_terms.json"
    site_terms = load_site_terms(site_path)

    events = {}
    for key, rel in REPLAYS.items():
        src = ROOT / rel
        ev, tr = load_replay_json(src)
        row = {"input": rel,
               "picker_in_artifact": json.loads(
                   src.read_text(encoding="utf-8")).get("picker"),
               "truth_mag": TRUTH_MAG[key]}
        for label, st in (("raw", None), ("site_corrected", site_terms)):
            p = simulate(ev, tr, site_terms=st)   # all params = CANONICAL
            fs = [f for f in p["frames"] if f.get("mag") is not None]
            first, last = fs[0], fs[-1]
            orel = p["origin_rel"] or 0.0
            floc = next(f for f in p["frames"] if "lat" in f)
            row[label] = {
                "final_mag": last["mag"], "final_err_km": last["err"],
                "abs_dmag": round(abs(last["mag"] - TRUTH_MAG[key]), 2),
                "first_loc_s": round(floc["t"] - orel, 1),
                "first_mag_s": round(first["t"] - orel, 1),
                "first_mag": first["mag"], "bconf": last.get("bconf"),
            }
        events[key] = row

    # audit-record timing fields for Taitung (same CANONICAL config; the
    # record additionally carries EEW/alert decisions and exposure)
    audit_p = ROOT / AUDIT
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    events["eq2026053"]["audit_record"] = {
        "t_first_loc_s": audit["t_first_loc_s"],
        "t_eew_s": audit.get("t_eew_s"),
        "alert_fired": audit["alert_fired"],
        "note": "post-hoc arrival-time replay; times are lower bounds",
    }

    ks = ("0403", "dapu", "eq2026053")
    mean_raw = round(sum(events[k]["raw"]["abs_dmag"] for k in ks)
                     / len(ks), 2)
    mean_site = round(sum(events[k]["site_corrected"]["abs_dmag"]
                          for k in ks) / len(ks), 2)

    ckpt_v3 = ROOT / "outputs" / "v3_verify_x83.pt"
    ckpt_ft = ROOT / "outputs" / "phasenet_cwa_ft.pt"
    file_hashes = {
        "outputs/v3_verify_x83.pt": sha256(ckpt_v3),
        "outputs/phasenet_cwa_ft.pt": sha256(ckpt_ft),
        "outputs/site_terms.json": sha256(site_path),
        "outputs/reproduction_report.json":
            sha256(ROOT / "outputs" / "reproduction_report.json"),
        # round-5: the ORIGINAL live-run log is historical evidence —
        # pinned here so neither it nor its canonical recomputation can
        # drift silently (their relationship is documented in the audit
        # record's provenance block)
        "outputs/audit_archive/audit_202607310058-EQ2026053-Waveform.txt":
            sha256(ROOT / "outputs" / "audit_archive" /
                   "audit_202607310058-EQ2026053-Waveform.txt"),
        **{rel: sha256(ROOT / rel) for rel in REPLAYS.values()},
    }

    commit = git_commit()
    dirty = git_dirty()
    if commit is None:
        print("[summary] WARNING: not a git checkout — git_commit is null "
              "and --verify WILL FAIL until regenerated inside the repo")
    elif dirty:
        print("[summary] WARNING: working tree is DIRTY — --verify will "
              "fail; commit code first, then regenerate (two-phase "
              "protocol)")

    e0, ed, et = events["0403"], events["dapu"], events["eq2026053"]
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                      time.gmtime()),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "git_commit_note": "two-phase protocol: code is committed FIRST; "
            "this summary is generated on that clean tree and lands in a "
            "second, derived-artifacts-only commit. --verify enforces "
            "that the recorded commit is HEAD or an ancestor whose diff "
            "to HEAD touches only derived paths, and that at generation "
            "time no NON-derived path was dirty (derived paths are "
            "necessarily dirty mid-protocol, so they are exempt from the "
            "dirty check — they are pinned by file_sha256 instead).",
        "source_file_sha256": {f: sha256(ROOT / f) for f in SOURCE_FILES},
        "parameters": dict(
            CANONICAL,
            velocity_model="homogeneous half-space, vp from CANONICAL, "
                           "vs=vp/1.73",
            note="all three events computed by simulate() with every "
                 "parameter defaulting to CANONICAL (no per-call-site "
                 "overrides); point estimates verified bootstrap-count "
                 "invariant for N in {20,30,60,100,200}",
        ),
        "checkpoint_identity": {
            "finding": "outputs/v3_verify_x83.pt and "
                "outputs/phasenet_cwa_ft.pt are byte-identical "
                "(same sha256) — the replay artifacts were produced with "
                "the committed fine-tuned weights under an alternate "
                "filename",
            "reproduction": "outputs/reproduction_report.json — machine-"
                "generated record: raw-waveform + dataless-inventory + "
                "checkpoint hashes, environment versions, canonical "
                "whole-JSON comparison, verdict "
                "identical_canonical_json; regenerate with "
                "scripts/verify_replay_reproduction.py (requires the "
                "GDMS raw waveforms, which are too large for the repo)",
            "training_provenance": "the checkpoint's training run is only "
                "partially traceable — a different, weaker provenance "
                "level than the replay artifacts, which are fully "
                "reproducible",
        },
        "file_sha256": file_hashes,
        "audit_content_sha256": audit_content_sha256(audit_p),
        "audit_content_note": "audit record hashed excluding "
            f"{list(NARRATIVE_FIELDS)} so LLM-report regeneration does "
            "not invalidate this manifest",
        "site_correction": {"file": "outputs/site_terms.json",
                            "n_stations": len(site_terms)},
        "events": events,
        "mean_abs_dmag": {"raw": mean_raw, "site_corrected": mean_site},
        "readme_quotes": [
            f"M{e0['site_corrected']['final_mag']:.2f}",
            f"M{ed['site_corrected']['final_mag']:.2f}",
            f"M{et['site_corrected']['final_mag']:.2f}",
            f"origin+{et['audit_record']['t_first_loc_s']}",
            f"origin+{et['audit_record']['t_eew_s']}",
            f"{mean_raw} → {mean_site}",
            "0.702", "0.635",
        ],
        "forbidden_quotes": ["M7.11", "M6.34", "0.680", "1.3 km",
                             "0.17 → 0.09", "M7.39", "M6.65 (Δ"],
    }

    out_path = ROOT / "outputs" / "results_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"[summary] wrote {out_path}")
    for k in ks:
        r, s = events[k]["raw"], events[k]["site_corrected"]
        print(f"[summary] {k}: raw M{r['final_mag']:.2f} "
              f"(d{r['abs_dmag']:.2f}, loc+{r['first_loc_s']}s) -> site "
              f"M{s['final_mag']:.2f} (d{s['abs_dmag']:.2f}) "
              f"err {s['final_err_km']:.1f} km")
    print(f"[summary] mean |dM| raw {mean_raw} -> site {mean_site}")


if __name__ == "__main__":
    main()
