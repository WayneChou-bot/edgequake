"""Build outputs/results_summary.json — the machine-readable source of
truth for every headline number in the README, with a full run manifest.

Born from an external review that caught README figures drifting from
rebuilt artifacts (different call sites used different max_stations).
The fix has three parts:
  1. one canonical parameter set (replay_sim.CANONICAL) shared by the
     dashboard, the audits, and this script;
  2. this manifest: parameters, checkpoint/site-file hashes, input
     artifact hashes, and per-event results in one committed JSON;
  3. `--check-readme`: verifies the README still quotes these numbers —
     run it locally or in CI so transcription drift fails loudly.

Usage:
    python scripts/build_results_summary.py            # compute + write
    python scripts/build_results_summary.py --check-readme
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

TRUTH_MAG = {"0403": 7.2, "dapu": 6.4}


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readme", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / "outputs" / "results_summary.json"

    if args.check_readme:
        summary = json.loads(out_path.read_text(encoding="utf-8"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        missing = [s for s in summary.get("readme_quotes", [])
                   if s not in readme]
        if missing:
            print("[summary] README OUT OF SYNC — missing quotes:")
            for s in missing:
                print("   ", s)
            sys.exit(1)
        print(f"[summary] README in sync "
              f"({len(summary.get('readme_quotes', []))} quoted figures)")
        return

    from demo_convergence import load_replay_json
    from edgequake.location.locator import PickLocator
    from edgequake.location.replay_sim import CANONICAL, simulate
    from edgequake.location.site import load_site_terms

    site_path = ROOT / "outputs" / "site_terms.json"
    site_terms = load_site_terms(site_path)
    loc = PickLocator(vp_km_s=CANONICAL["vp_km_s"])

    events = {}
    for key in ("0403", "dapu"):
        src = ROOT / "outputs" / f"replay_{key}.json"
        ev, tr = load_replay_json(src)
        row = {"input": f"outputs/replay_{key}.json",
               "input_sha256": sha256(src),
               "picker_in_artifact": json.loads(
                   src.read_text(encoding="utf-8")).get("picker"),
               "truth_mag": TRUTH_MAG[key]}
        for label, st in (("raw", None), ("site_corrected", site_terms)):
            p = simulate(ev, tr, max_stations=CANONICAL["max_stations"],
                         bootstrap=CANONICAL["bootstrap"], site_terms=st)
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

    # the Taitung audit record is produced by ingest_cwa_wave.py with the
    # same CANONICAL parameters; quote it rather than recompute
    audit_p = ROOT / "outputs" / "audit_archive" / "audit_eq2026053.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    events["eq2026053"] = {
        "input": "outputs/audit_archive/audit_eq2026053.json",
        "input_sha256": sha256(audit_p),
        "truth_mag": audit["cwa"]["mag"],
        "site_corrected": {
            "final_mag": audit["final_mag"],
            "final_err_km": audit["final_err_km"],
            "abs_dmag": round(abs(audit["final_mag"]
                                  - audit["cwa"]["mag"]), 2),
            "first_loc_s": audit["t_first_loc_s"],
            "first_mag_s": audit["t_first_mag_s"],
            "first_mag": audit["first_mag"],
            "t_eew_s": audit.get("t_eew_s"),
        },
        "note": "post-hoc arrival-time replay; times are lower bounds",
    }

    dm = [events[k]["site_corrected"]["abs_dmag"]
          for k in ("0403", "dapu", "eq2026053")]
    dm_raw = [events[k]["raw"]["abs_dmag"] for k in ("0403", "dapu")]
    dm_raw.append(round(abs(4.76 - 4.7), 2))   # taitung raw (pre-Phase-7
    # audit, outputs/audit_archive history); recomputed value in NOTES

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                      time.gmtime()),
        "git_commit": git_commit(),
        "parameters": dict(
            CANONICAL,
            velocity_model="homogeneous half-space, vp=6.2 km/s, "
                           "vs=vp/1.73",
            depth_grid_km=[round(float(z), 1) for z in loc.depth_grid],
            note="bootstrap-count invariance of point estimates verified "
                 "for N in {20,30,60,100,200} (identical to 3 decimals)",
        ),
        "checkpoints": {
            "outputs/v3_verify_x83.pt": {
                "sha256": sha256(ROOT / "outputs" / "v3_verify_x83.pt"),
                "provenance": "restored from a 2026-07-31 workspace "
                    "snapshot; verified by bit-exact reproduction of both "
                    "tracked replay artifacts (0403: 110 P/81 S picks; "
                    "dapu: 116 P/107 S picks — all arrival times, "
                    "probabilities and PGA identical). The checkpoint's "
                    "training run is only partially traceable; the replay "
                    "ARTIFACTS are fully reproducible, the training is "
                    "not — these are different provenance levels.",
            },
            "outputs/phasenet_cwa_ft.pt": {
                "sha256": sha256(ROOT / "outputs" / "phasenet_cwa_ft.pt")},
        },
        "site_correction": {
            "file": "outputs/site_terms.json",
            "sha256": sha256(site_path),
            "n_stations": len(site_terms),
        },
        "events": events,
        "mean_abs_dmag": {"raw": round(sum(dm_raw) / len(dm_raw), 2),
                          "site_corrected": round(sum(dm) / len(dm), 2)},
    }

    # figures the README quotes verbatim — --check-readme enforces these
    e0, ed, et = events["0403"], events["dapu"], events["eq2026053"]
    summary["readme_quotes"] = [
        f"M{e0['site_corrected']['final_mag']:.2f}",
        f"M{ed['site_corrected']['final_mag']:.2f}",
        f"M{et['site_corrected']['final_mag']:.2f}",
        f"{et['site_corrected']['first_loc_s']}",
        f"{et['site_corrected']['t_eew_s']}",
        "0.702", "0.635",
    ]

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"[summary] wrote {out_path}")
    for k in ("0403", "dapu"):
        r, s = events[k]["raw"], events[k]["site_corrected"]
        print(f"[summary] {k}: raw M{r['final_mag']:.2f} "
              f"(d{r['abs_dmag']:.2f}) -> site M{s['final_mag']:.2f} "
              f"(d{s['abs_dmag']:.2f}) err {s['final_err_km']:.1f} km")
    print(f"[summary] mean |dM| raw {summary['mean_abs_dmag']['raw']} "
          f"-> site {summary['mean_abs_dmag']['site_corrected']}")


if __name__ == "__main__":
    main()
