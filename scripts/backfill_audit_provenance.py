"""One-off (idempotent) provenance backfill for the Taitung audit record.

Round-5 external review found that the ORIGINAL live GitHub Actions log
(outputs/audit_archive/audit_202607310058-EQ2026053-Waveform.txt — first
magnitude M6.2, epicenter error 43 km, pre-CANONICAL wording) disagrees
with the tracked audit record (first M5.88, final M4.81, 17.4 km), with
no documented relationship between the two. Both are real: the log is
what the then-deployed code printed on 2026-07-31; the record was later
recomputed under CANONICAL parameters (single-sourced run params + site
corrections) from the same replay artifact.

Policy: the original log is NEVER modified — it is historical evidence.
This script writes the relationship INTO the audit record instead:

  provenance.original_live_log(+sha256)  the untouched original evidence
  provenance.original_run_commit         from --original-commit (look it
                                         up in the Actions history), else
                                         recorded as unknown
  provenance.canonical_recompute_commit  git HEAD when the tracked
                                         numbers were recomputed
  provenance.recompute_input_sha256      the replay artifact both runs
                                         consumed

The manifest (build_results_summary.py) hashes the original log too, so
--verify now notices if either side of this pair changes silently.

Usage:
    python scripts/backfill_audit_provenance.py            # defaults
    python scripts/backfill_audit_provenance.py --original-commit <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AUDIT = "outputs/audit_archive/audit_eq2026053.json"
ORIGINAL_LOG = ("outputs/audit_archive/"
                "audit_202607310058-EQ2026053-Waveform.txt")
REPLAY_INPUT = "outputs/replay_eq2026053.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--log", default=ORIGINAL_LOG)
    ap.add_argument("--replay-input", default=REPLAY_INPUT)
    ap.add_argument("--original-commit", default=None,
                    help="commit the original Actions run executed on "
                         "(from the Actions run page); omitted -> "
                         "recorded as unknown")
    args = ap.parse_args()

    audit_p = ROOT / args.audit
    log_p = ROOT / args.log
    replay_p = ROOT / args.replay_input
    for p in (audit_p, log_p, replay_p):
        if not p.exists():
            raise SystemExit(f"[provenance] missing {p}")

    rec = json.loads(audit_p.read_text(encoding="utf-8"))
    rec["provenance"] = {
        "original_live_log": args.log,
        "original_live_log_sha256": sha256(log_p),
        "original_run_commit": args.original_commit or "unknown — the "
            "original workflow did not record its commit (fixed since: "
            "records now carry generated_by_commit)",
        "original_run_note": "the original live Actions run (2026-07-31) "
            "used the then-deployed pre-CANONICAL code — before run-"
            "parameter single-sourcing and site corrections — and its "
            "log reports first magnitude M6.2 / 43 km. The log is "
            "retained verbatim as historical evidence and is superseded "
            "by this record.",
        "canonical_recompute_commit": head(),
        "recompute_input_sha256": {args.replay_input: sha256(replay_p)},
        "recompute_note": "this record's numbers are a post-hoc "
            "recomputation of the SAME replay artifact under CANONICAL "
            "parameters (replay_sim.CANONICAL) with site corrections; "
            "see results_summary.json for the parameter set and "
            "source-file hashes.",
    }
    audit_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[provenance] wrote provenance block into {args.audit}")
    print(f"[provenance]   original log sha256 "
          f"{rec['provenance']['original_live_log_sha256'][:16]}..., "
          f"recompute commit "
          f"{(rec['provenance']['canonical_recompute_commit'] or '?')[:10]}")


if __name__ == "__main__":
    main()
