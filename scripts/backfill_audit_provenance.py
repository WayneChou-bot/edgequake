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
This script writes the relationship INTO the audit record instead.

Commit semantics (round-6 review: the first version wrote HEAD-at-
backfill-time as "canonical_recompute_commit", which was wrong — this
script never recomputes anything). The layers now are:

  original_results_commit          the commit that ADDED the original
                                   live log (auto: first commit touching
                                   it — the Actions run committed its
                                   own outputs)
  original_run_commit              its PARENT — the code the Actions run
                                   actually executed on
  record_numbers_written_in_commit the commit that wrote the record's
                                   current numbers (auto: git pickaxe on
                                   the final_mag value)
  provenance_backfilled_by_commit  HEAD when THIS script ran — bookkeeping
                                   only, proves nothing about the numbers

Whether the numbers are CORRECT is not this script's claim at all: every
results_summary generation re-derives them from the replay artifact and
fails on mismatch (audit_cross_check), and the manifest hashes freeze
both files afterwards.

Usage:
    python scripts/backfill_audit_provenance.py              # all auto
    python scripts/backfill_audit_provenance.py --numbers-commit <sha>
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


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def head() -> str | None:
    return _git("rev-parse", "HEAD")


def first_commit_touching(rel: str) -> str | None:
    """First commit in history that touched `rel` (oldest first)."""
    out = _git("log", "--reverse", "--format=%H", "--", rel)
    return out.splitlines()[0] if out else None


def pickaxe_first(rel: str, needle: str) -> str | None:
    """Oldest commit whose diff of `rel` changed occurrences of needle —
    i.e. the commit that WROTE the current value."""
    out = _git("log", "--reverse", "--format=%H", "-S", needle, "--", rel)
    return out.splitlines()[0] if out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--log", default=ORIGINAL_LOG)
    ap.add_argument("--replay-input", default=REPLAY_INPUT)
    ap.add_argument("--numbers-commit", default=None,
                    help="override for record_numbers_written_in_commit "
                         "(normally auto-detected via git pickaxe)")
    args = ap.parse_args()

    audit_p = ROOT / args.audit
    log_p = ROOT / args.log
    replay_p = ROOT / args.replay_input
    for p in (audit_p, log_p, replay_p):
        if not p.exists():
            raise SystemExit(f"[provenance] missing {p}")

    rec = json.loads(audit_p.read_text(encoding="utf-8"))
    orig_results = first_commit_touching(args.log)
    orig_run = (_git("rev-parse", f"{orig_results}^")
                if orig_results else None)
    numbers = args.numbers_commit or (
        pickaxe_first(args.audit, f'"final_mag": {rec.get("final_mag")}')
        if rec.get("final_mag") is not None else None)
    rec["provenance"] = {
        "original_live_log": args.log,
        "original_live_log_sha256": sha256(log_p),
        "original_results_commit": orig_results or "unknown (git "
            "history unavailable)",
        "original_run_commit": orig_run or "unknown (git history "
            "unavailable)",
        "original_run_note": "the original live Actions run (2026-07-31) "
            "used the then-deployed pre-CANONICAL code — before run-"
            "parameter single-sourcing and site corrections — and its "
            "log reports first magnitude M6.2 / 43 km. The log is "
            "retained verbatim as historical evidence and is superseded "
            "by this record.",
        "record_numbers_written_in_commit": numbers or "unknown "
            "(pickaxe found no commit; pass --numbers-commit)",
        "provenance_backfilled_by_commit": head(),
        "recompute_input_sha256": {args.replay_input: sha256(replay_p)},
        "recompute_verification": "the numeric fields of this record are "
            "re-derived from the replay artifact at every "
            "results_summary generation and cross-checked field by "
            "field (audit_cross_check in results_summary.json); "
            "generation aborts on any mismatch, and --verify fails "
            "unless the cross-check status is 'match' and the record "
            "hash is unchanged since.",
    }
    audit_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    p = rec["provenance"]
    print(f"[provenance] wrote provenance block into {args.audit}")
    print(f"[provenance]   original log sha256 "
          f"{p['original_live_log_sha256'][:16]}...")
    print(f"[provenance]   original results commit "
          f"{str(p['original_results_commit'])[:10]} (run commit "
          f"{str(p['original_run_commit'])[:10]})")
    print(f"[provenance]   numbers written in "
          f"{str(p['record_numbers_written_in_commit'])[:10]}, "
          f"backfilled by {str(p['provenance_backfilled_by_commit'])[:10]}")


if __name__ == "__main__":
    main()
