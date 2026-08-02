"""Build docs/audit.json from outputs/audit_archive/audit_*.json.

Each shadow-audit run (scripts/ingest_cwa_wave.py --audit) drops one
machine-readable record per event into outputs/audit_archive/. This script
aggregates them into a single docs/audit.json the console's 稽核紀錄 tab
fetches, plus a copy in vercel/ so the Vercel deploy serves it too.

Run locally or as a workflow step right after the audit:
    python scripts/build_audit_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    arch = ROOT / "outputs" / "audit_archive"
    records = []
    for f in sorted(arch.glob("audit_*.json")):
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[audit-index] skip {f.name}: {e}", file=sys.stderr)
    # newest first (origin time desc; fall back to id)
    records.sort(key=lambda r: (r.get("origin_utc") or "", r.get("id") or ""),
                 reverse=True)

    dm = [abs(r["final_mag"] - r["cwa"]["mag"]) for r in records
          if r.get("final_mag") is not None
          and r.get("cwa", {}).get("mag") is not None]
    ek = [r["final_err_km"] for r in records
          if r.get("final_err_km") is not None]
    tl = [r["t_first_loc_s"] for r in records
          if r.get("t_first_loc_s") is not None]
    summary = {
        "n_events": len(records),
        "mean_abs_dmag": round(sum(dm) / len(dm), 2) if dm else None,
        "mean_err_km": round(sum(ek) / len(ek), 1) if ek else None,
        "mean_t_first_loc_s": round(sum(tl) / len(tl), 1) if tl else None,
        "n_alert_fired": sum(1 for r in records if r.get("alert_fired")),
        "n_report_sent": sum(1 for r in records if r.get("report_sent")),
    }
    out = {"summary": summary, "events": records}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    for dest in (ROOT / "docs" / "audit.json", ROOT / "vercel" / "audit.json"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"[audit-index] wrote {dest.relative_to(ROOT)}")
    print(f"[audit-index] {len(records)} events | "
          f"mean |dM| {summary['mean_abs_dmag']} | "
          f"mean err {summary['mean_err_km']} km")


if __name__ == "__main__":
    main()
