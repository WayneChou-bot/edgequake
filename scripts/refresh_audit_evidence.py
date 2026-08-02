"""Recompute machine-derived alert-gate evidence for an audit record.

Round-7 review: the public report narrated a PWS non-trigger REASON the
data did not support (it claimed "magnitude stayed below M5.0" while
quoting a first magnitude of 5.88 in the same breath). The reason must
be computed, not narrated — replay_sim.pws_evidence() derives it from
the replay frames, this script writes it into the audit record, and
llm_report.py turns it into mandatory verbatim sentences the report
cannot deviate from.

Safety: before touching the record, the seven cross-check fields are
recomputed from the replay artifact at CANONICAL parameters and must
match the record exactly — this script refuses to decorate a record
whose numbers it cannot reproduce.

Usage:
    python scripts/refresh_audit_evidence.py                # eq2026053
    python scripts/refresh_audit_evidence.py --id eq2026054
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="eq2026053")
    args = ap.parse_args()

    from demo_convergence import load_replay_json

    from edgequake.location.replay_sim import pws_evidence, simulate
    from edgequake.location.site import load_site_terms

    audit_p = ROOT / "outputs" / "audit_archive" / f"audit_{args.id}.json"
    replay_p = ROOT / "outputs" / f"replay_{args.id}.json"
    for p in (audit_p, replay_p):
        if not p.exists():
            raise SystemExit(f"[evidence] missing {p}")
    rec = json.loads(audit_p.read_text(encoding="utf-8"))

    st = load_site_terms(ROOT / "outputs" / "site_terms.json")
    ev, tr = load_replay_json(replay_p)
    payload = simulate(ev, tr, site_terms=st)   # params: CANONICAL

    # consistency guard — same seven fields as the summary cross-check
    orel = payload["origin_rel"] or 0.0
    frames = [f for f in payload["frames"] if f.get("mag") is not None]
    floc = next(f for f in payload["frames"] if "lat" in f)
    f_eew = next((f for f in payload["frames"] if f.get("eew")), None)
    f_alert = next((f for f in payload["frames"]
                    if any(c.get("alert") for c in f.get("cty", []))),
                   None)
    recomputed = {
        "t_first_loc_s": round(floc["t"] - orel, 1),
        "first_mag": frames[0]["mag"],
        "final_mag": frames[-1]["mag"],
        "final_err_km": frames[-1]["err"],
        "t_eew_s": (round(f_eew["t"] - orel, 1) if f_eew else None),
        "eew_fired": f_eew is not None,
        "alert_fired": f_alert is not None,
    }
    mismatch = {k: (rec.get(k), v) for k, v in recomputed.items()
                if rec.get(k) != v}
    if mismatch:
        print("[evidence] FATAL: record disagrees with CANONICAL "
              "recomputation — refusing to decorate it:")
        for k, (a, b) in mismatch.items():
            print(f"  - {k}: record={a!r} recomputed={b!r}")
        sys.exit(1)

    rec["pws_evidence"] = pws_evidence(payload)
    audit_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    evd = rec["pws_evidence"]
    print(f"[evidence] wrote pws_evidence into {audit_p.name}")
    print(f"[evidence]   fired={evd['fired']}, max_mag="
          f"{evd.get('max_mag')}, max county I="
          f"{evd.get('max_predicted_county_intensity')}, max obs="
          f"{evd.get('max_observed_pga_gal')} gal")
    if not evd["fired"]:
        print(f"[evidence]   blockers: {evd.get('blockers')}")


if __name__ == "__main__":
    main()
