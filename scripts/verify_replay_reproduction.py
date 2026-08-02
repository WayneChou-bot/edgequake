"""Machine-verifiable reproduction of the tracked replay artifacts.

External review (round 3) correctly noted that "we reproduced the picks
bit-exactly" was a prose claim with no committed procedure or record.
This script IS the procedure, and its output IS the record:

  1. hash the raw input waveforms and the checkpoint,
  2. re-run the exact ingestion (scripts/ingest_gdms.py) into a temp file,
  3. compare the ENTIRE regenerated JSON against the tracked
     outputs/replay_<event>.json via canonical serialization (sort_keys),
     plus per-field statistics for diagnostics,
  4. write outputs/reproduction_report.json (command, input/output hashes,
     environment versions, per-field stats, verdict) and exit non-zero
     unless the canonical JSONs are identical.

Verdict vocabulary (round-4 review: say exactly what is guaranteed):
  identical_canonical_json  — full JSON equal after canonical serialization
  selected_fields_match_within_1e-3 — only the field-level comparison holds
  MISMATCH                  — anything else
  SUBSET_RUN                — fewer than all known events were requested;
                              never written to the official report path
                              and always exits non-zero (round-5 review:
                              an empty/partial --events run used to
                              report success)

Scope (round-5 review): this harness covers exactly the two GDMS replay
artifacts (0403, dapu). The Taitung audit replay came from CWA's
published post-event waveform zip and is NOT covered here.

The raw GDMS waveforms are too large for the repo (see .gitignore), so a
third party needs to fetch them (GDMS, doi:10.7914/SN/T5) to re-run this;
the committed report ties checkpoint hash + input hashes + output hashes
together so the chain is auditable even without the re-run.

Usage:
    python scripts/verify_replay_reproduction.py --base-dir ..            # both events
    python scripts/verify_replay_reproduction.py --events dapu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = {
    "0403": ("raw_0403", ["hualien0403_HH.mseed", "hualien0403_EH.mseed",
                          "hualien0403_HL.mseed"]),
    "dapu": ("raw_dapu", ["dapu0121_HH.mseed", "dapu0121_HL.mseed"]),
}
CKPT = ROOT / "outputs" / "v3_verify_x83.pt"
FIELDS = ("t_p", "t_s", "p_prob", "s_prob", "pga_cmps2")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def compare(ref_path: Path, new_path: Path) -> dict:
    from obspy import UTCDateTime

    ref_doc = json.loads(ref_path.read_text(encoding="utf-8"))
    new_doc = json.loads(new_path.read_text(encoding="utf-8"))
    ref = {s["code"]: s for s in ref_doc["stations"]}
    new = {s["code"]: s for s in new_doc["stations"]}
    out = {"stations_ref": len(ref), "stations_new": len(new),
           "station_sets_equal": set(ref) == set(new), "fields": {}}
    ok = out["station_sets_equal"]
    for f in FIELDS:
        n = exact = presence_mismatch = 0
        mx = 0.0
        for c in sorted(set(ref) & set(new)):
            a, b = ref[c].get(f), new[c].get(f)
            if a is None or b is None:
                if a != b:
                    presence_mismatch += 1
                continue
            d = (abs(UTCDateTime(a) - UTCDateTime(b))
                 if f.startswith("t_") else abs(a - b))
            n += 1
            mx = max(mx, d)
            if d < 1e-3:
                exact += 1
        out["fields"][f] = {"n": n, "exact": exact, "max_diff": round(mx, 6),
                            "presence_mismatch": presence_mismatch}
        ok = ok and exact == n and presence_mismatch == 0
    out["canonical_sha256_ref"] = canonical_sha256(ref_doc)
    out["canonical_sha256_new"] = canonical_sha256(new_doc)
    out["new_artifact_sha256"] = sha256(new_path)
    if out["canonical_sha256_ref"] == out["canonical_sha256_new"]:
        out["verdict"] = "identical_canonical_json"
    elif ok:
        out["verdict"] = "selected_fields_match_within_1e-3"
    else:
        out["verdict"] = "MISMATCH"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(ROOT.parent),
                    help="folder containing raw_0403/raw_dapu/raw_resp")
    ap.add_argument("--events", nargs="*", default=sorted(RAW))
    ap.add_argument("--out", default=None,
                    help="output path; REQUIRED for subset runs (the "
                         "official outputs/reproduction_report.json may "
                         "only be written by a full-set run)")
    args = ap.parse_args()
    base = Path(args.base_dir)

    events = sorted(set(args.events))
    unknown = [e for e in events if e not in RAW]
    if unknown:
        raise SystemExit(f"[repro] unknown event(s): {unknown} — known: "
                         f"{sorted(RAW)}")
    if not events:
        raise SystemExit("[repro] refusing to run with an empty event "
                         "list — a no-op must not look like a pass")
    full_set = events == sorted(RAW)
    official = ROOT / "outputs" / "reproduction_report.json"
    if args.out:
        outp = Path(args.out)
        if not full_set and outp.resolve() == official.resolve():
            raise SystemExit("[repro] a subset run may not overwrite the "
                             "official report — choose another --out")
    elif full_set:
        outp = official
    else:
        raise SystemExit("[repro] subset run (--events "
                         + " ".join(events) + ") requires an explicit "
                         "--out; the official report is full-set only")

    import platform

    import numpy
    import obspy as _obspy
    import seisbench as _sb
    import torch as _torch

    dataless = base / "raw_resp" / "Dataless_CWASN.dataless"
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                      time.gmtime()),
        "events_requested": events,
        "full_event_set": sorted(RAW),
        "scope_note": "covers the GDMS replay artifacts only; the "
                      "Taitung audit replay (CWA post-event waveform "
                      "zip) is outside this harness",
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__, "obspy": _obspy.__version__,
            "torch": _torch.__version__, "seisbench": _sb.__version__,
        },
        "checkpoint": {"path": "outputs/v3_verify_x83.pt",
                       "sha256": sha256(CKPT)},
        "dataless_inventory": {
            "path": "raw_resp/Dataless_CWASN.dataless",
            "sha256": sha256(dataless) if dataless.exists() else None,
        },
        "procedure": "scripts/ingest_gdms.py --event <e> --weights none "
                     "--state-dict outputs/v3_verify_x83.pt (defaults, "
                     "threshold from CANONICAL), compared field-by-field "
                     "against tracked outputs/replay_<e>.json "
                     "(tolerance 1e-3 s / 1e-3)",
        "events": {},
    }
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        for e in events:
            dirname, files = RAW[e]
            inputs = {}
            for f in files:
                p = base / dirname / f
                if not p.exists():
                    raise SystemExit(f"[repro] missing raw input {p} — "
                                     "fetch from GDMS first")
                inputs[f"{dirname}/{f}"] = sha256(p)
            out = Path(td) / f"repro_{e}.json"
            cmd = [sys.executable, str(ROOT / "scripts" / "ingest_gdms.py"),
                   "--event", e, "--base-dir", str(base),
                   "--weights", "none", "--state-dict", str(CKPT),
                   "--out", str(out)]
            print(f"[repro] re-running ingestion for {e} ...")
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
            ref = ROOT / "outputs" / f"replay_{e}.json"
            cmp = compare(ref, out)
            cmp["inputs_sha256"] = inputs
            cmp["tracked_artifact"] = f"outputs/replay_{e}.json"
            cmp["tracked_artifact_sha256"] = sha256(ref)
            report["events"][e] = cmp
            all_ok = all_ok and cmp["verdict"] == "identical_canonical_json"
            print(f"[repro] {e}: {cmp['verdict']} "
                  + json.dumps(cmp["fields"]))

    if not full_set:
        report["verdict"] = "SUBSET_RUN"
    else:
        report["verdict"] = ("identical_canonical_json" if all_ok
                             else "MISMATCH")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[repro] wrote {outp} — verdict: {report['verdict']}")
    sys.exit(0 if (all_ok and full_set) else 1)


if __name__ == "__main__":
    main()
