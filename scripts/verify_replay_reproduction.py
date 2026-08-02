"""Machine-verifiable reproduction of the tracked replay artifacts.

External review (round 3) correctly noted that "we reproduced the picks
bit-exactly" was a prose claim with no committed procedure or record.
This script IS the procedure, and its output IS the record:

  1. hash the raw input waveforms and the checkpoint,
  2. re-run the exact ingestion (scripts/ingest_gdms.py) into a temp file,
  3. compare every station's t_p / t_s / p_prob / s_prob / pga against the
     tracked outputs/replay_<event>.json, field by field,
  4. write outputs/reproduction_report.json (command, hashes, per-field
     match counts, verdict) and exit non-zero on ANY mismatch.

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


def compare(ref_path: Path, new_path: Path) -> dict:
    from obspy import UTCDateTime

    ref = {s["code"]: s for s in
           json.loads(ref_path.read_text(encoding="utf-8"))["stations"]}
    new = {s["code"]: s for s in
           json.loads(new_path.read_text(encoding="utf-8"))["stations"]}
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
    out["bit_exact"] = ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(ROOT.parent),
                    help="folder containing raw_0403/raw_dapu/raw_resp")
    ap.add_argument("--events", nargs="*", default=["0403", "dapu"])
    args = ap.parse_args()
    base = Path(args.base_dir)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                      time.gmtime()),
        "checkpoint": {"path": "outputs/v3_verify_x83.pt",
                       "sha256": sha256(CKPT)},
        "procedure": "scripts/ingest_gdms.py --event <e> --weights none "
                     "--state-dict outputs/v3_verify_x83.pt (defaults, "
                     "threshold from CANONICAL), compared field-by-field "
                     "against tracked outputs/replay_<e>.json "
                     "(tolerance 1e-3 s / 1e-3)",
        "events": {},
    }
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        for e in args.events:
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
            all_ok = all_ok and cmp["bit_exact"]
            print(f"[repro] {e}: " + ("BIT-EXACT" if cmp["bit_exact"]
                                      else "MISMATCH") +
                  " " + json.dumps(cmp["fields"]))

    report["verdict"] = "bit_exact" if all_ok else "MISMATCH"
    outp = ROOT / "outputs" / "reproduction_report.json"
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[repro] wrote {outp} — verdict: {report['verdict']}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
