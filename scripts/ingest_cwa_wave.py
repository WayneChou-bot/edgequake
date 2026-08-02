"""Phase 6b: CWA E-A0015-004 strong-motion waveform -> replay JSON + audit.

Every significant felt earthquake, CWA publishes (~12 min after origin) a
zip of per-station ASCII waveforms: 3-component acceleration, 100 Hz, 90 s,
units already gal, station coordinates in the header — self-contained, no
instrument response needed. This script turns one event zip into the same
replay JSON the engine consumes, enabling the automated shadow-mode audit:

    CWA report -> download waveform zip -> AI picks -> engine replay
    -> "had EdgeQuake been running, it would have alerted at +X s"

Usage:
    python scripts/ingest_cwa_wave.py --zip path/to/E-A0015-004.zip
    python scripts/ingest_cwa_wave.py --dat-dir path/to/dats  (.dat or .dat.gz)
    python scripts/ingest_cwa_wave.py --fetch    (uses CWA_API_KEY, laptop)
Add --audit to also run the replay simulation and print the timeline.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FS = 100.0
TZ_OFF = 8 * 3600.0   # headers are GMT+8


def parse_dat(text: str) -> dict | None:
    """Parse one CWA ASCII waveform file -> header dict + (3,N) gal array."""
    hdr = {}
    rows = []
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.match(r"#\s*([^:]+):\s*(.+)", line)
            if m:
                hdr[m.group(1).strip()] = m.group(2).strip()
            continue
        parts = line.split()
        if len(parts) == 4:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    if not rows:
        return None
    arr = np.array(rows)
    # DataSequence: Time U(+); N(+); E(+)  ->  ZNE order
    zne = arr[:, [1, 2, 3]].T.astype(np.float32)
    return dict(hdr=hdr, zne=zne)


def parse_time(s: str):
    """'2026/07/31-00:58:36' or with .000 -> UTCDateTime (UTC)."""
    import obspy

    d, t = s.strip().split("-", 1)
    y, mo, dy = d.split("/")
    return obspy.UTCDateTime(f"{y}-{mo}-{dy}T{t}") - TZ_OFF


def load_event(files: dict[str, bytes]) -> dict:
    """files: {station_filename: raw text bytes}; returns parsed set."""
    stations = []
    ev_hdr = None
    for name, raw in sorted(files.items()):
        rec = parse_dat(raw.decode("utf-8", "replace"))
        if rec is None:
            continue
        h = rec["hdr"]
        if ev_hdr is None:
            ev_hdr = h
        stations.append(dict(
            code=h.get("StationCode", Path(name).stem.split("-")[-1]),
            name=h.get("StationName", ""),
            lat=float(h["StationLatitude(N)"]),
            lon=float(h["StationLongitude(E)"]),
            start=parse_time(h["StartTime(GMT+08)"].split(".")[0]),
            zne=rec["zne"],
            obs_int=Path(name).name.split("-")[0],   # filename prefix
        ))
    if ev_hdr is None:
        raise SystemExit("no parsable .dat files found")
    origin = parse_time(ev_hdr["Origin Time(GMT+08)"])
    truth = dict(lat=float(ev_hdr["EpicenterLatitude(N)"]),
                 lon=float(ev_hdr["EpicenterLongitude(E)"]),
                 depth_km=float(ev_hdr["Depth(km)"]),
                 mag=float(ev_hdr["Magnitude(Ml)"]),
                 label="CWA report solution (E-A0015-004 header)")
    return dict(origin=origin, truth=truth, stations=stations)


def fetch_latest(key: str, out_dir: Path) -> tuple[Path, str]:
    # NOTE: E-A0015-004 is a FILE-type dataset — it lives on the fileapi
    # endpoint (302 -> S3 JSON), NOT /api/v1/rest/datastore (that 404s)
    api = ("https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/E-A0015-004?"
           + urllib.parse.urlencode(dict(Authorization=key,
                                         downloadType="WEB", format="JSON")))
    with urllib.request.urlopen(api, timeout=20) as r:
        rec = json.load(r)
    node = rec["cwaopendata"]
    ident = node["identifier"]
    url = node["Dataset"]["Resource"]["ProductURL"]
    sent = node.get("sent", "")
    out = out_dir / f"cwawave_{ident}.zip"
    if not out.exists():   # the URL is overwritten per event — archive it
        print(f"[cwawave] downloading {ident} ...")
        urllib.request.urlretrieve(url, out)
    return out, sent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--dat-dir", default=None)
    ap.add_argument("--fetch", action="store_true",
                    help="poll E-A0015-004 for the latest event "
                         "(needs CWA_API_KEY)")
    ap.add_argument("--weights", default="none")
    ap.add_argument("--state-dict",
                    default=str(ROOT / "outputs" / "phasenet_cwa_ft.pt"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="pick threshold (default: CANONICAL)")
    ap.add_argument("--sent", default=None,
                    help="report sent time for the audit comparison "
                         "(e.g. 2026-07-31T01:10:18+8:00)")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threshold is None:
        from edgequake.location.replay_sim import CANONICAL
        args.threshold = CANONICAL["pick_threshold"]

    files: dict[str, bytes] = {}
    if args.fetch:
        key = os.environ.get("CWA_API_KEY", "")
        if not key:
            raise SystemExit("set CWA_API_KEY for --fetch")
        arch = ROOT / "outputs" / "audit_archive"
        arch.mkdir(parents=True, exist_ok=True)
        zpath, sent = fetch_latest(key, arch)
        args.zip = str(zpath)
        args.sent = args.sent or sent
    if args.zip:
        z = zipfile.ZipFile(args.zip)
        for n in z.namelist():
            if n.endswith(".dat"):
                files[n] = z.read(n)
    elif args.dat_dir:
        for p in sorted(Path(args.dat_dir).iterdir()):
            if p.name.endswith(".dat.gz"):
                files[p.name[:-3]] = gzip.decompress(p.read_bytes())
            elif p.name.endswith(".dat"):
                files[p.name] = p.read_bytes()
    else:
        raise SystemExit("need --zip, --dat-dir or --fetch")

    ev = load_event(files)
    origin, truth = ev["origin"], ev["truth"]
    ev_id = re.sub(r"\D", "", str(args.zip or args.dat_dir))[-7:] or "cwa"
    print(f"[cwawave] {len(ev['stations'])} stations | origin {origin} UTC | "
          f"CWA M{truth['mag']} @ {truth['lat']}/{truth['lon']} "
          f"depth {truth['depth_km']} km")

    from ingest_gdms import peaks, stitch_probs

    from edgequake.pickers.seisbench_picker import SeisBenchPhaseNet

    picker = SeisBenchPhaseNet(weights=args.weights,
                               state_dict_path=args.state_dict or None)
    out_stations = []
    for s in ev["stations"]:
        data = s["zne"].astype(np.float32)
        probs = stitch_probs(picker, data)
        p_idx = peaks(probs[1], args.threshold, int(2 * FS))
        s_idx = peaks(probs[2], args.threshold, int(2 * FS))
        p_times = [(s["start"] + i / FS, float(probs[1][i])) for i in p_idx]
        p_times = [(t, pr) for t, pr in p_times if t >= origin - 1]
        first_p = min(p_times, key=lambda x: x[0]) if p_times else None
        first_s = None
        if first_p:
            s_times = [(s["start"] + i / FS, float(probs[2][i]))
                       for i in s_idx if s["start"] + i / FS > first_p[0]]
            if s_times:
                first_s = min(s_times, key=lambda x: x[0])
        pga = float(np.abs(data).max())   # already gal
        i_pk = int(np.argmax(np.abs(data).max(axis=0)))
        out_stations.append({
            "code": s["code"], "lat": s["lat"], "lon": s["lon"],
            "elev_m": None, "pick_channel": "FBA",
            "t_p": str(first_p[0]) if first_p else None,
            "p_prob": round(first_p[1], 3) if first_p else None,
            "t_s": str(first_s[0]) if first_s else None,
            "s_prob": round(first_s[1], 3) if first_s else None,
            "pga_cmps2": round(pga, 3),
            "t_pga": str(s["start"] + i_pk / FS),
            "obs_intensity": s["obs_int"],
        })
    n_p = sum(1 for x in out_stations if x["t_p"])
    print(f"[cwawave] P picks {n_p}/{len(out_stations)}")

    out = Path(args.out) if args.out else (ROOT / "outputs" /
                                           f"replay_eq{ev_id}.json")
    out.write_text(json.dumps({
        "event": f"eq{ev_id}", "origin_utc": str(origin), "truth": truth,
        "picker": picker.name, "threshold": args.threshold,
        "report_sent": args.sent,
        "stations": out_stations,
    }, indent=1))
    print(f"[cwawave] wrote {out}")

    if args.audit:
        from demo_convergence import load_replay_json

        from edgequake.location.locator import haversine_km
        from edgequake.location.replay_sim import simulate

        ev_df, tr = load_replay_json(out)
        if len(ev_df) < 3:
            print("[audit] <3 picked stations — no location possible")
            return
        from edgequake.location.site import load_site_terms
        site_terms = load_site_terms(ROOT / "outputs" / "site_terms.json")
        if site_terms:
            print(f"[audit] site terms: {len(site_terms)} stations")
        from edgequake.impact import get_model
        from edgequake.similar import get_similar
        from edgequake.location.replay_sim import CANONICAL
        impact = get_model()
        payload = simulate(ev_df, tr,
                           max_stations=CANONICAL["max_stations"],
                           bootstrap=CANONICAL["bootstrap"],
                           site_terms=site_terms, exposure_model=impact,
                           similar_db=get_similar())
        t_first = min(x["tp"] for x in payload["stations"])
        o_rel = payload["origin_rel"] or 0.0
        first_loc = first_mag = first_alert = first_eew = None
        for f in payload["frames"]:
            if first_loc is None and "lat" in f:
                first_loc = f
            if first_mag is None and f.get("mag") is not None:
                first_mag = f
            if first_eew is None and f.get("eew"):
                first_eew = f
            if first_alert is None and any(
                    c.get("alert") for c in f.get("cty", [])):
                first_alert = f
        def rel(f):
            return f"origin+{f['t'] - o_rel:.1f}s" if f else "never"
        err = (haversine_km(first_mag["lat"], first_mag["lon"],
                            tr["lat"], tr["lon"]) if first_mag else None)
        last_mag = [f for f in payload["frames"]
                    if f.get("mag") is not None]
        last_mag = last_mag[-1] if last_mag else None
        err_final = (haversine_km(last_mag["lat"], last_mag["lon"],
                                  tr["lat"], tr["lon"]) if last_mag else None)
        print("[audit] ---- had EdgeQuake been running ----")
        print(f"[audit] first location : {rel(first_loc)}")
        print(f"[audit] first magnitude: {rel(first_mag)}"
              + (f"  M{first_mag['mag']:.1f} (CWA M{tr['mag']}) "
                 f"err {err:.0f} km" if first_mag else ""))
        if last_mag:
            print(f"[audit] final estimate : M{last_mag['mag']:.2f} "
                  f"err {err_final:.0f} km @ {last_mag['k']} stations")
        print(f"[audit] EEW criteria   : {rel(first_eew)}"
              "  (CWA rule: M>=4.5 & I>=3, ~origin+10-20s official)")
        print(f"[audit] PWS criteria   : {rel(first_alert)}")
        if args.sent:
            print(f"[audit] CWA report sent: {args.sent} "
                  f"(waveform zip ~12 min after origin)")

        # machine-readable audit record (collected into docs/audit.json by
        # scripts/build_audit_index.py — the public audit log)
        import time as _time

        arch = ROOT / "outputs" / "audit_archive"
        arch.mkdir(parents=True, exist_ok=True)
        rec = {
            "id": f"eq{ev_id}",
            "origin_utc": str(origin),
            "cwa": {"mag": tr["mag"], "lat": tr["lat"], "lon": tr["lon"],
                    "depth_km": tr["depth_km"]},
            "n_stations": len(out_stations), "n_picks": n_p,
            "t_first_loc_s": (round(first_loc["t"] - o_rel, 1)
                              if first_loc else None),
            "t_first_mag_s": (round(first_mag["t"] - o_rel, 1)
                              if first_mag else None),
            "first_mag": (round(first_mag["mag"], 2) if first_mag else None),
            "final_mag": (round(last_mag["mag"], 2) if last_mag else None),
            "final_err_km": (round(err_final, 1) if err_final is not None
                             else None),
            "t_eew_s": (round(first_eew["t"] - o_rel, 1)
                        if first_eew else None),
            "exposure": (last_mag or {}).get("exp"),
            "pop_version": payload.get("pop_version"),
            "similar": payload.get("similar"),
            "eew_fired": first_eew is not None,
            "alert_fired": first_alert is not None,
            "report_sent": args.sent,
            "audited_at": _time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                         _time.gmtime()),
        }
        (arch / f"audit_eq{ev_id}.json").write_text(
            json.dumps(rec, indent=1))
        print(f"[audit] wrote audit_eq{ev_id}.json")


if __name__ == "__main__":
    main()
