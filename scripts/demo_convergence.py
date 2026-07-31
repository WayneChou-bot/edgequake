"""Phase 2 demo: multi-station convergence on a real CWA event.

Stations are added one by one in true P-arrival order; after each trigger the
hypocenter is re-estimated with a bootstrap error ellipse. Output: convergence
curve (epicenter error + ellipse size vs. station count) and a map panel.

Uses catalog arrival times from CWA metadata (model-pick integration comes
next — this isolates the location problem first).

Usage:
    python scripts/demo_convergence.py --event 20121013190
    python scripts/demo_convergence.py --event 20021510590 --max-stations 30
    python scripts/demo_convergence.py --metadata-dir path/to/cwa  # default: SeisBench cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_event(metadata_dir: Path, event_id: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    truth = None
    for meta_csv in sorted(metadata_dir.glob("metadata_20*.csv")):
        df = pd.read_csv(meta_csv, low_memory=False)
        sel = df[df.source_event_id.astype(str) == str(event_id)]
        if len(sel):
            rows.append(sel)
    if not rows:
        raise SystemExit(f"event {event_id} not found in {metadata_dir}")
    ev = pd.concat(rows, ignore_index=True)
    ev = ev[np.isfinite(ev.trace_p_arrival_sample)]
    epoch = pd.Timestamp("1970-01-01")
    ev["t_p"] = (pd.to_datetime(ev.trace_p_arrival_time, format="mixed")
                 - epoch).dt.total_seconds()
    # CWA metadata contains occasional corrupt timestamps (e.g. year 2001 in a
    # 2020 event) — keep only arrivals within a sane window around origin time
    t_origin = (pd.to_datetime(ev.source_origin_time.iloc[0]) - epoch).total_seconds()
    sane = (ev.t_p >= t_origin - 10) & (ev.t_p <= t_origin + 300)
    n_bad = int((~sane).sum())
    if n_bad:
        print(f"[conv] dropped {n_bad} rows with corrupt arrival timestamps")
    ev = ev[sane].copy()
    # S arrivals (optional per station), same sanity window
    t_s = (pd.to_datetime(ev.trace_s_arrival_time, format="mixed", errors="coerce")
           - epoch).dt.total_seconds()
    t_s[(t_s < t_origin - 10) | (t_s > t_origin + 600)] = np.nan
    ev["t_s"] = t_s
    # per-station peak PGA across channels (for magnitude inversion)
    pga_max = ev.groupby("station_code").trace_pga_cmps2.transform("max")
    ev["station_pga"] = pga_max
    # one row per station: earliest P (stations can appear with multiple channels)
    ev = ev.sort_values("t_p").drop_duplicates("station_code", keep="first")
    r0 = ev.iloc[0]
    truth = dict(lat=float(r0.source_latitude_deg), lon=float(r0.source_longitude_deg),
                 depth_km=float(r0.source_depth_km), mag=float(r0.source_magnitude),
                 origin_time=str(r0.source_origin_time), gap_deg=float(r0.source_gap_deg))
    return ev, truth




def load_replay_json(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load an ingest_gdms.py replay JSON (AI picks + physical PGA)."""
    import json as _json

    d = _json.loads(Path(path).read_text())
    epoch = pd.Timestamp("1970-01-01")

    def to_s(x):
        if not x:
            return np.nan
        return (pd.to_datetime(x.replace("Z", "")) - epoch).total_seconds()

    rows = []
    for st in d["stations"]:
        if not st["t_p"]:
            continue
        rows.append(dict(
            station_code=st["code"], t_p=to_s(st["t_p"]), t_s=to_s(st["t_s"]),
            station_latitude_deg=st["lat"], station_longitude_deg=st["lon"],
            station_pga=st["pga_cmps2"] if st["pga_cmps2"] else np.nan,
        ))
    ev = pd.DataFrame(rows).sort_values("t_p").reset_index(drop=True)
    tr = d["truth"]
    truth = dict(lat=tr["lat"], lon=tr["lon"], depth_km=tr["depth_km"],
                 mag=tr["mag"], origin_time=d["origin_utc"].replace("Z", ""),
                 gap_deg=float("nan"),
                 origin_epoch=to_s(d["origin_utc"]),
                 source="AI picks, fine-tuned PhaseNet")
    return ev, truth




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="20121013190")
    ap.add_argument("--replay-json", default=None,
                    help="ingest_gdms.py output (AI picks) instead of catalog")
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--max-stations", type=int, default=40)
    ap.add_argument("--vp", type=float, default=6.2)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)
    else:
        import seisbench

        meta_dir = Path(seisbench.cache_root) / "datasets" / "cwa"

    from edgequake.location.locator import PickLocator, haversine_km
    from edgequake.location.magnitude import PgaMagnitude

    if args.replay_json:
        ev, truth = load_replay_json(Path(args.replay_json))
        args.event = Path(args.replay_json).stem.replace("replay_", "")
    else:
        ev, truth = load_event(meta_dir, args.event)
    print(f"[conv] event {args.event}: M{truth['mag']} depth {truth['depth_km']} km, "
          f"{len(ev)} stations with P")

    t_ref = ev.t_p.values.min()
    t_rel = ev.t_p.values - t_ref
    t_s_rel = ev.t_s.values - t_ref
    lats, lons = ev.station_latitude_deg.values, ev.station_longitude_deg.values

    locator = PickLocator(vp_km_s=args.vp)
    magest = PgaMagnitude()
    pga = ev.station_pga.values
    steps = []
    n_max = min(args.max_stations, len(ev))
    for k in range(3, n_max + 1):
        # a station's S leg only becomes available once that S wave arrived:
        # mask S arrivals later than the current wall-clock (= k-th P arrival)
        now = t_rel[k - 1]
        t_s_avail = np.where(t_s_rel[:k] <= now, t_s_rel[:k], np.nan)
        est = locator.locate(lats[:k], lons[:k], t_rel[:k], t_s=t_s_avail,
                             bootstrap=80)
        err = haversine_km(est.lat, est.lon, truth["lat"], truth["lon"])

        # magnitude: only stations whose PGA is plausibly final (S passed +2s),
        # distances from the CURRENT location estimate (location error
        # propagates into magnitude — the honest real-time chain)
        m_ok = np.isfinite(t_s_rel[:k]) & (t_s_rel[:k] + 2.0 <= now)
        mag = None
        if m_ok.any():
            d_est = np.array([haversine_km(est.lat, est.lon, la, lo)
                              for la, lo in zip(lats[:k][m_ok], lons[:k][m_ok])])
            mag = magest.estimate(pga[:k][m_ok], d_est, est.depth_km)

        steps.append({
            "n_stations": k,
            "t_since_first_trigger_s": round(float(t_rel[k - 1]), 2),
            "epicenter_error_km": round(err, 1),
            "depth_est_km": round(est.depth_km, 1),
            "ellipse_major_km": round(est.ellipse_major_km or 0, 1),
            "ellipse_minor_km": round(est.ellipse_minor_km or 0, 1),
            "rms_s": round(est.rms_s, 3),
            "est_lat": round(est.lat, 4), "est_lon": round(est.lon, 4),
            "mag_est": round(mag.mag, 2) if mag else None,
            "mag_sigma": round(mag.sigma, 2) if mag else None,
            "n_mag_stations": mag.n_stations if mag else 0,
        })
        if k in (3, 5, 8, 12, 20, n_max):
            s = steps[-1]
            mtxt = (f"M{s['mag_est']:.1f}±{s['mag_sigma']:.1f}"
                    if s["mag_est"] else "M --")
            print(f"  {k:3d} stations (+{s['t_since_first_trigger_s']:5.1f}s): "
                  f"err {s['epicenter_error_km']:6.1f} km, "
                  f"ellipse {s['ellipse_major_km']:.0f}x{s['ellipse_minor_km']:.0f} km, "
                  f"depth {s['depth_est_km']:.0f} km (true {truth['depth_km']:.0f}), "
                  f"{mtxt} (true M{truth['mag']:.1f})")

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    result = {"event_id": args.event, "truth": truth, "vp_km_s": args.vp,
              "steps": steps}
    out_json = out_dir / f"convergence_{args.event}.json"
    out_json.write_text(json.dumps(result, indent=2))

    from edgequake.location.convergence_plot import plot_convergence

    # final estimate with full station set for the map panel
    est_full = locator.locate(lats[:n_max], lons[:n_max], t_rel[:n_max],
                              t_s=t_s_rel[:n_max], bootstrap=120)
    png = out_dir / f"convergence_{args.event}.png"
    plot_convergence(result, est_full, lats[:n_max], lons[:n_max], str(png))
    print(f"[conv] wrote {out_json.name} / {png.name}")


if __name__ == "__main__":
    main()
