"""Build a multi-station event catalog from local CWA metadata (Phase 2).

Groups traces by source_event_id, counts stations with P picks, and writes an
events index (CSV + JSON summary) used to select replay/convergence events.

Usage:
    python scripts/build_event_catalog.py                 # test years 2020-2021
    python scripts/build_event_catalog.py --years 2019 2020 2021
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=[2020, 2021])
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--min-stations", type=int, default=5)
    ap.add_argument("--out", default="outputs/event_catalog.csv")
    args = ap.parse_args()

    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)
    else:
        import seisbench

        meta_dir = Path(seisbench.cache_root) / "datasets" / "cwa"

    dfs = []
    for y in args.years:
        p = meta_dir / f"metadata_{y}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    ev = df[(df.trace_category == "earthquake")
            & np.isfinite(df.trace_p_arrival_sample)]

    g = ev.groupby("source_event_id").agg(
        n_traces=("trace_name", "count"),
        n_stations=("station_code", "nunique"),
        n_s_picks=("trace_s_arrival_sample", lambda s: int(np.isfinite(s).sum())),
        magnitude=("source_magnitude", "first"),
        depth_km=("source_depth_km", "first"),
        lat=("source_latitude_deg", "first"),
        lon=("source_longitude_deg", "first"),
        gap_deg=("source_gap_deg", "first"),
        origin_time=("source_origin_time", "first"),
    ).reset_index()
    g = g[g.n_stations >= args.min_stations].sort_values(
        ["magnitude", "n_stations"], ascending=False)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    g.to_csv(out, index=False)
    summary = {
        "years": args.years,
        "events": int(len(g)),
        "by_station_count": {f">={k}": int((g.n_stations >= k).sum())
                             for k in (5, 8, 15, 30, 100)},
        "by_magnitude": {f">={m}": int((g.magnitude >= m).sum())
                         for m in (4, 5, 5.5, 6)},
        "top10": g.head(10)[["source_event_id", "magnitude", "depth_km",
                             "n_stations", "gap_deg", "origin_time"]]
                  .to_dict("records"),
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"[catalog] wrote {out} ({len(g)} events)")


if __name__ == "__main__":
    main()
