"""Phase 7: empirical site-effect terms from the CWA catalog metadata.

Each station sits on different ground: soft-basin sites (e.g. Taipei basin,
Lanyang plain) systematically amplify PGA, rock sites damp it. Our GMPE
    log10(PGA) = a*M + b*log10(R_hyp) + c
models the *average* site, so a station's typical residual against it IS its
site term:
    dS(station) = median over events of [log10(PGA_obs) - log10(PGA_pred)]
Correcting observed PGA by 10**dS before inverting for magnitude removes a
station-dependent bias from every M estimate (and tightens sigma).

Needs ONLY the metadata CSVs (trace_pga_cmps2 column) — no waveforms.
Run on the machine that has the SeisBench CWA cache:

    python scripts/build_site_terms.py                     # 2019-2021
    python scripts/build_site_terms.py --metadata-dir D:\\path\\to\\cwa

Output: outputs/site_terms.json  (commit it — engine + audits load it).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from edgequake.location.locator import haversine_km          # noqa: E402
from edgequake.location.magnitude import DEFAULT_COEF        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int,
                    default=[2019, 2020, 2021])
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--min-events", type=int, default=10,
                    help="station needs >= this many events to get a term")
    ap.add_argument("--out", default=str(ROOT / "outputs" /
                                         "site_terms.json"))
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
            print(f"[site] reading {p.name}...")
            dfs.append(pd.read_csv(p, low_memory=False))
        else:
            print(f"[site] WARNING: {p} missing, skipping year {y}")
    if not dfs:
        raise SystemExit("[site] no metadata found — check --metadata-dir")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[site] {len(df)} traces total")

    need = ["source_magnitude", "source_latitude_deg", "source_longitude_deg",
            "source_depth_km", "station_latitude_deg",
            "station_longitude_deg", "trace_pga_cmps2"]
    df = df[(df.trace_category == "earthquake")]
    for c in need:
        df = df[np.isfinite(pd.to_numeric(df[c], errors="coerce"))]
    df = df[df.trace_pga_cmps2 > 0.2]          # below ~0.2 gal: noise floor

    # one record per (event, station): the max PGA across its traces
    g = (df.groupby(["source_event_id", "station_code"])
           .agg(pga=("trace_pga_cmps2", "max"),
                mag=("source_magnitude", "first"),
                elat=("source_latitude_deg", "first"),
                elon=("source_longitude_deg", "first"),
                dep=("source_depth_km", "first"),
                slat=("station_latitude_deg", "first"),
                slon=("station_longitude_deg", "first"))
           .reset_index())
    print(f"[site] {len(g)} station-event records")

    d_ep = np.array([haversine_km(a, b, c, d) for a, b, c, d in
                     zip(g.elat, g.elon, g.slat, g.slon)])
    r = np.sqrt(d_ep ** 2 + g.dep.values ** 2)
    # protocol guards: skip near-field saturation (M>6.8) and far/weak tails
    ok = (r >= 5) & (r <= 300) & (g.mag.values >= 3.0) & (g.mag.values <= 6.8)
    g, r = g[ok].copy(), r[ok]
    a_, b_, c_ = DEFAULT_COEF["a"], DEFAULT_COEF["b"], DEFAULT_COEF["c"]
    res = (np.log10(g.pga.values)
           - (a_ * g.mag.values + b_ * np.log10(r) + c_))
    keep = np.abs(res) <= 1.5                  # outliers: bad meta / clipped
    g, res = g[keep].copy(), res[keep]
    g["res"] = res
    print(f"[site] {len(g)} records after quality cuts")

    st = (g.groupby("station_code")["res"]
            .agg(n="count", dS="median",
                 sig=lambda x: 1.4826 * np.median(np.abs(x - x.median())))
            .reset_index())
    st = st[st.n >= args.min_events].copy()
    st["dS"] = st.dS.clip(-0.6, 0.6).round(3)
    st["sig"] = st.sig.round(3)

    terms = {row.station_code: {"dS": float(row.dS), "n": int(row.n),
                                "sig": float(row.sig)}
             for row in st.itertuples()}
    out = {
        "_meta": {
            "definition": "dS = median log10(PGA_obs/PGA_pred) per station; "
                          "correct PGA by 10**dS before magnitude inversion",
            "coef": DEFAULT_COEF, "years": args.years,
            "min_events": args.min_events, "n_stations": len(terms),
            "n_records": int(len(g)),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                          time.gmtime()),
        },
        "terms": terms,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[site] wrote {outp} ({len(terms)} stations)")

    st = st.sort_values("dS")
    show = ["station_code", "dS", "n", "sig"]
    print("[site] strongest DE-amplifiers (rock):")
    print(st.head(8)[show].to_string(index=False))
    print("[site] strongest AMPLIFIERS (soft soil / basin):")
    print(st.tail(8)[show].to_string(index=False))
    print(f"[site] |dS|>0.2 at {int((st.dS.abs() > 0.2).sum())} stations "
          f"(a 0.2 dex site error alone biases single-station M by "
          f"{0.2 / a_:.2f})")


if __name__ == "__main__":
    main()
