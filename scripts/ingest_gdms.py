"""Phase 3 ingestion: GDMS miniSEED -> AI picks + physical PGA -> replay JSON.

End-to-end chain per station:
    waveform (velocity preferred: HH > EH; else HL acceleration)
      -> windowed fine-tuned PhaseNet -> P/S picks with confidence
    HL acceleration / instrument sensitivity -> PGA in cm/s^2
    dataless inventory -> station coordinates

Output: outputs/replay_<event>.json consumed by the Phase 2 replay engine
(location + magnitude + city countdowns), enabling full-chain historical
replays (2024-04-03 Hualien M7.2, 2025-01-21 Chiayi Dapu ML6.4).

Usage (laptop; raw data folders live next to the repo):
    python scripts/ingest_gdms.py --event 0403 --state-dict outputs/phasenet_cwa_ft.pt
    python scripts/ingest_gdms.py --event dapu --state-dict outputs/phasenet_cwa_ft.pt

NOTE: preset event "truth" values are approximate CWA-report numbers for
comparison overlays only — override with --truth lat,lon,depth,mag if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVENTS = {
    "0403": dict(
        dirname="raw_0403",
        files=["hualien0403_HH.mseed", "hualien0403_EH.mseed",
               "hualien0403_HL.mseed"],
        origin_utc="2024-04-02T23:58:11",
        truth=dict(lat=23.77, lon=121.67, depth_km=15.5, mag=7.2,
                   label="2024-04-03 Hualien ML7.2 (CWA report, approx)"),
    ),
    "dapu": dict(
        dirname="raw_dapu",
        files=["dapu0121_HH.mseed", "dapu0121_HL.mseed"],
        origin_utc="2025-01-20T16:17:27",
        truth=dict(lat=23.24, lon=120.52, depth_km=9.7, mag=6.4,
                   label="2025-01-21 Chiayi Dapu ML6.4 (CWA report, approx)"),
    ),
}

PICK_PRIORITY = ["HH", "EH", "HL"]  # velocity first; HL fallback for picking
FS = 100.0
WIN, HOP = 3001, 1500


def stitch_probs(picker, data):
    """Run the picker over sliding windows; stitch per-sample max prob."""
    n = data.shape[1]
    probs = np.zeros((3, n), dtype=np.float32)
    for start in range(0, max(n - WIN, 0) + 1, HOP):
        seg = data[:, start:start + WIN]
        if seg.shape[1] < WIN:
            seg = np.pad(seg, ((0, 0), (0, WIN - seg.shape[1])))
        out = picker.predict(seg)
        end = min(start + WIN, n)
        probs[:, start:end] = np.maximum(probs[:, start:end],
                                         out.probs[:, :end - start])
    return probs


def peaks(prob, thr, min_dist):
    idx = np.argsort(prob)[::-1]
    out = []
    for i in idx:
        if prob[i] < thr:
            break
        if all(abs(i - j) >= min_dist for j in out):
            out.append(int(i))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", choices=list(EVENTS), default="0403")
    ap.add_argument("--base-dir", default=str(ROOT.parent),
                    help="folder containing raw_0403/raw_dapu/raw_resp")
    ap.add_argument("--dataless", default=None)
    ap.add_argument("--weights", default="original",
                    help='"original" (needs internet once) or "none" (offline)')
    ap.add_argument("--state-dict", default=str(ROOT / "outputs" / "phasenet_cwa_ft.pt"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="pick threshold (default: CANONICAL)")
    ap.add_argument("--truth", default=None, help="lat,lon,depth_km,mag override")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threshold is None:
        from edgequake.location.replay_sim import CANONICAL
        args.threshold = CANONICAL["pick_threshold"]

    import obspy
    from obspy import UTCDateTime, read, read_inventory

    from edgequake.pickers.seisbench_picker import SeisBenchPhaseNet

    cfg = EVENTS[args.event]
    base = Path(args.base_dir)
    dataless = Path(args.dataless) if args.dataless else (
        base / "raw_resp" / "Dataless_CWASN.dataless")
    origin = UTCDateTime(cfg["origin_utc"])
    truth = dict(cfg["truth"])
    if args.truth:
        la, lo, dp, mg = map(float, args.truth.split(","))
        truth.update(lat=la, lon=lo, depth_km=dp, mag=mg, label="user override")

    print(f"[ingest] event {args.event} | origin {origin} | loading waveforms...")
    st = obspy.Stream()
    for f in cfg["files"]:
        st += read(str(base / cfg["dirname"] / f))
    st.merge(method=1, fill_value=0)
    inv = read_inventory(str(dataless))

    picker = SeisBenchPhaseNet(weights=args.weights,
                               state_dict_path=args.state_dict or None)
    print(f"[ingest] picker = {picker.name} | traces {len(st)}")

    # group traces by station
    by_sta: dict[str, dict[str, dict[str, obspy.Trace]]] = {}
    for tr in st:
        fam = tr.stats.channel[:2]
        comp = tr.stats.channel[-1]
        by_sta.setdefault(tr.stats.station, {}).setdefault(fam, {})[comp] = tr

    stations, skipped = [], 0
    t_start = time.perf_counter()
    for code, fams in sorted(by_sta.items()):
        # --- coordinates ---
        coord = None
        for fam, comps in fams.items():
            for tr in comps.values():
                try:
                    coord = inv.get_coordinates(tr.id, tr.stats.starttime)
                    break
                except Exception:
                    continue
            if coord:
                break
        if not coord:
            skipped += 1
            continue

        # --- picking stream: first family with 3 components ---
        pick_fam = next((f for f in PICK_PRIORITY
                         if f in fams and len(fams[f]) == 3), None)
        first_p = first_s = None
        if pick_fam:
            comps = fams[pick_fam]
            # component naming varies: ZNE or Z12 — fall back to sorted keys
            order = [c for c in "ZNE" if c in comps]
            if len(order) != 3:
                order = sorted(comps.keys())
            n_min = min(tr.stats.npts for tr in comps.values())
            data = np.stack([comps[c].data[:n_min] for c in order]).astype(np.float32)
            t0_trace = comps[order[0]].stats.starttime
            probs = stitch_probs(picker, data)
            p_idx = peaks(probs[1], args.threshold, int(2 * FS))
            s_idx = peaks(probs[2], args.threshold, int(2 * FS))
            p_times = [(t0_trace + i / FS, float(probs[1][i])) for i in p_idx]
            # v1 replay guard: drop physically impossible pre-origin picks
            # (real continuous data contains noise triggers — a production
            # system resolves these with phase association, e.g. GaMMA)
            p_times = [(t, pr) for t, pr in p_times if t >= origin - 1]
            if p_times:
                first_p = min(p_times, key=lambda x: x[0])
            s_times = [(t0_trace + i / FS, float(probs[2][i])) for i in s_idx]
            if first_p:
                s_times = [(t, pr) for t, pr in s_times if t > first_p[0]]
                if s_times:
                    first_s = min(s_times, key=lambda x: x[0])

        # --- PGA from HL (acceleration) ---
        pga = pga_t = None
        if "HL" in fams:
            best = 0.0
            for tr in fams["HL"].values():
                try:
                    resp = inv.get_response(tr.id, tr.stats.starttime)
                    sens = resp.instrument_sensitivity.value  # counts per m/s^2
                except Exception:
                    continue
                acc = np.abs(tr.data.astype(np.float64)) / sens * 100.0  # cm/s^2
                i = int(np.argmax(acc))
                if acc[i] > best:
                    best = float(acc[i])
                    pga_t = tr.stats.starttime + i / tr.stats.sampling_rate
            if best > 0:
                pga = best

        stations.append({
            "code": code,
            "lat": coord["latitude"], "lon": coord["longitude"],
            "elev_m": coord.get("elevation"),
            "pick_channel": pick_fam,
            "t_p": str(first_p[0]) if first_p else None,
            "p_prob": round(first_p[1], 3) if first_p else None,
            "t_s": str(first_s[0]) if first_s else None,
            "s_prob": round(first_s[1], 3) if first_s else None,
            "pga_cmps2": round(pga, 3) if pga else None,
            "t_pga": str(pga_t) if pga_t else None,
        })

    n_p = sum(1 for s in stations if s["t_p"])
    n_s = sum(1 for s in stations if s["t_s"])
    n_g = sum(1 for s in stations if s["pga_cmps2"])
    print(f"[ingest] {len(stations)} stations ({skipped} skipped, no coords) | "
          f"P picks {n_p} | S picks {n_s} | PGA {n_g} | "
          f"{time.perf_counter() - t_start:.0f}s")
    if n_p:
        tp = sorted(float(UTCDateTime(s["t_p"]) - origin)
                    for s in stations if s["t_p"])
        print(f"[ingest] first P at origin+{tp[0]:.1f}s, median origin+{np.median(tp):.1f}s")
    if n_g:
        top = sorted((s for s in stations if s["pga_cmps2"]),
                     key=lambda s: -s["pga_cmps2"])[:3]
        for s in top:
            print(f"[ingest] top PGA: {s['code']} {s['pga_cmps2']:.0f} cm/s2")

    out = Path(args.out) if args.out else (ROOT / "outputs" /
                                           f"replay_{args.event}.json")
    out.write_text(json.dumps({
        "event": args.event, "origin_utc": str(origin), "truth": truth,
        "picker": picker.name, "threshold": args.threshold,
        "stations": stations,
    }, indent=1))
    print(f"[ingest] wrote {out}")


if __name__ == "__main__":
    main()
