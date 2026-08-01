"""Blind test of the AI early-magnitude model on 0403 / Dapu (Phase 5).

The model was trained on 2019 and tested on 2020-2021; these events
(2024-04-03 Hualien M7.2, 2025-01-21 Chiayi Dapu ML6.4) are fully out of
sample AND out of time — the honest "train on the past, predict the real
event" protocol.

Per picked station: cut [P-1s, P+3s] from the raw GDMS counts, run MagNet,
then aggregate across stations in true arrival order (inverse-variance
weighted) to build the "AI magnitude vs time" curve, compared against the
physics chain (S-wave PGA magnitude from the convergence run) and the CWA
official reports.

Usage (laptop):
    python scripts/blind_magnet.py --event 0403
    python scripts/blind_magnet.py --event dapu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FS = 100.0
PRE_N, WIN = 100, 400   # P at sample 100 of 400 (1 s + 3 s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", choices=["0403", "dapu"], default="0403")
    ap.add_argument("--base-dir", default=str(ROOT.parent))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--v2", action="store_true",
                    help="distance-conditioned model: aux includes the "
                         "hypocentral distance from the LIVE location "
                         "estimate at that moment (never the truth)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    if args.weights is None:
        args.weights = str(ROOT / "outputs" / "magnet_cwa_v2" /
                           "magnet_cwa_v2.pt" if args.v2
                           else ROOT / "outputs" / "magnet_cwa.pt")

    import obspy
    import torch
    from ingest_gdms import EVENTS, PICK_PRIORITY

    from edgequake.models.magnet import MagNet

    cfg = EVENTS[args.event]
    base = Path(args.base_dir)
    replay = json.loads((ROOT / "outputs" /
                         f"replay_{args.event}.json").read_text())
    truth_mag = replay["truth"]["mag"]
    origin = obspy.UTCDateTime(replay["origin_utc"])

    st = obspy.Stream()
    for f in cfg["files"]:
        p = base / cfg["dirname"] / f
        if p.exists():
            st += obspy.read(str(p))
    st.merge(method=1, fill_value=0)
    by_sta = {}
    for tr in st:
        fam, comp = tr.stats.channel[:2], tr.stats.channel[-1]
        by_sta.setdefault(tr.stats.station, {}).setdefault(fam, {})[comp] = tr

    model = MagNet(n_aux=3 if args.v2 else 2)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # location-estimate timeline for the v2 distance input: what the
    # real-time chain believed at each moment (honest — not the catalog)
    conv = json.loads((ROOT / "outputs" /
                       f"convergence_{args.event}.json").read_text())
    t_first_p = min(obspy.UTCDateTime(s["t_p"]) for s in replay["stations"]
                    if s["t_p"])

    def est_at(t_rel_first):
        """Latest location estimate available at t seconds after the first
        trigger; None before the first 3-station solution."""
        best = None
        for st in conv["steps"]:
            if st["t_since_first_trigger_s"] <= t_rel_first:
                best = st
            else:
                break
        return best

    from edgequake.location.locator import haversine_km

    rows = []
    for s in replay["stations"]:
        if not s["t_p"]:
            continue
        tp = obspy.UTCDateTime(s["t_p"])
        fams = by_sta.get(s["code"], {})
        fam = next((f for f in PICK_PRIORITY
                    if f in fams and len(fams[f]) == 3), None)
        if fam is None:
            continue
        comps = fams[fam]
        order = [c for c in "ZNE" if c in comps]
        if len(order) != 3:
            order = sorted(comps.keys())
        seg = []
        ok = True
        for c in order:
            tr = comps[c].copy()
            if abs(tr.stats.sampling_rate - FS) > 1e-3:
                tr.resample(FS)
            i0 = int(round((tp - tr.stats.starttime) * FS)) - PRE_N
            if i0 < 0 or i0 + WIN > tr.stats.npts:
                ok = False
                break
            seg.append(tr.data[i0:i0 + WIN].astype(np.float64))
        if not ok:
            continue
        seg = np.stack(seg)
        peak, std = float(np.abs(seg).max()), float(seg.std())
        if peak <= 0 or std <= 0 or peak > 1e8:
            continue
        x = torch.tensor((seg / peak)[None].astype(np.float32))
        aux_v = [np.log10(peak), np.log10(std)]
        d_used = None
        if args.v2:
            e = est_at(float(tp - t_first_p) + 3.0)
            if e is not None:
                d_ep = haversine_km(e["est_lat"], e["est_lon"],
                                    s["lat"], s["lon"])
                d_used = float(np.hypot(d_ep, e["depth_est_km"]))
            else:
                d_used = 40.0   # trained fallback prior (no location yet)
            aux_v.append(np.log10(max(d_used, 1.0)))
        aux = torch.tensor([aux_v], dtype=torch.float32)
        with torch.no_grad():
            mu, sig = model.estimate(x, aux)
        rows.append(dict(code=s["code"], ch=fam,
                         t_avail=float(tp - origin) + 3.0,
                         mu=float(mu[0]), sigma=float(sig[0]),
                         d_used=(round(d_used, 1) if d_used else None)))

    rows.sort(key=lambda r: r["t_avail"])
    # cumulative inverse-variance aggregation in arrival order
    # (station predictions share the event, so the aggregated sigma is
    # optimistic — treat it as a lower bound)
    agg = []
    sw = swm = 0.0
    for r in rows:
        w = 1.0 / r["sigma"] ** 2
        sw += w
        swm += w * r["mu"]
        agg.append(dict(t=r["t_avail"], n=len(agg) + 1,
                        mag=swm / sw, sigma=(1.0 / sw) ** 0.5))

    # physics-chain timeline from the convergence run (t is per-k arrival
    # time since FIRST TRIGGER -> convert to since-origin)
    conv = json.loads((ROOT / "outputs" /
                       f"convergence_{args.event}.json").read_text())
    t_first_p = min(obspy.UTCDateTime(s["t_p"]) for s in replay["stations"]
                    if s["t_p"])
    off = float(t_first_p - origin)
    phys = [(s["t_since_first_trigger_s"] + off, s["mag_est"])
            for s in conv["steps"] if s["mag_est"]]

    from build_dashboard import OFFICIAL

    tag = "_v2" if args.v2 else ""
    summary = dict(
        event=args.event, model="v2-dist" if args.v2 else "v1",
        truth_mag=truth_mag, n_stations=len(rows),
        first_ai=dict(t=round(agg[0]["t"], 1), mag=round(agg[0]["mag"], 2),
                      sigma=round(agg[0]["sigma"], 2)) if agg else None,
        ai_at=[dict(t=round(a["t"], 1), n=a["n"], mag=round(a["mag"], 2),
                    sigma=round(a["sigma"], 2))
               for a in agg if a["n"] in (1, 3, 5, 10, 20, len(rows))],
        first_physics=dict(t=round(phys[0][0], 1),
                           mag=phys[0][1]) if phys else None,
        note="AI window: P-1s..P+3s raw counts; aggregated sigma is a "
             "lower bound (correlated stations); trained <=2019, this "
             "event is out-of-time",
    )
    out_json = ROOT / args.out / f"blind_magnet{tag}_{args.event}.json"
    out_json.write_text(json.dumps(
        {"summary": summary, "stations": rows, "aggregate": agg}, indent=1))
    print(json.dumps(summary, indent=1))

    # ---- plot ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a19")
    ax.set_facecolor("#20201f")
    for sp in ax.spines.values():
        sp.set_color("#4a4a48")
    ax.tick_params(colors="#c3c2b7")
    ta = [a["t"] for a in agg]
    ma = [a["mag"] for a in agg]
    sa = [a["sigma"] for a in agg]
    # v2 runs overlay the v1 baseline for the before/after story
    if args.v2:
        v1p = ROOT / args.out / f"blind_magnet_{args.event}.json"
        if v1p.exists():
            v1 = json.loads(v1p.read_text())["aggregate"]
            ax.step([a["t"] for a in v1], [a["mag"] for a in v1],
                    where="post", color="#8b97a3", lw=1.4, ls="--",
                    label="AI v1 (no distance input)")
    ax.fill_between(ta, [m - s for m, s in zip(ma, sa)],
                    [m + s for m, s in zip(ma, sa)],
                    color="#3987e5", alpha=0.18, step="post")
    ax.step(ta, ma, where="post", color="#3987e5", lw=2,
            label="AI early magnitude" +
            (" v2 (distance-conditioned)" if args.v2 else
             " (P+3s windows, aggregated)"))
    if phys:
        ax.step([t for t, _ in phys], [m for _, m in phys], where="post",
                color="#d95926", lw=2,
                label="physics chain (S-wave PGA + location)")
    ax.axhline(truth_mag, color="#ffffff", lw=1, ls="--",
               label=f"CWA final M{truth_mag}")
    for t_off, lbl in OFFICIAL.get(args.event, []):
        ax.axvline(t_off, color="#c98500", lw=1, ls=":")
        ax.text(t_off + 0.3, ax.get_ylim()[0] + 0.15, lbl.split(":")[0],
                color="#c98500", fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("seconds since origin", color="#c3c2b7")
    ax.set_ylabel("magnitude", color="#c3c2b7")
    ax.set_title(f"Blind test {args.event}: AI early magnitude vs physics "
                 f"chain (model trained on ≤2019)", color="white")
    leg = ax.legend(facecolor="#20201f", edgecolor="#4a4a48", fontsize=8)
    for txt in leg.get_texts():
        txt.set_color("#c3c2b7")
    out_png = ROOT / args.out / f"blind_magnet{tag}_{args.event}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"[blind] wrote {out_json.name} / {out_png.name}")


if __name__ == "__main__":
    main()
