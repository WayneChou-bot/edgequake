"""Animated convergence GIF v2 — time-driven "war-room" replay (Phase 2).

v2 upgrades over the station-stepped v1:
- fixed time steps (default 0.25 s/frame): P/S wavefronts expand smoothly and
  stations light up the moment the P front sweeps past them
- dark theme (palette validated for dark surfaces)
- Taiwan coastline underlay (Natural Earth 1:50m, trimmed to 5 KB in assets/)
- city countdowns: "S wave reaches Taipei in Xs" — the why-EEW-matters line
- trigger flash: a station brightens for ~0.4 s when it triggers
- magnitude pulse on first appearance

Usage:
    python scripts/make_convergence_gif.py --event 20021510590 --max-stations 40
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon as MplPolygon
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# dark theme (validated: #3987e5 / #d95926 on #1a1a19)
SURFACE, LAND, COAST = "#1a1a19", "#242423", "#4a4a48"
INK, INK2, GRID = "#ffffff", "#c3c2b7", "#2e2e2d"
BLUE, ORANGE, FLASH = "#3987e5", "#d95926", "#c98500"
KM_PER_DEG = 111.19

CITIES = [("Taipei", 25.038, 121.514), ("Taichung", 24.147, 120.674),
          ("Kaohsiung", 22.627, 120.301)]
VIEW = dict(lon=(119.7, 122.5), lat=(21.6, 25.6))

# official EEW timeline markers, seconds after ORIGIN (shown once passed)
OFFICIAL = {
    "0403": [(9.0, "CWA rpt#1: M6.2 -> no Taipei alert"),
             (15.0, "CWA rpt#2: M6.8")],
    "dapu": [(7.9, "CWA alert issued (+~5s delivery)")],
}


def km_to_deg(km, lat):
    return km / KM_PER_DEG, km / (KM_PER_DEG * np.cos(np.radians(lat)))


def load_coastline():
    gj = json.loads((ROOT / "assets" / "taiwan_coastline_ne50m.json").read_text())
    polys = []
    for f in gj["features"]:
        geom = f["geometry"]
        rings = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in rings:
            polys.append(np.array(poly[0]))  # outer ring
    return polys


def render_frame(ctx, now, est, mag, k, err_t, err_v, mag_age, eta_lines):
    lats, lons, t_p, truth, vp = (ctx["lats"], ctx["lons"], ctx["t_p"],
                                  ctx["truth"], ctx["vp"])
    vs = vp / 1.73
    fig = plt.figure(figsize=(9.6, 5.4), facecolor=SURFACE, dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[2.3, 1],
                          left=0.06, right=0.97, top=0.88, bottom=0.1,
                          hspace=0.38, wspace=0.2)
    ax = fig.add_subplot(gs[:, 0])
    axt = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])
    for a in (ax, axe):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=7)
    axt.axis("off")

    # --- coastline underlay ---
    for poly in ctx["coast"]:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=LAND,
                                edgecolor=COAST, linewidth=0.8, zorder=1))

    # --- stations ---
    trig = t_p <= now
    fresh = trig & (now - t_p <= 0.4)
    ax.scatter(lons[~trig], lats[~trig], s=9, facecolor="none",
               edgecolor=COAST, linewidth=0.8, zorder=3)
    ax.scatter(lons[trig & ~fresh], lats[trig & ~fresh], s=16, color=INK2,
               linewidth=0, zorder=4)
    if fresh.any():  # trigger flash
        ax.scatter(lons[fresh], lats[fresh], s=110, facecolor="none",
                   edgecolor=FLASH, linewidth=1.6, zorder=5)
        ax.scatter(lons[fresh], lats[fresh], s=24, color=FLASH, linewidth=0,
                   zorder=5)

    # --- cities ---
    for name, cla, clo in CITIES:
        ax.scatter([clo], [cla], marker="s", s=18, color=INK, linewidth=0,
                   zorder=6)
        ax.annotate(name, xy=(clo, cla), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color=INK2)

    if est is not None:
        # error ellipse + estimate
        if est.ellipse_major_km:
            dlat, dlon = km_to_deg(1.0, est.lat)
            ax.add_patch(Ellipse((est.lon, est.lat),
                                 2 * est.ellipse_major_km * dlon,
                                 2 * est.ellipse_minor_km * dlat,
                                 angle=90 - (est.ellipse_azimuth_deg or 0),
                                 facecolor=BLUE, alpha=0.22, edgecolor=BLUE,
                                 linewidth=1.2, zorder=7))
        ax.scatter([est.lon], [est.lat], marker="o", s=55, color=BLUE,
                   edgecolor=SURFACE, linewidth=1.2, zorder=9)
        # P and S wavefronts from the current estimate
        t_after = now - est.origin_time_s
        for r, color, lw in ((vp * t_after, INK2, 0.9),
                             (vs * t_after, ORANGE, 1.6)):
            if 0 < r < 600:
                dlat, dlon = km_to_deg(r, est.lat)
                ax.add_patch(Ellipse((est.lon, est.lat), 2 * dlon, 2 * dlat,
                                     facecolor="none", edgecolor=color,
                                     linewidth=lw, linestyle=(0, (5, 3)),
                                     zorder=8))
    ax.scatter([truth["lon"]], [truth["lat"]], marker="*", s=150,
               color=ORANGE, edgecolor=SURFACE, linewidth=0.7, zorder=8)
    ax.set_xlim(*VIEW["lon"])
    ax.set_ylim(*VIEW["lat"])
    ax.set_aspect(1.0 / np.cos(np.radians(23.7)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)

    # --- readout ---
    y = 0.98

    def line(txt, size, color, dy=None):
        nonlocal y
        axt.text(0.02, y, txt, fontsize=size, color=color,
                 family="DejaVu Sans Mono", va="top", transform=axt.transAxes)
        y -= dy if dy else (0.15 if size > 13 else 0.105)

    line(f"t = +{now:4.1f} s", 19, INK)
    line("since first station trigger", 8, INK2, 0.12)
    if mag:
        msize = 19 if mag_age < 2 else 15
        line(f"M {mag.mag:.1f} ± {mag.sigma:.1f}", msize, BLUE)
    else:
        line("M  —  (awaiting S data)", 11, INK2)
    if est is not None:
        line(f"depth {est.depth_km:3.0f} km   stations {k}", 10, INK)
        line(f"ellipse {est.ellipse_major_km or 0:.0f} × "
             f"{est.ellipse_minor_km or 0:.0f} km", 10, INK)
    else:
        line(f"stations {k}  (locating needs 3)", 10, INK2)
    y -= 0.02
    for name, eta in eta_lines:
        if eta > 0:
            line(f"{name:9s} S wave in {eta:4.0f} s", 10, ORANGE)
        else:
            line(f"{name:9s} S wave arrived", 10, INK2)
    passed = [lbl for t_m, lbl in ctx.get("markers", []) if now >= t_m]
    if passed:
        y -= 0.01
        for lbl in passed[-2:]:
            line(lbl, 8.5, FLASH)

    # --- error vs time ---
    axe.plot(err_t, err_v, color=BLUE, linewidth=1.6)
    axe.set_xlim(0, ctx["t_end"])
    axe.set_ylim(0, (max(err_v) if err_v else 50) * 1.2 + 1)
    axe.set_xlabel("time since first trigger (s)", color=INK2, fontsize=7.5)
    axe.set_ylabel("epicenter error (km)", color=INK2, fontsize=7.5)

    fig.suptitle(
        f"EdgeQuake — real-time convergence replay · M{truth['mag']:.1f} "
        f"{truth['origin_time'][:10]} Taiwan ({ctx['source']})",
        color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.text(0.02, 0.015, "Historical event replayed as a real-time stream — "
             "not a live warning service", color=INK2, fontsize=6.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="20021510590")
    ap.add_argument("--replay-json", default=None,
                    help="ingest_gdms.py output (AI picks) instead of catalog")
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--max-stations", type=int, default=40)
    ap.add_argument("--vp", type=float, default=6.2)
    ap.add_argument("--dt", type=float, default=0.25, help="seconds per frame")
    ap.add_argument("--frame-ms", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)
    else:
        import seisbench

        meta_dir = Path(seisbench.cache_root) / "datasets" / "cwa"

    from demo_convergence import load_event, load_replay_json

    from edgequake.location.locator import PickLocator, haversine_km
    from edgequake.location.magnitude import PgaMagnitude

    if args.replay_json:
        ev, truth = load_replay_json(Path(args.replay_json))
        args.event = Path(args.replay_json).stem.replace("replay_", "")
    else:
        ev, truth = load_event(meta_dir, args.event)
    n_max = min(args.max_stations, len(ev))
    t_ref = ev.t_p.values.min()
    t_p = (ev.t_p.values - t_ref)[:n_max]
    t_s = (ev.t_s.values - t_ref)[:n_max]
    lats = ev.station_latitude_deg.values[:n_max]
    lons = ev.station_longitude_deg.values[:n_max]
    pga = ev.station_pga.values[:n_max]

    locator = PickLocator(vp_km_s=args.vp)
    magest = PgaMagnitude()
    t_end = float(t_p[-1]) + 3.0
    # official markers: convert "after origin" to replay-timeline seconds
    markers = []
    if args.event in OFFICIAL and truth.get("origin_epoch"):
        origin_rel = truth["origin_epoch"] - t_ref  # negative: origin before 1st trigger
        markers = [(t_after + origin_rel, lbl) for t_after, lbl in OFFICIAL[args.event]]
    ctx = {"lats": lats, "lons": lons, "t_p": t_p, "truth": truth,
           "vp": args.vp, "t_end": t_end, "coast": load_coastline(),
           "markers": markers,
           "source": truth.get("source", "CWA catalog picks")}

    frames, err_t, err_v = [], [], []
    est, last_k, mag_first = None, 0, None
    eta_prev = {}
    for now in np.arange(0.0, t_end + 1e-9, args.dt):
        k = int((t_p <= now).sum())
        if k >= 3 and k != last_k:  # relocate on each new trigger
            t_s_avail = np.where(t_s[:k] <= now, t_s[:k], np.nan)
            est = locator.locate(lats[:k], lons[:k], t_p[:k], t_s=t_s_avail,
                                 bootstrap=60)
            last_k = k
        mag = None
        if est is not None:
            m_ok = np.isfinite(t_s[:k]) & (t_s[:k] + 2.0 <= now)
            if m_ok.any():
                d = np.array([haversine_km(est.lat, est.lon, la, lo)
                              for la, lo in zip(lats[:k][m_ok], lons[:k][m_ok])])
                mag = magest.estimate(pga[:k][m_ok], d, est.depth_km)
                if mag and mag_first is None:
                    mag_first = now
            err_t.append(now)
            err_v.append(haversine_km(est.lat, est.lon, truth["lat"],
                                      truth["lon"]))
        # city ETAs with monotonic display clamp (a countdown must never
        # increase; guards against origin-time drift between relocations)
        eta_lines = []
        if est is not None:
            vs = args.vp / 1.73
            for name, cla, clo in CITIES:
                r_hyp = np.hypot(haversine_km(est.lat, est.lon, cla, clo),
                                 est.depth_km)
                eta = est.origin_time_s + r_hyp / vs - now
                prev = eta_prev.get(name)
                if prev is not None:
                    eta = min(eta, prev - args.dt)
                eta_prev[name] = eta
                eta_lines.append((name, eta))
        mag_age = (now - mag_first) / args.dt if mag_first is not None else 99
        frames.append(render_frame(ctx, now, est, mag, k, err_t, err_v,
                                   mag_age, eta_lines))
        if len(frames) % 15 == 0:
            print(f"[gif] frame {len(frames)} (t=+{now:.1f}s)")

    out = Path(args.out) if args.out else (
        ROOT / "outputs" / f"convergence_{args.event}.gif")
    durations = [args.frame_ms] * len(frames)
    durations[0], durations[-1] = 900, 3000
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"[gif] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames)")


if __name__ == "__main__":
    main()
