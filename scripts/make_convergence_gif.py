"""Animated convergence GIF for the README (Phase 2).

Each frame = one more station triggered: stations light up, the estimate and
its 1-sigma ellipse update, the S wavefront expands from the current estimate,
and a live readout shows elapsed time / magnitude / depth. Frames are composed
with matplotlib and assembled with Pillow.

Usage:
    python scripts/make_convergence_gif.py --event 20021510590 --max-stations 40
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
KM_PER_DEG = 111.19


def km_to_deg(km, lat):
    return km / KM_PER_DEG, km / (KM_PER_DEG * np.cos(np.radians(lat)))


def render_frame(ev_ctx, k, est, mag, now, err_hist, n_hist):
    lats, lons, truth, vp = (ev_ctx["lats"], ev_ctx["lons"],
                             ev_ctx["truth"], ev_ctx["vp"])
    fig = plt.figure(figsize=(9.6, 5.4), facecolor=SURFACE, dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[2.2, 1],
                          left=0.07, right=0.97, top=0.88, bottom=0.1,
                          hspace=0.35, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    axt = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])

    for a in (ax, axe):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=7.5)
    axt.axis("off")

    # --- map ---
    ax.scatter(lons, lats, s=13, facecolor="none", edgecolor=GRID,
               linewidth=0.9, zorder=2)
    ax.scatter(lons[:k], lats[:k], s=22, color=INK2, linewidth=0, zorder=3,
               label=f"triggered ({k})")
    if est.ellipse_major_km:
        dlat, dlon = km_to_deg(1.0, est.lat)
        e = Ellipse((est.lon, est.lat),
                    width=2 * est.ellipse_major_km * dlon,
                    height=2 * est.ellipse_minor_km * dlat,
                    angle=90 - (est.ellipse_azimuth_deg or 0),
                    facecolor=BLUE, alpha=0.18, edgecolor=BLUE, linewidth=1.2,
                    zorder=4)
        ax.add_patch(e)
    ax.scatter([est.lon], [est.lat], marker="o", s=60, color=BLUE,
               edgecolor=SURFACE, linewidth=1.2, zorder=6, label="estimate")
    ax.scatter([truth["lon"]], [truth["lat"]], marker="*", s=170,
               color=ORANGE, edgecolor=SURFACE, linewidth=0.8, zorder=5,
               label="catalog")
    # S wavefront from current estimate
    t_after_origin = now - est.origin_time_s
    r_s = (vp / 1.73) * max(t_after_origin, 0)
    if 0 < r_s < 500:
        dlat, dlon = km_to_deg(r_s, est.lat)
        ax.add_patch(Ellipse((est.lon, est.lat), 2 * dlon, 2 * dlat,
                             facecolor="none", edgecolor=ORANGE,
                             linewidth=1.4, linestyle=(0, (5, 3)), zorder=4))
    pad = 0.9
    ax.set_xlim(truth["lon"] - pad, truth["lon"] + pad)
    ax.set_ylim(truth["lat"] - pad, truth["lat"] + pad)
    ax.set_aspect(1.0 / np.cos(np.radians(truth["lat"])))
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left")
    ax.set_title("station triggers · estimate · 1σ ellipse · S wavefront",
                 color=INK2, fontsize=8.5, loc="left")

    # --- readout ---
    mag_txt = (f"M {mag.mag:.1f} ± {mag.sigma:.1f}" if mag else "M  —  (awaiting S)")
    lines = [
        (f"t = +{now:.1f} s", 20, INK),
        ("since first station trigger", 8.5, INK2),
        ("", 6, INK2),
        (mag_txt, 16, BLUE if mag else INK2),
        (f"depth  {est.depth_km:.0f} km", 11, INK),
        (f"stations  {k}", 11, INK),
        (f"ellipse  {est.ellipse_major_km or 0:.0f} × "
         f"{est.ellipse_minor_km or 0:.0f} km", 11, INK),
    ]
    y = 0.97
    for txt, size, color in lines:
        axt.text(0.02, y, txt, fontsize=size, color=color,
                 family="DejaVu Sans Mono", va="top", transform=axt.transAxes)
        y -= 0.135 if size > 12 else 0.105

    # --- growing error curve ---
    axe.plot(n_hist, err_hist, color=BLUE, linewidth=1.8)
    axe.set_xlim(3, ev_ctx["n_max"])
    axe.set_ylim(0, max(err_hist) * 1.15 + 1)
    axe.set_xlabel("stations", color=INK2, fontsize=8)
    axe.set_ylabel("epicenter error (km)", color=INK2, fontsize=8)

    fig.suptitle(
        f"EdgeQuake — multi-station convergence replay · "
        f"M{truth['mag']:.1f} {truth['origin_time'][:10]} (CWA catalog picks)",
        color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.text(0.02, 0.015, "Historical event replayed as a real-time stream — "
             "not a live warning service", color=INK2, fontsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="20021510590")
    ap.add_argument("--metadata-dir", default=None)
    ap.add_argument("--max-stations", type=int, default=40)
    ap.add_argument("--vp", type=float, default=6.2)
    ap.add_argument("--frame-ms", type=int, default=350)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)
    else:
        import seisbench

        meta_dir = Path(seisbench.cache_root) / "datasets" / "cwa"

    from demo_convergence import load_event

    from edgequake.location.locator import PickLocator, haversine_km
    from edgequake.location.magnitude import PgaMagnitude

    ev, truth = load_event(meta_dir, args.event)
    t_ref = ev.t_p.values.min()
    t_rel = ev.t_p.values - t_ref
    t_s_rel = ev.t_s.values - t_ref
    lats, lons = ev.station_latitude_deg.values, ev.station_longitude_deg.values
    pga = ev.station_pga.values

    locator = PickLocator(vp_km_s=args.vp)
    magest = PgaMagnitude()
    n_max = min(args.max_stations, len(ev))
    ctx = {"lats": lats[:n_max], "lons": lons[:n_max], "truth": truth,
           "vp": args.vp, "n_max": n_max}

    frames, err_hist, n_hist = [], [], []
    for k in range(3, n_max + 1):
        now = t_rel[k - 1]
        t_s_avail = np.where(t_s_rel[:k] <= now, t_s_rel[:k], np.nan)
        est = locator.locate(lats[:k], lons[:k], t_rel[:k], t_s=t_s_avail,
                             bootstrap=60)
        m_ok = np.isfinite(t_s_rel[:k]) & (t_s_rel[:k] + 2.0 <= now)
        mag = None
        if m_ok.any():
            d_est = np.array([haversine_km(est.lat, est.lon, la, lo)
                              for la, lo in zip(lats[:k][m_ok], lons[:k][m_ok])])
            mag = magest.estimate(pga[:k][m_ok], d_est, est.depth_km)
        err_hist.append(haversine_km(est.lat, est.lon, truth["lat"], truth["lon"]))
        n_hist.append(k)
        frames.append(render_frame(ctx, k, est, mag, now, err_hist, n_hist))
        if k % 10 == 0:
            print(f"[gif] frame {k}/{n_max}")

    out = Path(args.out) if args.out else (
        ROOT / "outputs" / f"convergence_{args.event}.gif")
    durations = [args.frame_ms] * len(frames)
    durations[0], durations[-1] = 1200, 2500  # hold first & last frames
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"[gif] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames)")


if __name__ == "__main__":
    main()
