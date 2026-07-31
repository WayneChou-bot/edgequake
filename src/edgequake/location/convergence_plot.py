"""Convergence figure: map panel (stations/truth/estimate/bootstrap cloud) +
error and uncertainty vs. station count. Palette per project viz rules."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_convergence(result: dict, est_full, st_lats, st_lons, out_png: str):
    steps = result["steps"]
    truth = result["truth"]
    n = [s["n_stations"] for s in steps]
    err = [s["epicenter_error_km"] for s in steps]
    major = [s["ellipse_major_km"] for s in steps]
    t_trig = [s["t_since_first_trigger_s"] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE,
                             width_ratios=[1.2, 1, 1])
    for ax in axes:
        _style(ax)

    # 1 — map panel
    ax = axes[0]
    ax.scatter(st_lons, st_lats, s=14, facecolor="none", edgecolor=INK2,
               linewidth=0.8, label="stations (P)")
    if est_full.bootstrap_lons is not None:
        ax.scatter(est_full.bootstrap_lons, est_full.bootstrap_lats, s=4,
                   color=BLUE, alpha=0.25, linewidth=0, label="bootstrap cloud")
    path_lon = [s["est_lon"] for s in steps]
    path_lat = [s["est_lat"] for s in steps]
    ax.plot(path_lon, path_lat, color=BLUE, linewidth=1.0, alpha=0.6)
    ax.scatter([est_full.lon], [est_full.lat], marker="o", s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, zorder=5, label="final estimate")
    ax.scatter([truth["lon"]], [truth["lat"]], marker="*", s=180, color=ORANGE,
               edgecolor=SURFACE, linewidth=1.0, zorder=6, label="catalog epicenter")
    ax.set_xlabel("longitude", color=INK2, fontsize=9)
    ax.set_ylabel("latitude", color=INK2, fontsize=9)
    ax.set_aspect(1.0 / np.cos(np.radians(truth["lat"])))
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="best")
    ax.set_title(f"M{truth['mag']}  depth {truth['depth_km']:.0f} km",
                 color=INK, fontsize=10, loc="left")

    # 2 — epicenter error vs stations
    ax = axes[1]
    ax.plot(n, err, color=BLUE, linewidth=2.0)
    ax.set_xlabel("stations used", color=INK2, fontsize=9)
    ax.set_ylabel("epicenter error (km)", color=INK2, fontsize=9)
    ax.set_title("Error convergence", color=INK, fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    # annotate elapsed time at a few points
    for k in (0, len(n) // 3, len(n) - 1):
        ax.annotate(f"+{t_trig[k]:.0f}s", xy=(n[k], err[k]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=7.5, color=INK2)

    # 3 — uncertainty (ellipse major axis) vs stations
    ax = axes[2]
    ax.plot(n, major, color=ORANGE, linewidth=2.0)
    ax.set_xlabel("stations used", color=INK2, fontsize=9)
    ax.set_ylabel("1σ ellipse major axis (km)", color=INK2, fontsize=9)
    ax.set_title("Uncertainty convergence", color=INK, fontsize=10, loc="left")
    ax.set_ylim(bottom=0)

    fig.suptitle(
        f"Multi-station convergence — event {result['event_id']} "
        f"(homogeneous vp={result['vp_km_s']} km/s, catalog P+S arrivals)",
        color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
