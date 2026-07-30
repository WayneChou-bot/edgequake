"""Result figure for a replay run: waveform + rolling P/S probabilities + latency.

Palette (validated for CVD separation & contrast): P = #2a78d6 (blue),
S = #eb6834 (orange), waveform/ink = neutral grays. One y-axis per panel.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .replay.engine import ReplayResult

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
P_COLOR = "#2a78d6"
S_COLOR = "#eb6834"
SURFACE = "#fcfcfb"


def plot_replay(result: ReplayResult, stream, title: str, out_png: str) -> None:
    fs = result.sampling_rate
    n = result.prob_timeline.shape[1]
    t = np.arange(n) / fs
    z = stream.select(channel="*Z")[0].data if stream.select(channel="*Z") else stream[0].data
    z = np.asarray(z, dtype=float)
    tz = np.linspace(0, n / fs, len(z), endpoint=False)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 6.4), sharex=True, height_ratios=[2, 2, 1],
        facecolor=SURFACE,
    )
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)

    # 1 — vertical-component waveform with picks
    ax = axes[0]
    ax.plot(tz, z / (np.abs(z).max() or 1), color=INK2, linewidth=0.7)
    ax.set_ylabel("Z (norm.)", color=INK2, fontsize=9)
    for pick in result.picks:
        c = P_COLOR if pick.phase == "P" else S_COLOR
        ax.axvline(pick.time_s, color=c, linewidth=1.6)
        ax.annotate(
            f"{pick.phase}  conf {pick.prob:.2f}",
            xy=(pick.time_s, 1.0), xytext=(4, -2), textcoords="offset points",
            color=INK, fontsize=8.5, fontweight="bold",
        )
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)

    # 2 — rolling phase probabilities (what the model believed at each moment)
    ax = axes[1]
    ax.plot(t, result.prob_timeline[1], color=P_COLOR, linewidth=2.0, label="P probability")
    ax.plot(t, result.prob_timeline[2], color=S_COLOR, linewidth=2.0, label="S probability")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability", color=INK2, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)

    # 3 — inference latency per step
    ax = axes[2]
    ts = [s.t_s for s in result.steps]
    lat = [s.latency_ms for s in result.steps]
    ax.plot(ts, lat, color=INK2, linewidth=1.2)
    stats = result.latency_stats()
    ax.axhline(stats["p95_ms"], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate(
        f"p95 {stats['p95_ms']:.0f} ms", xy=(ts[-1], stats["p95_ms"]),
        xytext=(-2, 4), textcoords="offset points", ha="right",
        color=INK2, fontsize=8,
    )
    ax.set_ylabel("latency (ms)", color=INK2, fontsize=9)
    ax.set_xlabel("stream time (s) — historical waveform replayed as a real-time stream",
                  color=INK2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
