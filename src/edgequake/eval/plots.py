"""Evaluation figures: residual histograms + threshold sweep.

Palette (validated): P = #2a78d6, S = #eb6834, ink grays for text/axes.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .picking import EvalResult

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
P_COLOR, S_COLOR = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_evaluation(result: EvalResult, threshold: float, out_png: str,
                    sweep: np.ndarray | None = None) -> None:
    if sweep is None:
        sweep = np.round(np.arange(0.05, 0.95, 0.05), 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    for ax in axes:
        _style(ax)

    # 1 — residual histograms at the operating threshold
    ax = axes[0]
    res = result.residuals_at(threshold)
    bins = np.linspace(-result.tolerance_s, result.tolerance_s, 41)
    for phase, color in (("P", P_COLOR), ("S", S_COLOR)):
        r = res[phase]
        if len(r):
            ax.hist(r, bins=bins, histtype="step", linewidth=2.0,
                    color=color, label=f"{phase} (n={len(r)})")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("pick residual (s)", color=INK2, fontsize=9)
    ax.set_ylabel("count", color=INK2, fontsize=9)
    ax.set_title(f"Residuals @ thr={threshold}", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # 2 — F1 vs threshold
    ax = axes[1]
    f1 = {"P": [], "S": []}
    for thr in sweep:
        m = result.metrics_at(float(thr))
        for phase in ("P", "S"):
            f1[phase].append(m["phases"][phase]["f1"])
    for row, (phase, color) in enumerate((("P", P_COLOR), ("S", S_COLOR))):
        ax.plot(sweep, f1[phase], color=color, linewidth=2.0, label=phase)
        best = int(np.argmax(f1[phase]))
        ax.plot(sweep[best], f1[phase][best], "o", color=color, markersize=8)
        ax.annotate(f"{phase} best {f1[phase][best]:.2f} @ thr {sweep[best]:.2f}",
                    xy=(0.02, 0.16 - 0.11 * row), xycoords="axes fraction",
                    fontsize=8, color=INK)
    ax.set_xlabel("detection threshold", color=INK2, fontsize=9)
    ax.set_ylabel("F1", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold sweep", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")

    # 3 — precision-recall curves over the sweep
    ax = axes[2]
    for phase, color in (("P", P_COLOR), ("S", S_COLOR)):
        pr, rc = [], []
        for thr in sweep:
            m = result.metrics_at(float(thr))["phases"][phase]
            if m["tp"] + m["fp"] == 0:   # nothing detected — skip degenerate point
                continue
            pr.append(m["precision"])
            rc.append(m["recall"])
        ax.plot(rc, pr, color=color, linewidth=2.0, label=phase)
    ax.set_xlabel("recall", color=INK2, fontsize=9)
    ax.set_ylabel("precision", color=INK2, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision–recall", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.suptitle(
        f"{result.picker} on {result.dataset} — tol ±{result.tolerance_s}s, "
        f"{len(result.caches)} windows",
        color=INK, fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
