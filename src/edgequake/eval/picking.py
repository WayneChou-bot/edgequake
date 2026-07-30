"""Phase-picking evaluation following the "Which picker fits my data?" protocol
(Münchmeyer et al. 2022, JGR, doi:10.1029/2021JB023499), simplified:

For every test trace, cut a window that contains the labeled arrival at a
random (seeded) position away from the edges, run the picker ONCE, cache the
per-sample P/S probability curves, then sweep detection thresholds on the
cached curves. A predicted peak matches a label if |residual| <= tolerance.

Metrics per phase and threshold: precision, recall, F1, and residual stats
(MAE, RMSE, std) over matched picks. Noise traces (no labels) contribute
false-alarm counts normalized to false alarms per trace-hour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.loader import arrival_columns
from ..pickers.base import Picker


def local_peaks(prob: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    """Indices of local maxima above threshold, greedily separated by min_dist."""
    above = prob >= threshold
    if not above.any():
        return []
    idx = np.argsort(prob)[::-1]
    peaks: list[int] = []
    for i in idx:
        if not above[i]:
            break
        if all(abs(i - p) >= min_dist for p in peaks):
            peaks.append(int(i))
    return sorted(peaks)


@dataclass
class TraceCache:
    """One evaluated trace: cached probability curves + ground-truth samples."""
    probs: np.ndarray                    # (3, W) noise/P/S
    truth: dict[str, float]              # phase -> arrival sample within window
    is_noise: bool = False


@dataclass
class EvalResult:
    picker: str
    dataset: str
    tolerance_s: float
    sampling_rate: float
    window_hours: float = 0.0            # total evaluated signal duration
    caches: list[TraceCache] = field(default_factory=list)

    def metrics_at(self, threshold: float, min_dist_s: float = 1.0) -> dict:
        fs = self.sampling_rate
        tol = int(round(self.tolerance_s * fs))
        out: dict = {"threshold": threshold, "phases": {}, "noise_false_alarms": 0}
        for ch, phase in ((1, "P"), (2, "S")):
            tp = fp = fn = 0
            residuals: list[float] = []
            for c in self.caches:
                peaks = local_peaks(c.probs[ch], threshold, int(min_dist_s * fs))
                if c.is_noise or phase not in c.truth:
                    fp += len(peaks)
                    continue
                t_true = c.truth[phase]
                matched = [p for p in peaks if abs(p - t_true) <= tol]
                if matched:
                    tp += 1
                    best = min(matched, key=lambda p: abs(p - t_true))
                    residuals.append((best - t_true) / fs)
                    fp += len(peaks) - 1  # extra peaks on a labeled trace
                else:
                    fn += 1
                    fp += len(peaks)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            r = np.array(residuals)
            out["phases"][phase] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4),
                "residual_mae_s": round(float(np.abs(r).mean()), 4) if len(r) else None,
                "residual_rmse_s": round(float(np.sqrt((r ** 2).mean())), 4) if len(r) else None,
                "residual_std_s": round(float(r.std()), 4) if len(r) else None,
                "n_residuals": len(r),
            }
        n_noise_peaks = sum(
            len(local_peaks(c.probs[ch], threshold, int(min_dist_s * fs)))
            for c in self.caches if c.is_noise for ch in (1, 2)
        )
        noise_hours = sum(c.probs.shape[1] for c in self.caches if c.is_noise) / fs / 3600
        out["noise_false_alarms"] = n_noise_peaks
        out["noise_false_alarms_per_hour"] = (
            round(n_noise_peaks / noise_hours, 2) if noise_hours else None
        )
        return out

    def residuals_at(self, threshold: float, min_dist_s: float = 1.0) -> dict[str, np.ndarray]:
        fs = self.sampling_rate
        tol = int(round(self.tolerance_s * fs))
        res: dict[str, list[float]] = {"P": [], "S": []}
        for ch, phase in ((1, "P"), (2, "S")):
            for c in self.caches:
                if c.is_noise or phase not in c.truth:
                    continue
                peaks = local_peaks(c.probs[ch], threshold, int(min_dist_s * fs))
                matched = [p for p in peaks if abs(p - c.truth[phase]) <= tol]
                if matched:
                    best = min(matched, key=lambda p: abs(p - c.truth[phase]))
                    res[phase].append((best - c.truth[phase]) / fs)
        return {k: np.array(v) for k, v in res.items()}


def evaluate_picker(
    picker: Picker,
    dataset,
    dataset_name: str = "dataset",
    split: str = "test",
    limit: int | None = None,
    tolerance_s: float = 0.5,
    edge_fraction: float = 0.25,
    seed: int = 42,
) -> EvalResult:
    """Run the picker over labeled windows of `dataset` and cache prob curves.

    edge_fraction: labeled arrival is placed uniformly inside the central
    (1 - 2*edge_fraction) of the window, so the model cannot exploit a fixed
    arrival position and edge effects don't clip the target.
    """
    rng = np.random.default_rng(seed)
    ds = {"train": dataset.train, "dev": dataset.dev, "test": dataset.test}.get(split)
    ds = ds() if ds is not None else dataset
    meta = ds.metadata
    cols = arrival_columns(meta)
    if not cols:
        raise ValueError("No arrival-sample columns found — is this a noise set?")

    win = picker.window_samples
    fs = picker.sampling_rate
    lo, hi = int(win * edge_fraction), int(win * (1 - edge_fraction))

    n = len(meta) if limit is None else min(limit, len(meta))
    order = rng.permutation(len(meta))[:n]

    result = EvalResult(picker=picker.name, dataset=dataset_name,
                        tolerance_s=tolerance_s, sampling_rate=fs)
    for i in order:
        row = meta.iloc[int(i)]
        wf = ds.get_waveforms(int(i))          # (C, N) at picker fs (loader resamples)
        if wf.shape[0] != 3 or np.isnan(wf).any():
            continue
        # arrival samples in metadata are in the trace's ORIGINAL sampling
        # rate; waveforms are resampled to the picker rate by the loader —
        # scale the arrivals accordingly (matters for 200 Hz TSMIP traces)
        fs_trace = float(row.get("trace_sampling_rate_hz", fs) or fs)
        scale = fs / fs_trace
        # anchor the window on the earliest labeled arrival present
        arrivals = {ph: float(row[c]) * scale for ph, c in cols.items()
                    if c in row and np.isfinite(row[c])}
        if not arrivals:
            # unlabeled trace = noise window: count its false alarms
            if wf.shape[1] >= win:
                start = int(rng.integers(0, wf.shape[1] - win + 1))
                out = picker.predict(wf[:, start : start + win])
                result.caches.append(
                    TraceCache(probs=out.probs, truth={}, is_noise=True)
                )
            continue
        anchor_phase = min(arrivals, key=arrivals.get)
        anchor = arrivals[anchor_phase]
        target_pos = int(rng.integers(lo, hi))
        start = int(round(anchor - target_pos))
        start = max(0, min(start, wf.shape[1] - win))
        if wf.shape[1] < win:
            continue
        window = wf[:, start : start + win]
        truth = {ph: a - start for ph, a in arrivals.items()
                 if 0 <= a - start < win}
        out = picker.predict(window)
        result.caches.append(TraceCache(probs=out.probs, truth=truth))
        result.window_hours += win / fs / 3600
    return result


def evaluate_noise(picker: Picker, dataset, limit: int | None = None,
                   result: EvalResult | None = None, seed: int = 43) -> EvalResult:
    """Append noise-only windows (false-alarm measurement) to an EvalResult."""
    rng = np.random.default_rng(seed)
    meta = dataset.metadata
    n = len(meta) if limit is None else min(limit, len(meta))
    order = rng.permutation(len(meta))[:n]
    win = picker.window_samples
    if result is None:
        result = EvalResult(picker=picker.name, dataset="noise",
                            tolerance_s=0.5, sampling_rate=picker.sampling_rate)
    for i in order:
        wf = dataset.get_waveforms(int(i))
        if wf.shape[0] != 3 or wf.shape[1] < win or np.isnan(wf).any():
            continue
        start = int(rng.integers(0, wf.shape[1] - win + 1))
        out = picker.predict(wf[:, start : start + win])
        result.caches.append(TraceCache(probs=out.probs, truth={}, is_noise=True))
    return result
