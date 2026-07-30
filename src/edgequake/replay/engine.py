"""Replay engine: historical waveform -> simulated real-time stream -> rolling inference.

Historical seismic waveform replayed as a real-time stream. NOT a live warning
service — this is the honest core of the whole project (see project plan §1).

Design:
    ObsPy Stream (3-comp) --hop--> RingBuffer --window--> Picker.predict()
        -> per-step: latest P/S probabilities, wall-clock inference latency
        -> trigger logic: threshold + refractory period -> picks with confidence

`speed` controls replay pacing: 0 = as fast as possible (batch validation),
1.0 = true real time (interactive demo), N = N-times faster than real time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..pickers.base import Picker


@dataclass
class Pick:
    phase: str            # "P" or "S"
    time_s: float         # seconds from stream start
    prob: float           # peak probability = confidence
    detected_at_s: float  # stream time when the trigger fired (>= time_s)

    @property
    def detection_delay_s(self) -> float:
        """How long after the arrival the system knew about it."""
        return self.detected_at_s - self.time_s


@dataclass
class StepRecord:
    t_s: float                       # stream time at end of this hop
    p_max: float                     # max P prob in the fresh part of the window
    s_max: float
    latency_ms: float                # wall-clock inference latency


@dataclass
class ReplayResult:
    steps: list[StepRecord] = field(default_factory=list)
    picks: list[Pick] = field(default_factory=list)
    prob_timeline: np.ndarray | None = None   # (3, n_samples) stitched N/P/S probs
    sampling_rate: float = 100.0

    def latency_stats(self) -> dict:
        lat = np.array([s.latency_ms for s in self.steps])
        return {
            "n_inferences": len(lat),
            "mean_ms": float(lat.mean()),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "max_ms": float(lat.max()),
        }


class ReplayEngine:
    def __init__(
        self,
        picker: Picker,
        hop_s: float = 0.5,
        p_threshold: float = 0.3,
        s_threshold: float = 0.3,
        refractory_s: float = 2.0,
    ):
        self.picker = picker
        self.hop_s = hop_s
        self.thresholds = {"P": p_threshold, "S": s_threshold}
        self.refractory_s = refractory_s

    def run(self, stream, speed: float = 0.0, on_step=None) -> ReplayResult:
        """stream: obspy.Stream with 3 traces, same sampling rate & alignment.

        on_step: optional callback(StepRecord, ReplayResult) for live UIs
        (WebSocket push, dashboard update, ...).
        """
        from .ring_buffer import RingBuffer

        fs_in = float(stream[0].stats.sampling_rate)
        fs = self.picker.sampling_rate
        st = stream.copy()
        if abs(fs_in - fs) > 1e-6:
            st.resample(fs)
        data = np.stack([tr.data.astype(np.float32) for tr in st])  # (3, N)
        n_total = data.shape[1]

        win = self.picker.window_samples
        hop = int(round(self.hop_s * fs))
        buf = RingBuffer(capacity=win, n_channels=3)

        result = ReplayResult(sampling_rate=fs)
        # stitched probability timeline: take the freshest `hop` samples of
        # every window's output so each stream sample gets exactly one estimate
        timeline = np.zeros((3, n_total), dtype=np.float32)
        timeline[0] = 1.0  # noise prior before first inference
        last_pick_t = {"P": -1e9, "S": -1e9}

        pos = 0
        while pos < n_total:
            chunk = data[:, pos : pos + hop]
            pos += chunk.shape[1]
            buf.push(chunk)
            t_stream = pos / fs

            t0 = time.perf_counter()
            out = self.picker.predict(buf.window())
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # freshest part of this window = the chunk we just pushed
            fresh = out.probs[:, -chunk.shape[1] :]
            timeline[:, pos - chunk.shape[1] : pos] = fresh

            step = StepRecord(
                t_s=t_stream,
                p_max=float(fresh[1].max()),
                s_max=float(fresh[2].max()),
                latency_ms=latency_ms,
            )
            result.steps.append(step)

            # trigger logic on the fresh region — suppressed during cold start:
            # a zero-padded buffer head creates an artificial step that looks
            # like an onset to the model (verified false P trigger at t=0)
            for ch, phase in ((1, "P"), (2, "S")) if buf.filled else ():
                thr = self.thresholds[phase]
                if fresh[ch].max() >= thr and t_stream - last_pick_t[phase] > self.refractory_s:
                    i_peak = int(fresh[ch].argmax())
                    t_pick = (pos - chunk.shape[1] + i_peak) / fs
                    result.picks.append(
                        Pick(phase=phase, time_s=t_pick,
                             prob=float(fresh[ch].max()), detected_at_s=t_stream)
                    )
                    last_pick_t[phase] = t_stream

            if on_step is not None:
                on_step(step, result)

            if speed > 0:  # pace like real time
                budget = self.hop_s / speed - (time.perf_counter() - t0)
                if budget > 0:
                    time.sleep(budget)

        result.prob_timeline = timeline
        return result
