"""Build a tiny SeisBench-format dataset for pipeline validation (NOT for
measuring model quality).

Each event trace = the ObsPy bundled real earthquake (BW.RJOB) injected into
tiled background noise at a random offset; P/S labels are the event's nominal
arrivals shifted by the offset. Nominal arrivals (P=4.51 s, S=5.72 s after
record start) come from the Phase 0 replay picks — approximate truth, good
enough to verify the evaluation plumbing end to end.

Output: data/synthetic_test/{metadata.csv, waveforms.hdf5}

Usage: python scripts/make_synthetic_dataset.py [n_events] [n_noise]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import obspy

ROOT = Path(__file__).resolve().parents[1]
P_NOMINAL_S, S_NOMINAL_S = 4.51, 5.72
TRACE_LEN_S = 60.0
FS = 100.0


def main() -> None:
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_noise = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rng = np.random.default_rng(7)

    st = obspy.read()
    st.detrend("demean")
    order = {"Z": 0, "N": 1, "E": 2}
    ev = np.zeros((3, st[0].stats.npts), dtype=np.float32)
    for tr in st:
        ev[order[tr.stats.channel[-1]]] = tr.data.astype(np.float32)
    noise_seg = ev[:, : int(3 * FS)]  # pre-event noise of the record itself

    n_total = int(TRACE_LEN_S * FS)

    from seisbench.data.base import WaveformDataWriter

    out = ROOT / "data" / "synthetic_test"
    out.mkdir(parents=True, exist_ok=True)
    with WaveformDataWriter(out / "metadata.csv", out / "waveforms.hdf5") as w:
        w.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": FS,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        def make_noise() -> np.ndarray:
            reps = int(np.ceil(n_total / noise_seg.shape[1]))
            base = np.tile(noise_seg, reps)[:, :n_total].copy()
            base += rng.normal(0, 0.3 * base.std(), base.shape).astype(np.float32)
            return base

        for k in range(n_events):
            data = make_noise()
            off = int(rng.integers(5 * FS, (TRACE_LEN_S - 35) * FS))
            scale = float(rng.uniform(0.7, 1.5))
            data[:, off : off + ev.shape[1]] += scale * ev
            w.add_trace(
                {
                    "trace_name": f"synth_ev_{k:04d}",
                    "trace_sampling_rate_hz": FS,
                    "trace_p_arrival_sample": off + P_NOMINAL_S * FS,
                    "trace_s_arrival_sample": off + S_NOMINAL_S * FS,
                    "split": "test",
                    "source_magnitude": round(float(rng.uniform(1.5, 3.0)), 1),
                    "station_code": "RJOB",
                },
                data,
            )
        for k in range(n_noise):
            w.add_trace(
                {
                    "trace_name": f"synth_no_{k:04d}",
                    "trace_sampling_rate_hz": FS,
                    "split": "test",
                    "station_code": "RJOB",
                },
                make_noise(),
            )
    print(f"[synth] wrote {n_events} event + {n_noise} noise traces -> {out}")


if __name__ == "__main__":
    main()
