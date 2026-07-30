"""Phase 0 demo: stream a real earthquake record through the replay engine.

Usage:
    python scripts/demo_replay.py                 # ObsPy bundled example event
    python scripts/demo_replay.py path/to/*.mseed # any 3-component miniSEED

Picker resolution order:
    1. SeisBench pretrained PhaseNet (normal laptop path, needs weight download)
    2. AI4EPS TensorFlow checkpoint in ./PhaseNet (offline/GitHub-only path)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import obspy

from edgequake.replay.engine import ReplayEngine
from edgequake.viz import plot_replay


def load_stream(argv: list[str]) -> tuple[obspy.Stream, str]:
    if len(argv) > 1:
        st = obspy.read(argv[1])
        label = Path(argv[1]).stem
    else:
        st = obspy.read()  # bundled example: BW.RJOB 2009-08-24 local event
        label = "BW.RJOB 2009-08-24 (ObsPy bundled example event)"
    st.detrend("demean")
    return st, label


def prepend_warmup_noise(st: obspy.Stream, warmup_s: float = 30.0,
                         noise_from_s: float = 3.0) -> obspy.Stream:
    """Simulate continuous streaming: tile the record's own pre-event noise in
    front of it so the picker's window fills with realistic background before
    the event arrives (a live station is never cold-started on an event)."""
    import numpy as np

    out = st.copy()
    for tr in out:
        fs = tr.stats.sampling_rate
        n_noise = int(noise_from_s * fs)
        n_pad = int(warmup_s * fs)
        noise = tr.data[:n_noise].astype(tr.data.dtype)
        reps = int(np.ceil(n_pad / n_noise))
        pad = np.tile(noise, reps)[:n_pad]
        tr.data = np.concatenate([pad, tr.data])
        tr.stats.starttime -= warmup_s
    return out


def build_picker():
    try:
        from edgequake.pickers.seisbench_picker import SeisBenchPhaseNet

        return SeisBenchPhaseNet(weights="original")
    except Exception as e:  # offline fallback
        print(f"[demo] SeisBench weights unavailable ({type(e).__name__}); "
              "falling back to TF checkpoint")
        from edgequake.pickers.tf_phasenet import TFPhaseNet

        return TFPhaseNet(repo_dir=str(Path(__file__).resolve().parents[1] / "PhaseNet"))


def main() -> None:
    st, label = load_stream(sys.argv)
    warmup_s = 30.0
    st = prepend_warmup_noise(st, warmup_s=warmup_s)
    picker = build_picker()
    print(f"[demo] picker = {picker.name}")

    # NOTE: with the PhaseNet default threshold 0.3 the demo event's P peak
    # (0.27, out-of-domain station) is missed — threshold choice is exactly the
    # calibration question Phase 1 studies. 0.25 catches it; both are defensible.
    engine = ReplayEngine(picker, hop_s=0.5, p_threshold=0.25, s_threshold=0.3)
    result = engine.run(st, speed=0.0)  # 0 = fast validation; 1.0 = true real time

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    png = out_dir / "replay_demo.png"
    plot_replay(result, st, f"EdgeQuake replay — {label} — {picker.name}", str(png))

    summary = {
        "picker": picker.name,
        "hop_s": engine.hop_s,
        "picks": [
            {
                "phase": p.phase,
                "time_s": round(p.time_s, 2),
                "confidence": round(p.prob, 3),
                "detection_delay_s": round(p.detection_delay_s, 2),
            }
            for p in result.picks
        ],
        "latency": result.latency_stats(),
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[demo] figure -> {png}")


if __name__ == "__main__":
    main()
