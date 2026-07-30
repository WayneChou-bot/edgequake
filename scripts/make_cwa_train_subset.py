"""Build a compact PhaseNet fine-tuning subset from the locally cached CWA
2019 chunk — small enough (~4 GB) to upload to Kaggle as a Dataset.

Why 2019: the official split is <=2018 train / 2019 dev / >=2020 test, but the
<=2018 waveform files are 47-89 GB each — beyond both a laptop link and Kaggle
disk. Training on the locally available dev year (2019) is a disclosed,
resource-constrained deviation; the 2020-2021 TEST years stay untouched.

What it does per trace: resample to 100 Hz (handled by SeisBench), rescale
arrival samples from the trace's original rate, trim to [P-10s, S+20s] capped
at 60 s, store float32 ZNE. Output: SeisBench-format dir with standard column
names and a 90/10 train/dev split.

Usage:
    python scripts/make_cwa_train_subset.py                # 70k traces max
    python scripts/make_cwa_train_subset.py --max-traces 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FS = 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="*", default=["_2019"])
    ap.add_argument("--max-traces", type=int, default=70_000)
    ap.add_argument("--out", default=str(ROOT / "data" / "cwa2019_compact"))
    ap.add_argument("--before-s", type=float, default=10.0, help="margin before P")
    ap.add_argument("--after-s", type=float, default=20.0, help="margin after S")
    ap.add_argument("--cap-s", type=float, default=60.0, help="max stored length")
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-path", default=None,
                    help="use any local SeisBench dataset dir instead of CWA")
    args = ap.parse_args()

    from edgequake.data.loader import arrival_columns, load, load_cwa

    if args.source_path:
        ds = load(args.source_path, sampling_rate=FS, component_order="ZNE")
    else:
        ds = load_cwa(args.chunks, sampling_rate=FS, component_order="ZNE",
                      confirm=True)  # cached locally -> no download happens
    meta = ds.metadata
    cols = arrival_columns(meta)
    print(f"[subset] source traces: {len(meta)}, arrival cols: {cols}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(meta))

    from seisbench.data.base import WaveformDataWriter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    bytes_est = 0
    with WaveformDataWriter(out / "metadata.csv", out / "waveforms.hdf5") as w:
        w.data_format = {
            "dimension_order": "CW", "component_order": "ZNE",
            "sampling_rate": FS, "measurement": "velocity",
            "unit": "counts", "instrument_response": "not restituted",
        }
        for i in order:
            if n_written >= args.max_traces:
                break
            row = meta.iloc[int(i)]
            fs_trace = float(row.get("trace_sampling_rate_hz", FS) or FS)
            scale = FS / fs_trace
            try:
                p = float(row[cols["P"]]) * scale
                s = float(row[cols["S"]]) * scale
            except (KeyError, TypeError, ValueError):
                n_skipped += 1
                continue
            if not (np.isfinite(p) and np.isfinite(s)) or s <= p:
                n_skipped += 1
                continue
            wf = ds.get_waveforms(int(i))
            if wf.shape[0] != 3 or np.isnan(wf).any():
                n_skipped += 1
                continue
            start = int(max(0, p - args.before_s * FS))
            end = int(min(wf.shape[1], s + args.after_s * FS,
                          start + args.cap_s * FS))
            if end - start < 15 * FS or s >= end:  # too short / S clipped out
                n_skipped += 1
                continue
            seg = wf[:, start:end].astype(np.float32)
            w.add_trace({
                "trace_name": f"cwa19_{n_written:06d}",
                "trace_sampling_rate_hz": FS,
                "trace_p_arrival_sample": p - start,
                "trace_s_arrival_sample": s - start,
                "split": "dev" if rng.random() < args.dev_fraction else "train",
                "source_magnitude": row.get("source_magnitude", np.nan),
                "station_code": row.get("station_code", ""),
                "source_origin_time": row.get("source_origin_time", ""),
            }, seg)
            bytes_est += seg.nbytes
            n_written += 1
            if n_written % 5000 == 0:
                print(f"[subset] {n_written} written "
                      f"(~{bytes_est / 1e9:.1f} GB), {n_skipped} skipped")
    print(f"[subset] DONE: {n_written} traces (~{bytes_est / 1e9:.1f} GB raw), "
          f"{n_skipped} skipped -> {out}")
    print("[subset] upload this folder as a Kaggle Dataset "
          "(metadata.csv + waveforms.hdf5)")


if __name__ == "__main__":
    main()
