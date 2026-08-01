"""Extract P-wave onset windows for the AI early-magnitude model (Phase 5).

Task: single-station "first seconds of P" -> final catalog magnitude.
For every CWA trace with a P arrival and a magnitude label, cut a window
[P - PRE, P + POST] (default 2 s + 4 s = 600 samples @100 Hz; the training
notebook crops 4 s sub-windows with random jitter for pick-time robustness).

Amplitude handling: waveforms are raw counts (not restituted). The window is
stored peak-normalized (float16) plus log10(peak) / log10(std) scalars, so
the model sees shape AND absolute scale — magnitude information lives in
both. Dead-channel guard: traces with insane counts are skipped.

Anti-leakage rules baked in:
  * windows never see past P + POST (no S, no PGA — nothing from the future)
  * the notebook splits train/dev BY EVENT ID, never by trace
  * temporal protocol: train/dev on _2019, test on _2020+_2021,
    blind test on 0403/Dapu GDMS raw data (2024/2025)

Usage (laptop; chunks are already in the local SeisBench cache):
    python scripts/extract_magwin.py                       # 2019+2020+2021
    python scripts/extract_magwin.py --chunks _2019        # train year only
Output: data/magwin/magwin<chunk>.npz  (upload folder as a Kaggle Dataset)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FS = 100.0


def extract_chunk(chunk: str, pre_s: float, post_s: float, out_dir: Path,
                  max_traces: int | None) -> None:
    from edgequake.data.loader import arrival_columns, load_cwa

    ds = load_cwa([chunk], sampling_rate=FS, component_order="ZNE",
                  confirm=True)  # cached locally -> no download happens
    meta = ds.metadata
    pcol = arrival_columns(meta)["P"]
    n_pre, n_post = int(pre_s * FS), int(post_s * FS)
    win = n_pre + n_post

    X, amp, y, ev_ids, stations, chans, dists = [], [], [], [], [], [], []
    n_skip = 0
    t0 = time.perf_counter()
    n_total = len(meta) if max_traces is None else min(len(meta), max_traces)
    for i in range(n_total):
        row = meta.iloc[i]
        mag = row.get("source_magnitude", np.nan)
        fs_trace = float(row.get("trace_sampling_rate_hz", FS) or FS)
        try:
            p = float(row[pcol]) * (FS / fs_trace)
        except (KeyError, TypeError, ValueError):
            n_skip += 1
            continue
        if not (np.isfinite(p) and np.isfinite(mag)):
            n_skip += 1
            continue
        start = int(round(p)) - n_pre
        try:
            wf = ds.get_waveforms(i)
        except Exception:
            # CWA metadata occasionally references blocks missing from the
            # waveform hdf5 (e.g. 20201107 ILA052) — skip, don't crash
            n_skip += 1
            continue
        if wf.shape[0] != 3 or start < 0 or start + win > wf.shape[1]:
            n_skip += 1
            continue
        seg = wf[:, start:start + win].astype(np.float64)
        if np.isnan(seg).any():
            n_skip += 1
            continue
        peak = float(np.abs(seg).max())
        std = float(seg.std())
        # dead/blown channel guard (the BatchNorm-poisoning lesson, again)
        if peak <= 0 or std <= 0 or peak > 1e8:
            n_skip += 1
            continue
        # hypocentral distance (v2: distance-conditioned aux input — the
        # model shouldn't have to marginalize over unknown distance)
        try:
            from edgequake.location.locator import haversine_km
            d_ep = haversine_km(float(row["source_latitude_deg"]),
                                float(row["source_longitude_deg"]),
                                float(row["station_latitude_deg"]),
                                float(row["station_longitude_deg"]))
            d_hyp = float(np.hypot(d_ep,
                                   float(row.get("source_depth_km", 0) or 0)))
        except (KeyError, TypeError, ValueError):
            d_hyp = np.nan
        X.append((seg / peak).astype(np.float16))
        amp.append((np.log10(peak), np.log10(std)))
        dists.append(d_hyp)
        y.append(float(mag))
        ev_ids.append(str(row.get("source_event_id", "")))
        stations.append(str(row.get("station_code", "")))
        chans.append(str(row.get("trace_channel",
                                 row.get("station_channels", "")))[:8])
        if len(X) % 20000 == 0:
            rate = len(X) / (time.perf_counter() - t0)
            print(f"[magwin]{chunk} {len(X)} windows "
                  f"({i + 1}/{n_total} traces, {rate:.0f}/s)")

    if not X:
        print(f"[magwin]{chunk} nothing extracted ({n_skip} skipped)")
        return
    Xa = np.stack(X)
    ya = np.array(y, dtype=np.float32)
    out = out_dir / f"magwin{chunk}.npz"
    np.savez_compressed(
        out, X=Xa, amp=np.array(amp, dtype=np.float32), y=ya,
        dist=np.array(dists, dtype=np.float32),
        event=np.array(ev_ids), station=np.array(stations),
        chan=np.array(chans),
        meta=np.array([FS, pre_s, post_s], dtype=np.float32))
    gb = out.stat().st_size / 1e9
    hist, edges = np.histogram(ya, bins=[0, 3, 4, 5, 6, 8])
    print(f"[magwin]{chunk} DONE: {len(Xa)} windows ({n_skip} skipped) "
          f"-> {out.name} ({gb:.2f} GB)")
    print(f"[magwin]{chunk} events {len(set(ev_ids))} | mag bins "
          f"<3:{hist[0]} 3-4:{hist[1]} 4-5:{hist[2]} 5-6:{hist[3]} "
          f"6+:{hist[4]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="*",
                    default=["_2019", "_2020", "_2021"])
    ap.add_argument("--pre", type=float, default=2.0,
                    help="seconds before P (crop margin for jitter aug)")
    ap.add_argument("--post", type=float, default=4.0,
                    help="seconds after P")
    ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "magwin"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in args.chunks:
        extract_chunk(chunk, args.pre, args.post, out_dir, args.max_traces)
    print(f"[magwin] all done -> {out_dir}")
    print("[magwin] upload the magwin folder as a Kaggle Dataset, then run "
          "kaggle/edgequake_magnet.ipynb")


if __name__ == "__main__":
    main()
