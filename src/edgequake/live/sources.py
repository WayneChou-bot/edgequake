"""Pluggable waveform sources for the live engine.

Every source yields Ticks: a batch of Packets plus the current stream time.
Times are epoch seconds (UTC). Data units:
    kind="pick" : raw counts (the picker normalizes per window)
    kind="acc"  : physical acceleration in cm/s^2 (for PGA / magnitude)

Sources:
    ReplaySource   — GDMS miniSEED replayed at wall-clock (or accelerated)
                     speed. Fully offline; used for development and demos.
    SeedLinkSource — real-time SeedLink client (validated against GEOFON;
                     no public Taiwan feed exists today). Counts only, so
                     magnitude needs a channel->sensitivity map if used
                     for more than pick/latency validation.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

PICK_PRIORITY = ["HH", "EH", "HL"]


@dataclass
class Packet:
    code: str
    comp: str          # "Z" | "N" | "E" (or "1"/"2" mapped to N/E)
    kind: str          # "pick" | "acc"
    fs: float
    t0: float          # epoch of first sample
    data: np.ndarray   # float32


@dataclass
class Tick:
    now: float                     # stream time, epoch seconds
    packets: list = field(default_factory=list)


@dataclass
class StationMeta:
    code: str
    lat: float
    lon: float
    pick_channel: str | None = None


class ReplaySource:
    """Replay GDMS miniSEED as a real-time stream.

    Chunks every trace into `chunk_s` slices and releases them on a simulated
    clock running at `speed`x wall time. Acceleration (HL) traces are
    converted to cm/s^2 via the dataless inventory at load time.
    """

    def __init__(self, mseed_files, dataless_path, speed=1.0, chunk_s=1.0,
                 start_offset_s=20.0, max_stations=None, fs=100.0):
        import obspy
        from obspy import read, read_inventory

        st = obspy.Stream()
        for f in mseed_files:
            st += read(str(f))
        st.merge(method=1, fill_value=0)
        inv = read_inventory(str(dataless_path))

        by_sta: dict[str, dict[str, dict[str, obspy.Trace]]] = {}
        for tr in st:
            fam, comp = tr.stats.channel[:2], tr.stats.channel[-1]
            by_sta.setdefault(tr.stats.station, {}).setdefault(fam, {})[comp] = tr

        self.fs = fs
        self.chunk_s = chunk_s
        self.speed = speed
        self.stations: dict[str, StationMeta] = {}
        self._traces: list[tuple[Packet, np.ndarray, float]] = []  # meta only
        self._series: list[dict] = []

        t_min = None
        for code, fams in sorted(by_sta.items()):
            if max_stations and len(self.stations) >= max_stations:
                break
            coord = None
            for fam, comps in fams.items():
                for tr in comps.values():
                    try:
                        coord = inv.get_coordinates(tr.id, tr.stats.starttime)
                        break
                    except Exception:
                        continue
                if coord:
                    break
            if not coord:
                continue

            pick_fam = next((f for f in PICK_PRIORITY
                             if f in fams and len(fams[f]) == 3), None)
            entries = []
            if pick_fam:
                comps = fams[pick_fam]
                order = [c for c in "ZNE" if c in comps]
                if len(order) != 3:
                    order = sorted(comps.keys())
                for want, key in zip("ZNE", order):
                    tr = comps[key]
                    entries.append(dict(comp=want, kind="pick",
                                        data=tr.data.astype(np.float32),
                                        fs=float(tr.stats.sampling_rate),
                                        t0=float(tr.stats.starttime.timestamp)))
            if "HL" in fams:
                for tr in fams["HL"].values():
                    try:
                        resp = inv.get_response(tr.id, tr.stats.starttime)
                        sens = resp.instrument_sensitivity.value
                    except Exception:
                        continue
                    acc = tr.data.astype(np.float32) / sens * 100.0  # cm/s^2
                    entries.append(dict(comp=tr.stats.channel[-1], kind="acc",
                                        data=acc,
                                        fs=float(tr.stats.sampling_rate),
                                        t0=float(tr.stats.starttime.timestamp)))
            if not entries:
                continue
            self.stations[code] = StationMeta(
                code=code, lat=coord["latitude"], lon=coord["longitude"],
                pick_channel=pick_fam)
            for e in entries:
                e["code"] = code
                self._series.append(e)
            t0s = min(e["t0"] for e in entries)
            t_min = t0s if t_min is None else min(t_min, t0s)

        self.t_start = (t_min or 0.0) + start_offset_s
        self.t_end = max(e["t0"] + len(e["data"]) / e["fs"]
                         for e in self._series)

    def ticks(self) -> Iterator[Tick]:
        now = self.t_start
        wall0 = time.monotonic()
        sim0 = now
        while now < self.t_end:
            nxt = now + self.chunk_s
            pkts = []
            for e in self._series:
                i0 = int(max(0.0, (now - e["t0"])) * e["fs"])
                i1 = int(max(0.0, (nxt - e["t0"])) * e["fs"])
                i1 = min(i1, len(e["data"]))
                if i1 <= i0:
                    continue
                pkts.append(Packet(code=e["code"], comp=e["comp"],
                                   kind=e["kind"], fs=e["fs"],
                                   t0=e["t0"] + i0 / e["fs"],
                                   data=e["data"][i0:i1]))
            # pace the simulated clock
            target_wall = wall0 + (nxt - sim0) / self.speed
            lag = target_wall - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            yield Tick(now=nxt, packets=pkts)
            now = nxt


class SeedLinkSource:
    """Real-time SeedLink client (background thread -> queue -> ticks).

    Validated against GEOFON (geofon.gfz.de:18000, ~4-7 s feed latency).
    There is currently no public Taiwan SeedLink feed; this source exists so
    the engine is live-ready the moment one is available (e.g. a personal
    Raspberry Shake). Data stays in counts -> pick/latency validation only,
    unless a {station: counts-per-cm/s^2} sensitivity map is provided.
    """

    def __init__(self, server, streams, stations_meta, chunk_s=1.0,
                 sensitivity=None):
        self.server = server
        self.streams = streams          # [(net, sta, chan_selector)]
        self.stations = {m.code: m for m in stations_meta}
        self.chunk_s = chunk_s
        self.sensitivity = sensitivity or {}
        self._q: queue.Queue = queue.Queue()
        self._thread = None

    def _run_client(self):
        from obspy.clients.seedlink.easyseedlink import create_client

        def on_data(tr):
            comp = tr.stats.channel[-1]
            comp = {"1": "N", "2": "E"}.get(comp, comp)
            self._q.put(Packet(
                code=tr.stats.station, comp=comp, kind="pick",
                fs=float(tr.stats.sampling_rate),
                t0=float(tr.stats.starttime.timestamp),
                data=tr.data.astype(np.float32)))

        client = create_client(self.server, on_data=on_data)
        for net, sta, sel in self.streams:
            client.select_stream(net, sta, sel)
        client.run()

    def ticks(self) -> Iterator[Tick]:
        self._thread = threading.Thread(target=self._run_client, daemon=True)
        self._thread.start()
        while True:
            time.sleep(self.chunk_s)
            pkts = []
            while True:
                try:
                    pkts.append(self._q.get_nowait())
                except queue.Empty:
                    break
            yield Tick(now=time.time(), packets=pkts)
