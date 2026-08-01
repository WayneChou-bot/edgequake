"""ExpTech TREM real-time source (Phase 6): Taiwan's community MEMS network.

TREM-Net (ExpTech 探索科技) publishes per-station real-time ground-motion
values — PGA / PGV / intensity, ~1 s cadence — via an open HTTP endpoint:
    station list:  https://api-1.exptech.dev/api/v1/trem/station
    realtime:      https://api-1.exptech.dev/api/v1/trem/rts

This is NOT raw waveform (no PhaseNet picking possible). Instead the engine
runs in TRIGGER mode: a station whose PGA jumps above threshold counts as a
triggered arrival; association -> location -> PGA magnitude -> county PWS
all reuse the existing physics chain unchanged.

Usage etiquette (their own apps poll the same endpoint at 1 Hz):
  * poll interval >= 1 s, single connection
  * attribute the source in any UI ("資料來源：ExpTech TREM，僅供參考")
  * research / personal use — not a public warning service

TremSimSource replays a historical event as synthetic RTS frames (built from
outputs/replay_<event>.json) so the trigger chain can be developed and
demoed fully offline.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Iterator

import numpy as np

from .sources import Packet, StationMeta, Tick

API = "https://api-1.exptech.dev/api/v1/trem"


def _get_json(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, headers={
        "User-Agent": "EdgeQuake-research/0.1 (github.com/WayneChou-bot)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class TremRtsSource:
    """Live polling of the TREM RTS endpoint (1 Hz)."""

    def __init__(self, poll_s: float = 1.0):
        self.poll_s = max(1.0, poll_s)   # never hammer a community API
        raw = _get_json(f"{API}/station")
        self.stations: dict[str, StationMeta] = {}
        for sid, ent in raw.items():
            try:
                if not ent.get("work", False):
                    continue
                info = ent["info"][-1]   # latest siting record
                self.stations[str(sid)] = StationMeta(
                    code=str(sid), lat=float(info["lat"]),
                    lon=float(info["lon"]),
                    pick_channel=str(ent.get("net", "TREM")))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        print(f"[trem] station list: {len(self.stations)} working stations")

    def ticks(self) -> Iterator[Tick]:
        while True:
            t0 = time.monotonic()
            pkts = []
            try:
                rts = _get_json(f"{API}/rts")
                t_data = float(rts.get("time", time.time() * 1e3)) / 1e3
                for sid, v in (rts.get("station") or {}).items():
                    if str(sid) not in self.stations:
                        continue
                    pkts.append(Packet(
                        code=str(sid), comp="T", kind="trig", fs=1.0,
                        t0=t_data,
                        data=np.array([float(v.get("pga", 0.0)),
                                       float(v.get("pgv", 0.0)),
                                       float(v.get("i", -3.0))],
                                      dtype=np.float32)))
                yield Tick(now=t_data, packets=pkts)
            except Exception as e:
                print(f"[trem] poll failed ({e}); retrying...")
                yield Tick(now=time.time(), packets=[])
            lag = self.poll_s - (time.monotonic() - t0)
            if lag > 0:
                time.sleep(lag)


class TremSimSource:
    """Offline simulation: synthesize RTS frames from a replay JSON.

    Model per station: quiet noise until the S wave arrives (t_s, falling
    back to t_p + distance-scaled S-P), then PGA ramps to the recorded final
    value over ~4 s and decays. Crude but exercises the whole trigger chain.
    """

    def __init__(self, replay_json_path, speed: float = 1.0,
                 start_offset_s: float = 15.0):
        d = json.loads(open(replay_json_path).read())
        self.speed = speed
        self.stations: dict[str, StationMeta] = {}
        self._ev = []
        import obspy

        origin = obspy.UTCDateTime(d["origin_utc"].replace("Z", ""))
        for s in d["stations"]:
            if s.get("lat") is None:
                continue
            code = str(s["code"])
            self.stations[code] = StationMeta(code=code, lat=s["lat"],
                                              lon=s["lon"],
                                              pick_channel="SIM")
            tp = (float(obspy.UTCDateTime(s["t_p"]) - origin)
                  if s.get("t_p") else None)
            ts = (float(obspy.UTCDateTime(s["t_s"]) - origin)
                  if s.get("t_s") else (tp + 6.0 if tp else None))
            pga = float(s["pga_cmps2"]) if s.get("pga_cmps2") else None
            self._ev.append((code, tp, ts, pga))
        self.t_start = -start_offset_s
        self.t_end = max((ts or 0) for _, _, ts, _ in self._ev) + 30.0
        self._t0_wall = None

    def _pga_at(self, tp, ts, pga_final, t):
        rng = abs(hash((tp, ts))) % 100 / 100.0
        noise = 0.15 + 0.5 * rng
        if tp is None or pga_final is None or t < tp:
            return noise
        if t < ts:                      # P-wave phase: a few % of final
            return max(noise, 0.05 * pga_final)
        if t < ts + 4.0:                # S ramp
            return max(noise, pga_final * (0.3 + 0.7 * (t - ts) / 4.0))
        return max(noise, pga_final * np.exp(-(t - ts - 4.0) / 20.0))

    def ticks(self) -> Iterator[Tick]:
        epoch0 = 1_700_000_000.0        # arbitrary fixed epoch for the sim
        t = self.t_start
        wall0 = time.monotonic()
        while t < self.t_end:
            pkts = []
            for code, tp, ts, pga in self._ev:
                p = self._pga_at(tp, ts, pga, t)
                i = np.log10(max(p, 1e-3)) * 2.0     # rough intensity-ish
                pkts.append(Packet(code=code, comp="T", kind="trig", fs=1.0,
                                   t0=epoch0 + t,
                                   data=np.array([p, p / 10.0, i],
                                                 dtype=np.float32)))
            yield Tick(now=epoch0 + t, packets=pkts)
            t += 1.0
            lag = (t - self.t_start) / self.speed - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
