"""Streaming EEW engine: buffers -> picks -> association -> location/magnitude.

Consumes Ticks from any source (replay or live), maintains per-station ring
buffers on a fixed time grid, runs the fine-tuned PhaseNet every `stride_s`
of stream time, associates picks into an event, and produces a JSON-ready
state dict for the web console.

This is the same physics chain as the replay console (PickLocator +
PgaMagnitude + CWA county intensities/PWS rules) — only the plumbing is
streaming instead of precomputed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..location.locator import PickLocator, haversine_km
from ..location.magnitude import DEFAULT_COEF, PgaMagnitude
from ..location.replay_sim import (COUNTIES, pga_to_intensity, predict_pga,
                                   pws_alert)

WIN = 3001
FS = 100.0


@dataclass
class LiveStation:
    code: str
    lat: float
    lon: float
    pick_channel: str | None
    buf: dict = field(default_factory=dict)      # comp -> np.ndarray (grid)
    acc_peaks: list = field(default_factory=list)  # (t_end, max_abs cm/s^2)
    tp: float | None = None
    pprob: float | None = None
    ts: float | None = None
    sprob: float | None = None
    ai_mu: float | None = None     # MagNet early-magnitude (per station)
    ai_sig: float | None = None
    last_data: float = 0.0

    def pga_since(self, t: float) -> float | None:
        vals = [a for te, a in self.acc_peaks if te >= t]
        return max(vals) if vals else None


class LiveEngine:
    def __init__(self, picker, stations_meta, buf_s=120.0, stride_s=1.0,
                 threshold=0.3, assoc_n=3, assoc_win_s=15.0,
                 event_timeout_s=60.0, mode_label="replay", notifier=None,
                 trigger_mode=False, trig_pga=2.5):
        # trigger_mode: sources that deliver per-station PGA values instead
        # of waveforms (e.g. TREM RTS). No PhaseNet, no S picks — a PGA
        # jump IS the arrival; magnitude waits a fixed dwell after trigger.
        self.trigger_mode = trigger_mode
        self.trig_pga = trig_pga
        self.notifier = notifier
        self.magnet = None          # set via load_magnet() (waveform modes)
        self.picker = picker
        self.thr = threshold
        self.buf_n = int(buf_s * FS)
        self.stride_s = stride_s
        self.assoc_n = assoc_n
        self.assoc_win_s = assoc_win_s
        self.event_timeout_s = event_timeout_s
        self.mode_label = mode_label
        self.stations = {m.code: LiveStation(m.code, m.lat, m.lon,
                                             m.pick_channel)
                         for m in stations_meta.values()}
        # crustal depth grid + HARD 80 km ceiling: an unconstrained deep
        # solution absorbs systematically-late mispicks (S picked as P)
        # with tiny residuals. Taiwan EEW targets are crustal.
        self.locator = PickLocator(depth_grid=[5, 10, 15, 20, 30, 40,
                                               60, 80],
                                   max_depth_km=80.0)
        self.magest = PgaMagnitude()
        self.now = None
        self._last_infer = None
        self.event = None            # dict while active
        self._quiet_until = None     # refractory: no new event declarations
        self._eta_prev: dict[str, float] = {}
        self.log: list = []
        self.n_infer = 0
        self.infer_ms = 0.0

    # ------------------------------------------------------------- ingest
    def on_tick(self, tick):
        prev = self.now
        self.now = tick.now
        shift = int(round((self.now - prev) * FS)) if prev else 0
        if shift:
            for st in self.stations.values():
                for comp, arr in st.buf.items():
                    if shift >= self.buf_n:
                        arr[:] = 0
                    else:
                        arr[:-shift] = arr[shift:]
                        arr[-shift:] = 0
        for p in tick.packets:
            st = self.stations.get(p.code)
            if st is None:
                continue
            if p.kind == "trig":
                pga_now = float(p.data[0])
                st.acc_peaks.append((p.t0, pga_now))
                if len(st.acc_peaks) > 400:
                    st.acc_peaks = st.acc_peaks[-400:]
                st.last_data = self.now
                if st.tp is None and pga_now >= self.trig_pga:
                    st.tp = p.t0
                    st.pprob = 0.9 if pga_now >= 8.0 else 0.6
                    self._log(f"trigger {st.code} (PGA {pga_now:.1f} gal)")
                continue
            if p.kind == "acc":
                if len(p.data):
                    st.acc_peaks.append((p.t0 + len(p.data) / p.fs,
                                         float(np.abs(p.data).max())))
                    if len(st.acc_peaks) > 400:
                        st.acc_peaks = st.acc_peaks[-400:]
                continue
            if p.comp not in st.buf:
                st.buf[p.comp] = np.zeros(self.buf_n, dtype=np.float32)
            arr = st.buf[p.comp]
            i0 = self.buf_n - int(round((self.now - p.t0) * FS))
            j0, j1 = max(i0, 0), min(i0 + len(p.data), self.buf_n)
            if j1 > j0:
                arr[j0:j1] = p.data[j0 - i0:j1 - i0]
                st.last_data = self.now

        if self._last_infer is None or \
                self.now - self._last_infer >= self.stride_s - 1e-6:
            self._last_infer = self.now
            self._infer()
            self._associate()
            if self.magnet is not None:
                self._ai_magnitude()
            self._expire()

    def load_magnet(self, path) -> bool:
        """Load the distance-conditioned MagNet (v2) for AI early magnitude."""
        try:
            import torch

            from ..models.magnet import MagNet

            m = MagNet(n_aux=3)
            m.load_state_dict(torch.load(str(path), map_location="cpu"))
            m.eval()
            self.magnet = m
            return True
        except Exception as e:
            print(f"[engine] magnet not loaded ({e})")
            return False

    def _ai_magnitude(self):
        """Per-station MagNet on the P+3s window, once available; then
        inverse-variance aggregate with an event-correlation floor (the
        blind-test lesson: station errors are correlated, never let the
        aggregated sigma pretend otherwise)."""
        import torch

        prev = self.event.get("_est") if self.event else None
        for st in self.stations.values():
            if (st.tp is None or st.ai_mu is not None or len(st.buf) < 3
                    or self.now < st.tp + 3.2):
                continue
            i0 = self.buf_n - int(round((self.now - (st.tp - 1.0)) * FS))
            if i0 < 0 or i0 + 400 > self.buf_n:
                continue
            seg = np.stack([st.buf[c][i0:i0 + 400] for c in ("Z", "N", "E")
                            if c in st.buf]).astype(np.float64)
            if seg.shape[0] != 3:
                continue
            peak, std = float(np.abs(seg).max()), float(seg.std())
            if peak <= 0 or std <= 0 or peak > 1e8:
                continue
            if prev is not None:
                d = float(np.hypot(haversine_km(prev.lat, prev.lon,
                                                st.lat, st.lon),
                                   prev.depth_km))
            else:
                d = 40.0   # trained fallback prior
            x = torch.tensor((seg / peak)[None].astype(np.float32))
            aux = torch.tensor([[np.log10(peak), np.log10(std),
                                 np.log10(max(d, 1.0))]],
                               dtype=torch.float32)
            mu, sig = self.magnet.estimate(x, aux)
            st.ai_mu, st.ai_sig = float(mu[0]), float(sig[0])
        if self.event is not None:
            mus = [(s.ai_mu, s.ai_sig) for s in self.stations.values()
                   if s.ai_mu is not None and s.code in self.event["members"]]
            if mus:
                w = np.array([1.0 / s ** 2 for _, s in mus])
                m = np.array([m_ for m_, _ in mus])
                self.event["ai_mag"] = round(float((w * m).sum() / w.sum()), 2)
                self.event["ai_sig"] = round(
                    float(np.sqrt(1.0 / w.sum() + 0.3 ** 2)), 2)
                self.event["n_ai"] = len(mus)

    # ------------------------------------------------------------ picking
    def _infer(self):
        t0 = time.perf_counter()
        for st in self.stations.values():
            if len(st.buf) < 3 or self.now - st.last_data > 5.0:
                continue
            seg = np.stack([st.buf[c][-WIN:] for c in ("Z", "N", "E")
                            if c in st.buf])
            if seg.shape[0] != 3 or not np.any(seg):
                continue
            probs = self.picker.predict(seg).probs
            t_of = lambda i: self.now - (WIN - i) / FS
            if st.tp is None:
                i = int(np.argmax(probs[1]))
                if probs[1][i] >= self.thr:
                    st.tp, st.pprob = t_of(i), float(probs[1][i])
                    self._log(f"P pick {st.code} (conf {st.pprob:.2f})")
            elif st.ts is None:
                # only consider samples after tp + 0.5 s
                i_min = max(0, WIN - int((self.now - (st.tp + 0.5)) * FS))
                if i_min < WIN:
                    i = i_min + int(np.argmax(probs[2][i_min:]))
                    if probs[2][i] >= self.thr:
                        st.ts, st.sprob = t_of(i), float(probs[2][i])
        self.n_infer += 1
        self.infer_ms = (time.perf_counter() - t0) * 1e3

    # -------------------------------------------------------- association
    def _associate(self):
        picked = sorted([s for s in self.stations.values()
                         if s.tp is not None], key=lambda s: s.tp)
        if self.event is None:
            # refractory window: the coda of a closed event keeps producing
            # picks — do not declare a "new" event from them
            if self._quiet_until and self.now < self._quiet_until:
                return
            for i, s in enumerate(picked):
                grp = [x for x in picked[i:] if x.tp <= s.tp + self.assoc_win_s]
                # coda re-triggers are uniformly low-confidence: demand at
                # least two confident picks before declaring an event
                n_conf = sum(1 for x in grp if (x.pprob or 0) >= 0.5)
                if len(grp) >= self.assoc_n and n_conf >= 2:
                    self.event = {"t_first": s.tp, "declared_at": self.now,
                                  "last_pick": grp[-1].tp,
                                  "members": {x.code for x in grp}}
                    self._eta_prev = {}
                    self._log(f"EVENT declared: {len(grp)} stations within "
                              f"{self.assoc_win_s:.0f}s")
                    break
        if self.event is None:
            return
        ev = self.event
        t_ref = ev["t_first"]

        # --- incremental association: a new pick joins only if it matches
        # the current solution's predicted P arrival (±3.5 s). Without this,
        # stations whose true P fell below threshold get their S/coda picked
        # as "P" later, and a free-depth solver happily absorbs those
        # systematically-late picks by drifting deep — with small residuals.
        prev = ev.get("_est")
        for s in picked:
            if s.code in ev["members"] or s.tp > t_ref + 60.0:
                continue
            if prev is None:
                if s.tp <= t_ref + self.assoc_win_s:
                    ev["members"].add(s.code)
                continue
            d_hyp = np.hypot(haversine_km(prev.lat, prev.lon, s.lat, s.lon),
                             prev.depth_km)
            pred = t_ref + prev.origin_time_s + d_hyp / self.locator.vp
            if abs(s.tp - pred) <= 3.5:
                ev["members"].add(s.code)

        members = [s for s in picked if s.code in ev["members"]]
        ev["last_pick"] = max(s.tp for s in members)
        if len(members) < 3:
            return
        lats = np.array([s.lat for s in members])
        lons = np.array([s.lon for s in members])
        t_p = np.array([s.tp - t_ref for s in members])
        # S legs are powerful but dangerous: a coda peak picked as "S" at
        # 1.73x slowness drags the solution deep. Gate each S leg against
        # the previous solution's predicted S arrival.
        def s_ok(s):
            if s.ts is None or s.ts > self.now:
                return False
            if prev is None:
                return (s.ts - s.tp) <= 25.0
            d = np.hypot(haversine_km(prev.lat, prev.lon, s.lat, s.lon),
                         prev.depth_km)
            pred_s = (t_ref + prev.origin_time_s +
                      d * self.locator.VP_VS_RATIO / self.locator.vp)
            return abs(s.ts - pred_s) <= 4.5
        t_s = np.array([(s.ts - t_ref) if s_ok(s) else np.nan
                        for s in members])
        est = self.locator.locate(lats, lons, t_p, t_s=t_s, bootstrap=40)
        # a deep solution (>90 km) that used S legs is almost always
        # S-mispick absorption (systematically-late legs at 1.73x slowness
        # fit best by burying the source). Taiwan EEW targets are crustal —
        # retry with P legs only before accepting a deep solution.
        if est.depth_km > 90 and np.isfinite(t_s).any():
            est_p = self.locator.locate(lats, lons, t_p, t_s=None,
                                        bootstrap=40)
            if est_p.depth_km < est.depth_km:
                est = est_p
        # prune members that stopped fitting (residual > 4 s) — takes effect
        # on the next cycle's relocation
        d_hyp = np.array([np.hypot(haversine_km(est.lat, est.lon, la, lo),
                                   est.depth_km)
                          for la, lo in zip(lats, lons)])
        res = t_p - (est.origin_time_s + d_hyp / self.locator.vp)
        for m, r in zip(members, res):
            if abs(r) > 4.0 and len(ev["members"]) > 4:
                ev["members"].discard(m.code)
        ev["_est"] = est
        ev.update(lat=est.lat, lon=est.lon, depth=est.depth_km,
                  t0=est.origin_time_s + t_ref, k=len(members),
                  emaj=est.ellipse_major_km, emin=est.ellipse_minor_km,
                  eaz=est.ellipse_azimuth_deg)

        mag = None
        m_pga, m_d = [], []
        for s in members:
            if self.trigger_mode:
                # no S picks: PGA is plausibly final a fixed dwell after
                # the trigger (S passes within ~8 s for near stations)
                if self.now < s.tp + 8.0:
                    continue
            elif s.ts is None or s.ts + 2.0 > self.now:
                continue
            pga = s.pga_since(s.tp - 1.0)
            if pga:
                m_pga.append(pga)
                m_d.append(haversine_km(est.lat, est.lon, s.lat, s.lon))
        if m_pga:
            mag = self.magest.estimate(np.array(m_pga), np.array(m_d),
                                       est.depth_km)
            new_mag = round(mag.mag, 1)
            if ev.get("mag") is None or abs(new_mag - ev["mag"]) >= 0.3:
                self._log(f"M {new_mag} ({len(m_pga)} PGA stations)")
            ev.update(mag=new_mag, msig=round(mag.sigma, 2),
                      n_mag=len(m_pga))
            a_, b_, c_ = DEFAULT_COEF["a"], DEFAULT_COEF["b"], DEFAULT_COEF["c"]
            for key_r, pga_th in (("r4", 25.0), ("r5", 80.0)):
                r_hyp = 10 ** ((a_ * mag.mag + c_ - np.log10(pga_th)) / (-b_))
                r2 = r_hyp ** 2 - est.depth_km ** 2
                ev[key_r] = round(float(np.sqrt(r2)), 1) if r2 > 0 else None

        # alert quality gate: never issue a public-alert flag from a weak
        # solution (few stations / huge uncertainty)
        q_ok = len(members) >= 6 and (est.ellipse_major_km or 999) <= 80

        vs = self.locator.vp / 1.73
        cty, any_alert = [], False
        for name, cla, clo in COUNTIES:
            d_ep = haversine_km(est.lat, est.lon, cla, clo)
            r_hyp = float(np.hypot(d_ep, est.depth_km))
            eta = ev["t0"] + r_hyp / vs - self.now
            prev = self._eta_prev.get(name)
            if prev is not None:
                eta = min(eta, prev - self.stride_s)
            self._eta_prev[name] = eta
            entry = {"name": name, "eta": round(eta, 1)}
            if mag:
                p = predict_pga(mag.mag, d_ep, est.depth_km)
                i = pga_to_intensity(p)
                al = pws_alert(mag.mag, i) and q_ok
                entry.update(i=i, alert=al)
                any_alert = any_alert or al
            cty.append(entry)
        if any_alert and not ev.get("alerted"):
            ev["alerted"] = True
            self._log("PWS ALERT criteria met")
            if self.notifier and self.notifier.channels:
                from .notify import format_alert
                ev_snap = dict(ev, cty=cty)
                subject, body = format_alert(ev_snap, self.mode_label)
                self.notifier.send(subject, body)   # non-blocking
                self._log(f"notification sent ({'+'.join(self.notifier.channels)})")
        ev["cty"] = cty
        ev["alert"] = any_alert

    def _expire(self):
        # close a finished event; drop stale picks that never associated
        if self.event and self.now - self.event["last_pick"] > \
                self.event_timeout_s:
            self._log("event closed (no new picks)")
            self.event = None
            self._quiet_until = self.now + 120.0
            for s in self.stations.values():
                s.tp = s.ts = s.pprob = s.sprob = None
        if self.event is None:
            for s in self.stations.values():
                if s.tp is not None and self.now - s.tp > 90.0:
                    s.tp = s.ts = s.pprob = s.sprob = None

    def _log(self, msg):
        self.log.append({"t": round(self.now, 1), "msg": msg})
        self.log = self.log[-40:]
        print(f"[engine] {msg}", flush=True)

    # -------------------------------------------------------------- state
    def state(self, wf_n=6, wf_s=60.0, wf_fs=10.0):
        ev = self.event
        stations = []
        t_ref = ev["t_first"] if ev else None
        for s in self.stations.values():
            stations.append({
                "code": s.code, "lat": s.lat, "lon": s.lon,
                "tp": round(s.tp - t_ref, 2) if (s.tp and t_ref) else None,
                "ts": round(s.ts - t_ref, 2) if (s.ts and t_ref) else None,
                "pprob": round(s.pprob, 2) if s.pprob else None,
                "pga": (round(s.pga_since(s.tp - 1.0), 1)
                        if (s.tp and s.pga_since(s.tp - 1.0)) else None),
                "fresh": bool(s.tp and self.now - s.tp <= 1.5),
            })
        event = None
        if ev and "lat" in ev:
            event = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in ev.items()
                     if k in ("lat", "lon", "depth", "mag", "msig", "k",
                              "emaj", "emin", "eaz", "r4", "r5", "alert",
                              "cty", "n_mag", "ai_mag", "ai_sig", "n_ai")}
            event["age"] = round(self.now - ev["t_first"], 1)
            event["t0_age"] = round(self.now - ev["t0"], 1)

        # waveform strips: earliest picked stations (else quiet placeholder)
        picked = sorted([s for s in self.stations.values() if s.tp],
                        key=lambda s: s.tp)[:wf_n]
        if len(picked) < wf_n:
            extra = [s for s in self.stations.values()
                     if s.tp is None and "Z" in s.buf][:wf_n - len(picked)]
            picked += extra
        dec = int(FS / wf_fs)
        n_keep = int(wf_s * wf_fs)
        strips = []
        for s in picked:
            if "Z" not in s.buf:
                continue
            z = s.buf["Z"][-int(wf_s * FS):]
            z = z[:len(z) - len(z) % dec].reshape(-1, dec).mean(axis=1)
            peak = max(np.abs(z).max(), 1e-9)
            strips.append({
                "code": s.code, "ch": s.pick_channel or "--",
                "z": np.round(z[-n_keep:] / peak, 3).tolist(),
                "tp": round(s.tp - (self.now - wf_s), 2) if s.tp else None,
                "ts": round(s.ts - (self.now - wf_s), 2) if s.ts else None,
                "pprob": round(s.pprob, 2) if s.pprob else None,
            })

        return {
            "mode": self.mode_label,
            "updated_wall": time.time(),
            "now": round(self.now, 2) if self.now else None,
            "n_stations": len(self.stations),
            "n_live": sum(1 for s in self.stations.values()
                          if self.now - s.last_data < 5.0),
            "infer_ms": round(self.infer_ms, 1),
            "stations": stations,
            "event": event,
            "counties": [{"name": n, "lat": la, "lon": lo}
                         for n, la, lo in COUNTIES],
            "log": self.log[::-1],
            "wf": {"fs": wf_fs, "span": wf_s, "strips": strips},
        }

    def write_state(self, path):
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state(), f, separators=(",", ":"))
        os.replace(tmp, path)
