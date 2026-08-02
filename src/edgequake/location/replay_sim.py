"""Time-stepped replay simulation shared by the dashboard builder (Phase 3).

Runs the anytime estimator over a replay event at fixed time steps and
returns JSON-serializable frames: location + uncertainty + magnitude +
per-county predicted intensity and PWS alert decisions.
"""
from __future__ import annotations

import numpy as np

from .locator import PickLocator, haversine_km
from .magnitude import DEFAULT_COEF, PgaMagnitude

# approximate county reference points (city halls / centroids)
COUNTIES = [
    ("Taipei", 25.04, 121.56), ("New Taipei", 25.01, 121.46),
    ("Keelung", 25.13, 121.74), ("Taoyuan", 24.99, 121.30),
    ("Hsinchu C.", 24.81, 120.97), ("Hsinchu Co.", 24.84, 121.01),
    ("Miaoli", 24.56, 120.82), ("Taichung", 24.15, 120.67),
    ("Changhua", 24.05, 120.52), ("Nantou", 23.96, 120.97),
    ("Yunlin", 23.71, 120.43), ("Chiayi C.", 23.48, 120.45),
    ("Chiayi Co.", 23.45, 120.25), ("Tainan", 23.00, 120.23),
    ("Kaohsiung", 22.63, 120.30), ("Pingtung", 22.55, 120.55),
    ("Yilan", 24.75, 121.75), ("Hualien", 23.99, 121.60),
    ("Taitung", 22.75, 121.15), ("Penghu", 23.57, 119.58),
]

# CWA intensity scale approximation from PGA (cm/s^2): lower bounds
INTENSITY_PGA = [(0.8, 1), (2.5, 2), (8.0, 3), (25.0, 4),
                 (80.0, 5), (140.0, 5.5), (250.0, 6), (440.0, 6.5), (800.0, 7)]


def pga_to_intensity(pga: float) -> float:
    level = 0
    for lo, lv in INTENSITY_PGA:
        if pga >= lo:
            level = lv
    return level


def predict_pga(mag: float, dist_ep_km: float, depth_km: float,
                coef=DEFAULT_COEF) -> float:
    r = np.sqrt(dist_ep_km ** 2 + depth_km ** 2)
    return float(10 ** (coef["a"] * mag + coef["b"] * np.log10(max(r, 1)) + coef["c"]))


def pws_alert(mag: float, intensity: float) -> bool:
    """CWA public-alert thresholds: (M>=5.0 & I>=4) or (M>=6.5 & I>=3)."""
    return (mag >= 5.0 and intensity >= 4) or (mag >= 6.5 and intensity >= 3)


def simulate(ev, truth, vp=6.2, dt=0.25, max_stations=60, bootstrap=60):
    """ev: DataFrame from load_replay_json/load_event; returns dict payload."""
    n_max = min(max_stations, len(ev))
    t_ref = float(ev.t_p.values.min())
    t_p = ev.t_p.values[:n_max] - t_ref
    t_s = ev.t_s.values[:n_max] - t_ref
    lats = ev.station_latitude_deg.values[:n_max]
    lons = ev.station_longitude_deg.values[:n_max]
    pga = ev.station_pga.values[:n_max]
    t_end = float(t_p[-1]) + 3.0

    locator = PickLocator(vp_km_s=vp)
    magest = PgaMagnitude()
    vs = vp / 1.73

    frames = []
    est, last_k = None, 0
    eta_prev: dict[str, float] = {}
    for now in np.arange(0.0, t_end + 1e-9, dt):
        k = int((t_p <= now).sum())
        if k >= 3 and k != last_k:
            t_s_avail = np.where(t_s[:k] <= now, t_s[:k], np.nan)
            est = locator.locate(lats[:k], lons[:k], t_p[:k], t_s=t_s_avail,
                                 bootstrap=bootstrap)
            last_k = k
        frame = {"t": round(float(now), 2), "k": k}
        if est is not None:
            mag = None
            m_ok = np.isfinite(t_s[:k]) & (t_s[:k] + 2.0 <= now)
            if m_ok.any():
                d = np.array([haversine_km(est.lat, est.lon, la, lo)
                              for la, lo in zip(lats[:k][m_ok], lons[:k][m_ok])])
                mag = magest.estimate(pga[:k][m_ok], d, est.depth_km)
            frame.update({
                "lat": round(est.lat, 4), "lon": round(est.lon, 4),
                "depth": round(est.depth_km, 1),
                "t0": round(est.origin_time_s, 2),
                "emaj": round(est.ellipse_major_km or 0, 1),
                "emin": round(est.ellipse_minor_km or 0, 1),
                "eaz": round(est.ellipse_azimuth_deg or 0, 1),
                "err": round(haversine_km(est.lat, est.lon,
                                          truth["lat"], truth["lon"]), 1),
            })
            # honest epicenter-confidence %: fraction of bootstrap
            # solutions within 20 km of the estimate
            if getattr(est, "bootstrap_lats", None) is not None:
                bd = np.array([haversine_km(bl, bo, est.lat, est.lon)
                               for bl, bo in zip(est.bootstrap_lats,
                                                 est.bootstrap_lons)])
                frame["bconf"] = int(round(float((bd <= 20.0).mean()) * 100))
            if mag:
                frame["mag"] = round(mag.mag, 2)
                frame["msig"] = round(mag.sigma, 2)
                # predicted-intensity contour radii from the attenuation model
                a_, b_, c_ = (DEFAULT_COEF["a"], DEFAULT_COEF["b"],
                              DEFAULT_COEF["c"])
                for key_r, pga_th in (("r4", 25.0), ("r5", 80.0)):
                    r_hyp = 10 ** ((a_ * mag.mag + c_ - np.log10(pga_th)) / (-b_))
                    r_ep2 = r_hyp ** 2 - est.depth_km ** 2
                    if r_ep2 > 0:
                        frame[key_r] = round(float(np.sqrt(r_ep2)), 1)
            # counties: predicted intensity + PWS decision + S-wave ETA
            cty = []
            for name, cla, clo in COUNTIES:
                d_ep = haversine_km(est.lat, est.lon, cla, clo)
                r_hyp = float(np.hypot(d_ep, est.depth_km))
                eta = est.origin_time_s + r_hyp / vs - now
                prev = eta_prev.get(name)
                if prev is not None:
                    eta = min(eta, prev - dt)
                eta_prev[name] = eta
                entry = {"eta": round(float(eta), 1)}
                if mag:
                    p = predict_pga(mag.mag, d_ep, est.depth_km)
                    i = pga_to_intensity(p)
                    entry.update({"i": i, "alert": pws_alert(mag.mag, i)})
                cty.append(entry)
            frame["cty"] = cty
        frames.append(frame)

    return {
        "truth": {k: truth[k] for k in ("lat", "lon", "depth_km", "mag",
                                        "origin_time") if k in truth},
        "origin_rel": (round(truth["origin_epoch"] - t_ref, 2)
                       if truth.get("origin_epoch") else None),
        "source": truth.get("source", "catalog picks"),
        "vp": vp, "dt": dt, "t_end": round(t_end, 2),
        "stations": [{"code": str(c), "lat": round(float(la), 4),
                      "lon": round(float(lo), 4), "tp": round(float(tp), 2),
                      "ts": (round(float(ts), 2) if np.isfinite(ts) else None),
                      "pga": (round(float(pg), 1) if np.isfinite(pg) else None),
                      "pprob": (round(float(pp), 2) if np.isfinite(pp) else None)}
                     for c, la, lo, tp, ts, pg, pp in zip(
                         ev.station_code.values[:n_max], lats, lons, t_p, t_s,
                         pga,
                         (ev.p_prob.values[:n_max] if "p_prob" in ev
                          else np.full(n_max, np.nan)))],
        "counties": [{"name": n, "lat": la, "lon": lo} for n, la, lo in COUNTIES],
        "frames": frames,
    }
