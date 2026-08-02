"""Time-stepped replay simulation shared by the dashboard builder (Phase 3).

Runs the anytime estimator over a replay event at fixed time steps and
returns JSON-serializable frames: location + uncertainty + magnitude +
per-county predicted intensity and PWS alert decisions.
"""
from __future__ import annotations

import numpy as np

from .locator import PickLocator, haversine_km
from .magnitude import DEFAULT_COEF, PgaMagnitude

# Single source of truth for replay/audit run parameters. Dashboard,
# results summary and the CWA-waveform audit all import this — external
# review found result figures drifting because different call sites used
# different max_stations/bootstrap. Point estimates are bootstrap-count
# invariant (verified for N in 20..200); N=60 keeps the Monte-Carlo std
# of the bootstrap confidence fraction under ~6% (binomial, p~0.2).
CANONICAL = {"max_stations": 60, "bootstrap": 60, "seed": 0,
             "vp_km_s": 6.2, "dt_s": 0.25, "pick_threshold": 0.3,
             # same crustal grid + hard ceiling as the live engine, so the
             # offline replay and the live path share one depth policy
             "depth_grid_km": [5, 10, 15, 20, 30, 40, 60, 80],
             "max_depth_km": 80.0}

# Alert quality gates, single-sourced (round-5 review: the replay's PWS
# decision used only the magnitude/intensity thresholds while the live
# engine additionally required station count, ellipse size and observed
# shaking). The live engine imports THESE values; the replay applies the
# same numbers — but see the PGA-timing approximation note in simulate().
GATES = {"min_stations": 6, "max_ellipse_km": 80.0,
         "eew_min_obs_gal": 10.0, "pws_min_obs_gal": 25.0}

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


def simulate(ev, truth, vp=None, dt=None, max_stations=None, bootstrap=None,
             site_terms=None, exposure_model=None, similar_db=None):
    """ev: DataFrame from load_replay_json/load_event; returns dict payload.

    All run parameters default to CANONICAL — call sites may override for
    experiments, but the dashboard/audit/summary paths pass nothing and
    therefore cannot drift (external review round 3).
    """
    vp = CANONICAL["vp_km_s"] if vp is None else vp
    dt = CANONICAL["dt_s"] if dt is None else dt
    max_stations = (CANONICAL["max_stations"] if max_stations is None
                    else max_stations)
    bootstrap = CANONICAL["bootstrap"] if bootstrap is None else bootstrap
    n_max = min(max_stations, len(ev))
    t_ref = float(ev.t_p.values.min())
    t_p = ev.t_p.values[:n_max] - t_ref
    t_s = ev.t_s.values[:n_max] - t_ref
    lats = ev.station_latitude_deg.values[:n_max]
    lons = ev.station_longitude_deg.values[:n_max]
    pga = ev.station_pga.values[:n_max]
    # Phase 7: site-corrected PGA for MAGNITUDE inversion only (raw PGA
    # still drives intensity display / EEW-gate shaking checks)
    if site_terms:
        corr = np.array([10.0 ** site_terms.get(str(c), 0.0)
                         for c in ev.station_code.values[:n_max]])
        pga_m = pga / corr
    else:
        pga_m = pga
    t_end = float(t_p[-1]) + 3.0

    locator = PickLocator(vp_km_s=vp,
                          depth_grid=CANONICAL["depth_grid_km"],
                          max_depth_km=CANONICAL["max_depth_km"])
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
                                 bootstrap=bootstrap,
                                 seed=CANONICAL["seed"])
            last_k = k
        frame = {"t": round(float(now), 2), "k": k}
        if est is not None:
            mag = None
            m_ok = np.isfinite(t_s[:k]) & (t_s[:k] + 2.0 <= now)
            if m_ok.any():
                d = np.array([haversine_km(est.lat, est.lon, la, lo)
                              for la, lo in zip(lats[:k][m_ok], lons[:k][m_ok])])
                mag = magest.estimate(pga_m[:k][m_ok], d, est.depth_km)
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
                if exposure_model is not None:
                    e = exposure_model.exposure(est.lat, est.lon,
                                                est.depth_km, mag.mag)
                    frame["exp"] = {k: e[k] for k in ("i3", "i4", "i5")}
                # predicted-intensity contour radii from the attenuation model
                a_, b_, c_ = (DEFAULT_COEF["a"], DEFAULT_COEF["b"],
                              DEFAULT_COEF["c"])
                for key_r, pga_th in (("r4", 25.0), ("r5", 80.0)):
                    r_hyp = 10 ** ((a_ * mag.mag + c_ - np.log10(pga_th)) / (-b_))
                    r_ep2 = r_hyp ** 2 - est.depth_km ** 2
                    if r_ep2 > 0:
                        frame[key_r] = round(float(np.sqrt(r_ep2)), 1)
            # counties: predicted intensity + PWS decision + S-wave ETA.
            # Both alert tiers apply the SAME numeric quality gates as
            # the live engine (GATES). APPROXIMATION (round-5 review):
            # the replay artifact stores each station's record-peak PGA
            # without its timing, so "observed shaking" counts a
            # station's peak once its S-window has passed (t_s+2s <=
            # now); the live engine uses causally-observed PGA up to
            # `now`, and a peak can arrive after S+2s. Gate NUMBERS are
            # shared; gate TIMING is approximate — full causal parity is
            # the LiveEngine-feed audit on the roadmap.
            obs = (float(np.nanmax(pga[:k][m_ok])) if m_ok.any() else 0.0)
            gate_ok = (k >= GATES["min_stations"]
                       and (est.ellipse_major_km or 999)
                       <= GATES["max_ellipse_km"])
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
                    entry.update({"i": i, "alert": bool(
                        pws_alert(mag.mag, i) and gate_ok
                        and obs >= GATES["pws_min_obs_gal"])})
                cty.append(entry)
            frame["cty"] = cty
            # EEW tier (CWA 強震即時警報 issuance rule): M>=4.5 and
            # predicted intensity >=3 somewhere, plus the shared gates
            if mag:
                max_i = max((c.get("i", 0) for c in cty), default=0)
                frame["eew"] = bool(
                    mag.mag >= 4.5 and max_i >= 3 and gate_ok
                    and obs >= GATES["eew_min_obs_gal"])
        frames.append(frame)

    # Phase 10: similar historical events from the FINAL estimate (the
    # event's own catalog row is excluded by origin date)
    similar = None
    if similar_db is not None and est is not None:
        last = [f for f in frames if f.get("mag") is not None]
        if last:
            similar = similar_db.find(
                last[-1]["lat"], last[-1]["lon"], last[-1]["depth"],
                last[-1]["mag"], k=3,
                exclude_date=str(truth.get("origin_time", ""))[:10] or None)

    return {
        "truth": {k: truth[k] for k in ("lat", "lon", "depth_km", "mag",
                                        "origin_time") if k in truth},
        "similar": similar,
        "origin_rel": (round(truth["origin_epoch"] - t_ref, 2)
                       if truth.get("origin_epoch") else None),
        "source": truth.get("source", "catalog picks"),
        "pop_version": (exposure_model.version
                        if exposure_model is not None else None),
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
