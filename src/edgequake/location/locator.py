"""Pick-based hypocenter location with uncertainty — Phase 2 core.

Method (deliberately interpretable, v1):
    unknowns   : epicenter (lat, lon), depth, origin time t0
    observations: P arrival times at k stations
    model      : t_i = t0 + dist3d(station_i, source) / vp   (homogeneous vp)
    solve      : coarse 3-D grid search (t0 eliminated analytically by
                 de-meaning residuals) -> local least-squares refinement
    uncertainty: station bootstrap -> epicenter covariance -> error ellipse

Known limitation (documented on purpose): a homogeneous velocity model is
crude for deep events; residual bias absorbs into depth/origin-time trade-off.
Good enough to demonstrate convergence behaviour; a 1-D Taiwan model is the
natural upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_EARTH = 6371.0
VP_DEFAULT = 6.2  # km/s, crustal average for Taiwan-scale distances


def geo_to_xy(lat, lon, lat0, lon0):
    """Local flat-earth km coordinates around (lat0, lon0)."""
    x = np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return x, y


def xy_to_geo(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / R_EARTH)
    lon = lon0 + np.degrees(np.asarray(x) / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


@dataclass
class LocationEstimate:
    lat: float
    lon: float
    depth_km: float
    origin_time_s: float           # relative to the time reference of arrivals
    rms_s: float
    n_stations: int
    # bootstrap uncertainty (epicenter)
    ellipse_major_km: float | None = None
    ellipse_minor_km: float | None = None
    ellipse_azimuth_deg: float | None = None
    depth_std_km: float | None = None
    bootstrap_lats: np.ndarray | None = field(default=None, repr=False)
    bootstrap_lons: np.ndarray | None = field(default=None, repr=False)

    @property
    def ellipse_area_km2(self) -> float | None:
        if self.ellipse_major_km is None:
            return None
        return float(np.pi * self.ellipse_major_km * self.ellipse_minor_km)


class PickLocator:
    def __init__(self, vp_km_s: float = VP_DEFAULT,
                 depth_grid=None, horiz_extent_km: float = 150.0,
                 coarse_step_km: float = 10.0):
        self.vp = vp_km_s
        self.depth_grid = np.array(depth_grid if depth_grid is not None
                                   else [5, 10, 20, 30, 50, 75, 100, 130, 170])
        self.horiz_extent = horiz_extent_km
        self.coarse_step = coarse_step_km

    VP_VS_RATIO = 1.73

    # ---- forward model -------------------------------------------------
    def _tt(self, sx, sy, sz, x, y, slow):
        """Travel time from source to stations; slow = per-observation
        slowness factor (1 for P, VP_VS_RATIO for S)."""
        return np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + sz ** 2) * slow / self.vp

    def _misfit(self, sx, sy, sz, x, y, t, slow):
        tt = self._tt(sx, sy, sz, x, y, slow)
        r = t - tt
        r = r - r.mean()
        return float(np.sqrt((r ** 2).mean()))

    # ---- solve ---------------------------------------------------------
    def locate(self, st_lat, st_lon, t_p, t_s=None, bootstrap: int = 100,
               seed: int = 0) -> LocationEstimate:
        """st_lat/st_lon: station coords (deg); t_p: P arrival times (s,
        any common reference). Needs >= 3 stations."""
        st_lat = np.asarray(st_lat, float)
        st_lon = np.asarray(st_lon, float)
        t_p = np.asarray(t_p, float)
        n = len(t_p)
        if n < 3:
            raise ValueError("need >= 3 P arrivals")

        # anchor local frame on the earliest station (closest to epicenter)
        i0 = int(np.argmin(t_p))
        lat0, lon0 = float(st_lat[i0]), float(st_lon[i0])
        x, y = geo_to_xy(st_lat, st_lon, lat0, lon0)

        # observation vectors: P legs, plus optional S legs (S-P time is the
        # classic distance constraint that stabilizes depth/distance when only
        # a few nearly-simultaneous P arrivals are available)
        ox, oy, ot = [x], [y], [t_p]
        oslow = [np.ones(n)]
        if t_s is not None:
            t_s = np.asarray(t_s, float)
            has_s = np.isfinite(t_s)
            if has_s.any():
                ox.append(x[has_s]); oy.append(y[has_s]); ot.append(t_s[has_s])
                oslow.append(np.full(int(has_s.sum()), self.VP_VS_RATIO))
        X = np.concatenate(ox); Y = np.concatenate(oy)
        T = np.concatenate(ot); SLOW = np.concatenate(oslow)

        sx, sy, sz = self._solve_xy(X, Y, T, SLOW)
        sx, sy, sz = self._refine(sx, sy, sz, X, Y, T, SLOW)
        rms = self._misfit(sx, sy, sz, X, Y, T, SLOW)
        tt = self._tt(sx, sy, sz, X, Y, SLOW)
        # origin time from P legs only — P picks are sharper, and S legs
        # inherit the vp/vs-ratio assumption error
        p_leg = SLOW == 1.0
        t0 = float((T - tt)[p_leg].mean())
        lat, lon = xy_to_geo(sx, sy, lat0, lon0)

        est = LocationEstimate(lat=float(lat), lon=float(lon),
                               depth_km=float(sz), origin_time_s=t0,
                               rms_s=rms, n_stations=n)

        if bootstrap and n >= 4:
            rng = np.random.default_rng(seed)
            m = len(T)
            bx, by, bz = [], [], []
            for _ in range(bootstrap):
                idx = rng.integers(0, m, m)
                if len(np.unique(idx)) < 3:
                    continue
                gx, gy, gz = self._solve_xy(X[idx], Y[idx], T[idx], SLOW[idx])
                gx, gy, gz = self._refine(gx, gy, gz, X[idx], Y[idx], T[idx], SLOW[idx])
                bx.append(gx); by.append(gy); bz.append(gz)
            bx, by, bz = map(np.array, (bx, by, bz))
            if len(bx) >= 10:
                cov = np.cov(np.vstack([bx, by]))
                evals, evecs = np.linalg.eigh(cov)
                # 1-sigma ellipse axes (km)
                est.ellipse_minor_km = float(np.sqrt(max(evals[0], 0)))
                est.ellipse_major_km = float(np.sqrt(max(evals[1], 0)))
                est.ellipse_azimuth_deg = float(
                    np.degrees(np.arctan2(evecs[0, 1], evecs[1, 1])) % 180)
                est.depth_std_km = float(bz.std())
                blat, blon = xy_to_geo(bx, by, lat0, lon0)
                est.bootstrap_lats, est.bootstrap_lons = blat, blon
        return est

    def _solve_xy(self, x, y, t, slow):
        """Coarse grid search, fully vectorized over the 3-D grid."""
        ext, step = self.horiz_extent, self.coarse_step
        g = np.arange(-ext, ext + step, step)
        GX, GY, GZ = np.meshgrid(g, g, self.depth_grid, indexing="ij")
        pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)  # (G,3)
        dx = pts[:, 0:1] - x[None, :]
        dy = pts[:, 1:2] - y[None, :]
        tt = np.sqrt(dx ** 2 + dy ** 2 + pts[:, 2:3] ** 2) * slow[None, :] / self.vp
        r = t[None, :] - tt
        r = r - r.mean(axis=1, keepdims=True)
        rms = np.sqrt((r ** 2).mean(axis=1))
        i = int(np.argmin(rms))
        return float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])

    def _refine(self, sx, sy, sz, x, y, t, slow):
        """Bounded local refinement. Bounds are essential: with origin time
        eliminated, only arrival DIFFERENCES constrain the source — for deep
        events all stations trigger nearly simultaneously, and an unbounded
        solver runs away to infinity where all travel times are equal
        (verified failure: 8,000+ km epicenter errors with method='lm')."""
        from scipy.optimize import least_squares

        ext = self.horiz_extent * 2
        z_lo, z_hi = 0.5, float(self.depth_grid[-1]) * 1.2

        def resid(p):
            r = t - self._tt(p[0], p[1], p[2], x, y, slow)
            return r - r.mean()

        try:
            x0 = [np.clip(sx, -ext + 1, ext - 1),
                  np.clip(sy, -ext + 1, ext - 1),
                  np.clip(sz, z_lo + 0.1, z_hi - 0.1)]
            sol = least_squares(resid, x0=x0, method="trf",
                                bounds=([-ext, -ext, z_lo], [ext, ext, z_hi]),
                                max_nfev=200)
            return float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
        except Exception:
            return sx, sy, sz


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) — for epicenter error reporting."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R_EARTH * np.arcsin(np.sqrt(a)))
