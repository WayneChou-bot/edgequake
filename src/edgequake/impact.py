"""Phase 9: PAGER-style population-exposure estimates.

Given a location + magnitude, predict PGA on the WorldPop 1 km Taiwan grid
(assets/tw_pop_1km.npz, built by scripts/fetch_pop_grid.py) and sum the
population inside each predicted-intensity band — turning "M6.1 at depth
18 km" into "about 950k people in shaking of intensity 4+".

Honest limits (stated wherever the numbers are shown):
  - point-source GMPE, average site (no per-cell amplification),
  - residential (static) population — no day/night difference,
  - same attenuation coefficients as the magnitude chain, so exposure
    inherits their uncertainty. These are order-of-magnitude numbers.

The whole grid is ~440k cells; one exposure evaluation is a few ms of
vectorized numpy — cheap enough to recompute on every magnitude update in
the live engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .location.magnitude import DEFAULT_COEF

# CWA intensity lower-bound PGA (cm/s^2) — keep in sync with replay_sim
_BANDS = [(3, 8.0), (4, 25.0), (5, 80.0)]   # reported bands: I>=3/4/5
KM_PER_DEG = 111.19


class ExposureModel:
    def __init__(self, asset: str | Path):
        d = np.load(asset)
        self.pop = d["pop"].astype(np.float32)
        lat0, lon0 = float(d["lat0"]), float(d["lon0"])
        dlat, dlon = float(d["dlat"]), float(d["dlon"])
        ny, nx = self.pop.shape
        self.lats = lat0 - np.arange(ny, dtype=np.float32) * dlat  # row centers
        self.lons = lon0 + np.arange(nx, dtype=np.float32) * dlon
        self.meta = json.loads(str(d["meta"]))
        rel = self.meta.get("release")
        self.version = (f"WorldPop {self.meta.get('popyear', '?')}" +
                        (f" {rel}" if rel else ""))
        # precompute row-wise km scaling (equirectangular is fine at 1 km)
        self._ky = (self.lats[:, None] * 0)  # placeholder shape helper
        self._coslat = np.cos(np.radians(self.lats))[:, None]
        # drop empty cells once: keeps the hot loop at ~n_inhabited cells
        mask = self.pop > 0
        self._plat = np.repeat(self.lats, nx).reshape(ny, nx)[mask]
        self._plon = np.tile(self.lons, ny).reshape(ny, nx)[mask]
        self._pcos = np.cos(np.radians(self._plat))
        self._ppop = self.pop[mask]

    def exposure(self, lat: float, lon: float, depth_km: float,
                 mag: float, coef=DEFAULT_COEF) -> dict:
        dx = (self._plon - lon) * KM_PER_DEG * self._pcos
        dy = (self._plat - lat) * KM_PER_DEG
        r = np.sqrt(dx * dx + dy * dy + depth_km * depth_km)
        np.maximum(r, 1.0, out=r)
        log_pga = (coef["a"] * mag + coef["b"] * np.log10(r) + coef["c"])
        out = {}
        for band, pga_lo in _BANDS:
            th = np.log10(pga_lo)
            out[f"i{band}"] = int(self._ppop[log_pga >= th].sum())
        out["pop_version"] = self.version
        return out


_model = None


def get_model(root: str | Path | None = None):
    """Lazy singleton; returns None if the asset is missing (feature off)."""
    global _model
    if _model is None:
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        asset = root / "assets" / "tw_pop_1km.npz"
        if not asset.exists():
            return None
        _model = ExposureModel(asset)
    return _model


def fmt_pop(n: int, lang: str = "zh") -> str:
    """96,300 -> '9.6萬' / '96k' — display helper mirrored in the JS."""
    if lang == "zh":
        if n >= 1e8:
            return f"{n/1e8:.1f}億"
        if n >= 1e4:
            return f"{n/1e4:.0f}萬" if n >= 1e5 else f"{n/1e4:.1f}萬"
        return str(int(n))
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return str(int(n))
