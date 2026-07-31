"""Attenuation-based magnitude estimation (Phase 2).

Inverts a simple PGA attenuation relation fitted on the CWA 2019 training
year (protocol-clean: coefficients never saw the 2020-2021 test years):

    log10(PGA) = a*M + b*log10(R_hyp) + c
    =>  M_i = (log10(PGA_i) - b*log10(R_i) - c) / a          per station
    M_event = median(M_i),  spread = 1.4826*MAD (robust sigma)

Out-of-sample accuracy (2020-2021, >=5 stations): MAE 0.21 (M4-5),
0.23 (M5-6). Known limitation: PGA saturates for M6+ near-field records —
the same physics behind operational underestimation of large events
(documented in README); a Pd/waveform-based estimator is the upgrade path.

Real-time caveat: catalog PGA is the final peak, typically reached around/after
the S arrival — in the convergence simulation a station's PGA is only used
once its S wave (+ margin) has arrived.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_COEF = {"a": 0.7973371480699482, "b": -2.38856745429763,
                "c": 0.701718041150017}


@dataclass
class MagnitudeEstimate:
    mag: float
    sigma: float          # robust spread of single-station estimates
    n_stations: int


class PgaMagnitude:
    def __init__(self, coef: dict | str | Path | None = None):
        if isinstance(coef, (str, Path)):
            coef = json.loads(Path(coef).read_text())
        self.coef = coef or DEFAULT_COEF

    def station_mags(self, pga_cmps2, dist_ep_km, depth_km):
        pga = np.asarray(pga_cmps2, float)
        r = np.sqrt(np.asarray(dist_ep_km, float) ** 2 + float(depth_km) ** 2)
        ok = np.isfinite(pga) & (pga > 0) & np.isfinite(r) & (r > 1)
        a, b, c = self.coef["a"], self.coef["b"], self.coef["c"]
        m = np.full(len(pga), np.nan)
        m[ok] = (np.log10(pga[ok]) - b * np.log10(r[ok]) - c) / a
        return m

    def estimate(self, pga_cmps2, dist_ep_km, depth_km) -> MagnitudeEstimate | None:
        m = self.station_mags(pga_cmps2, dist_ep_km, depth_km)
        m = m[np.isfinite(m)]
        if len(m) == 0:
            return None
        med = float(np.median(m))
        mad = float(np.median(np.abs(m - med)))
        return MagnitudeEstimate(mag=med, sigma=1.4826 * mad if len(m) > 2 else 0.5,
                                 n_stations=int(len(m)))
