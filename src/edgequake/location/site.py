"""Station site-effect terms (Phase 7).

outputs/site_terms.json is built by scripts/build_site_terms.py from catalog
PGA residuals. dS is in log10 units: observed PGA at that station is
typically 10**dS times the GMPE-average site, so dividing by 10**dS gives
the shaking an average site would have seen — which is what the magnitude
inversion assumes.

Coverage note: terms exist for CWA seismic stations (the catalog years).
Unknown stations (e.g. TREM MEMS codes) fall back to dS=0 — no correction,
same behavior as before Phase 7.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_site_terms(path: str | Path) -> dict[str, float]:
    """Return {station_code: dS}; {} if the file is absent/invalid."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: float(v["dS"]) for k, v in d.get("terms", {}).items()}
    except Exception:
        return {}


def pga_correction(code: str, terms: dict[str, float] | None) -> float:
    """Multiplicative factor: corrected = observed / factor."""
    if not terms:
        return 1.0
    return 10.0 ** terms.get(code, 0.0)
