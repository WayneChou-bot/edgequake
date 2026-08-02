"""Phase 10: similar historical event retrieval.

"M6.1 at 23.9N 121.5E, 18 km deep" means little to most people;
"like the 2022 Chihshang earthquake" means a lot. Given a live estimate,
search the USGS-derived catalog (assets/quake_catalog.json, built by
scripts/fetch_quake_catalog.py) for the closest historical events in a
joint (epicenter distance, magnitude, depth) metric.

Scoring: score = d_epi/40km + |dM|/0.4 + |d_depth|/25km — one unit of
score per "clearly different" step in each dimension. Lower is better.
Famous events carry curated zh-TW names (matched by UTC date, so the
labels attach to real catalog rows, never invented coordinates).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .location.locator import haversine_km

# curated zh names for events people know: (name, mainshock USGS mag).
# Keyed by UTC date (near-midnight local times can shift a day — both
# dates listed where relevant). The label only attaches when the catalog
# row's magnitude is within ±0.4 of the mainshock — otherwise a same-day
# aftershock would wear the mainshock's name (a real bug caught in
# testing: a 921-day M6.4 aftershock labeled "921 集集大地震").
FAMOUS = {
    "1999-09-20": ("921 集集大地震", 7.7),
    "2002-03-31": ("331 大地震", 7.1),
    "2006-12-26": ("恆春地震", 7.1),
    "2010-03-04": ("甲仙地震", 6.3),
    "2013-06-02": ("南投地震", 6.2),
    "2016-02-05": ("美濃地震", 6.4),
    "2016-02-06": ("美濃地震", 6.4),
    "2018-02-06": ("花蓮地震", 6.4),
    "2019-04-18": ("秀林地震", 6.1),
    "2022-09-17": ("關山地震", 6.5),
    "2022-09-18": ("池上地震", 6.9),
    "2024-04-02": ("0403 花蓮地震", 7.4),
    "2024-04-03": ("0403 花蓮地震", 7.4),
    "2025-01-20": ("大埔地震", 6.2),
    "2025-01-21": ("大埔地震", 6.2),
}


def famous_name(date: str, mag: float) -> str | None:
    ent = FAMOUS.get(date)
    if ent and abs(mag - ent[1]) <= 0.4:
        return ent[0]
    return None


class SimilarEvents:
    def __init__(self, asset: str | Path):
        d = json.loads(Path(asset).read_text(encoding="utf-8"))
        ev = d.get("events", [])
        self.meta = d.get("_meta", {})
        self.t = [e["t"] for e in ev]
        self.lat = np.array([e["lat"] for e in ev])
        self.lon = np.array([e["lon"] for e in ev])
        self.depth = np.array([e["depth"] for e in ev])
        self.mag = np.array([e["mag"] for e in ev])
        self.place = [e.get("place", "") for e in ev]

    def find(self, lat: float, lon: float, depth_km: float, mag: float,
             k: int = 3, exclude_date: str | None = None) -> list[dict]:
        """exclude_date ('YYYY-MM-DD'): skip catalog rows within ±3 days —
        so a replayed/audited event never 'matches itself'."""
        if not len(self.mag):
            return []
        d = np.array([haversine_km(lat, lon, la, lo)
                      for la, lo in zip(self.lat, self.lon)])
        score = (d / 40.0 + np.abs(self.mag - mag) / 0.4
                 + np.abs(self.depth - depth_km) / 25.0)
        if exclude_date:
            try:
                x = np.datetime64(exclude_date)
                dt = np.array([np.datetime64(t[:10]) for t in self.t])
                score[np.abs((dt - x).astype(int)) <= 3] = np.inf
            except Exception:
                pass
        out = []
        for i in np.argsort(score)[:k]:
            if not np.isfinite(score[i]):
                break
            date = self.t[i][:10]
            out.append({
                "t": self.t[i], "mag": float(self.mag[i]),
                "depth": float(self.depth[i]),
                "d_km": round(float(d[i]), 1),
                "place": self.place[i],
                "zh": famous_name(date, float(self.mag[i])),
                "score": round(float(score[i]), 2),
            })
        return out


_model = None


def get_similar(root: str | Path | None = None):
    """Lazy singleton; None if the catalog asset is missing."""
    global _model
    if _model is None:
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        asset = root / "assets" / "quake_catalog.json"
        if not asset.exists():
            return None
        _model = SimilarEvents(asset)
    return _model


def fmt_similar(s: dict, lang: str = "zh") -> str:
    name = s.get("zh") or s.get("place") or s["t"][:10]
    if lang == "zh":
        return f"{s['t'][:10]} {name} M{s['mag']:.1f}（相距 {s['d_km']:.0f} km）"
    return f"{s['t'][:10]} {s.get('place') or name} M{s['mag']:.1f} ({s['d_km']:.0f} km away)"
