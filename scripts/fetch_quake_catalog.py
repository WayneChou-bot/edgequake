"""Phase 10 data: historical Taiwan earthquake catalog for similar-event
retrieval.

Pulls the USGS FDSN event catalog (no key needed) for the Taiwan region,
M>=5.0 since 1973 — every significant quake incl. 921 Chi-Chi, 2016
Meinong, 2018/2024 Hualien, 2022 Chihshang — and bakes a compact
assets/quake_catalog.json the engine searches in ~1 ms.

Why USGS and not CWA: CWA open data only exposes recent reports; USGS
gives five decades in one unauthenticated call. Magnitudes are Mw/mb-class
(vs CWA ML) — close enough for "which past quake was this like", and the
difference is stated where results are shown.

Usage (any machine with internet):
    python scripts/fetch_quake_catalog.py            # build/refresh
Re-run any time: the file is rebuilt from the live query (idempotent).
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
       "&minlatitude=21.3&maxlatitude=25.8"
       "&minlongitude=119.2&maxlongitude=123.0"
       "&minmagnitude=5.0&starttime=1973-01-01&orderby=time&limit=20000")
ASSET = ROOT / "assets" / "quake_catalog.json"


def main() -> None:
    print("[catalog] querying USGS FDSN (Taiwan region, M>=5, 1973+)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "EdgeQuake"})
    with urllib.request.urlopen(req, timeout=120) as r:
        gj = json.load(r)
    events = []
    for f in gj.get("features", []):
        p, g = f.get("properties", {}), f.get("geometry", {})
        c = g.get("coordinates") or [None, None, None]
        if p.get("mag") is None or c[0] is None:
            continue
        events.append({
            "t": time.strftime("%Y-%m-%dT%H:%M",
                               time.gmtime(p["time"] / 1000)),
            "lat": round(c[1], 3), "lon": round(c[0], 3),
            "depth": round(c[2] or 0, 1),
            "mag": round(p["mag"], 1), "mt": p.get("magType", ""),
            "place": (p.get("place") or "")[:60],
        })
    events.sort(key=lambda e: e["t"])
    out = {
        "_meta": {
            "source": "USGS FDSN event API (earthquake.usgs.gov)",
            "region": "21.3-25.8N 119.2-123.0E", "minmag": 5.0,
            "span": f"1973..{events[-1]['t'][:4]}" if events else "",
            "n_events": len(events),
            "note": "USGS Mw/mb-class magnitudes, not CWA ML",
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                        time.gmtime()),
        },
        "events": events,
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_text(json.dumps(out, ensure_ascii=False,
                                separators=(",", ":")), encoding="utf-8")
    print(f"[catalog] wrote {ASSET} ({ASSET.stat().st_size/1e3:.0f} KB, "
          f"{len(events)} events)")
    big = [e for e in events if e["mag"] >= 7.0]
    print(f"[catalog] M7+: {len(big)} — " +
          "; ".join(f"{e['t'][:10]} M{e['mag']}" for e in big[-6:]))


if __name__ == "__main__":
    main()
