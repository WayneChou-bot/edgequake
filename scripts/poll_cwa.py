"""Poll CWA open data for recent earthquake reports -> monitor board JSON.

Runs on the laptop next to the monitor page: fetches 顯著有感 (E-A0015-001)
and 小區域 (E-A0016-001) reports every --interval seconds, normalizes them
into the shape web/monitor.html expects, and (optionally) serves the board.

Setup: register at https://opendata.cwa.gov.tw (free), get an API key
(會員中心 -> API授權碼), then:
    set CWA_API_KEY=CWA-XXXX...           (Windows; or use --key)
    python scripts/poll_cwa.py --serve
    -> http://localhost:8700

Offline development: `python scripts/poll_cwa.py --mock --serve` uses fake
events so the page can be built/tested without a key or network.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
DATASETS = [("E-A0015-001", "significant"), ("E-A0016-001", "local")]

I_ORDER = {"1級": 1, "2級": 2, "3級": 3, "4級": 4, "5弱": 5, "5強": 5.5,
           "6弱": 6, "6強": 6.5, "7級": 7}


def norm_intensity(txt: str) -> str | None:
    m = re.search(r"([1-7])(級|弱|強)?", str(txt or ""))
    if not m:
        return None
    return m.group(1) + (m.group(2) if m.group(2) in ("弱", "強") else "級")


def parse_report(eq: dict, kind: str) -> dict | None:
    try:
        info = eq.get("EarthquakeInfo", {})
        epi = info.get("Epicenter", {})
        mag = info.get("EarthquakeMagnitude", {})
        # max intensity across shaking areas
        best, best_v = None, -1
        for area in (eq.get("Intensity", {}) or {}).get("ShakingArea", []):
            it = norm_intensity(area.get("AreaIntensity"))
            v = I_ORDER.get(it or "", -1)
            if v > best_v:
                best, best_v = it, v
        return dict(
            t=str(info.get("OriginTime", ""))[:16],
            lat=float(epi.get("EpicenterLatitude")),
            lon=float(epi.get("EpicenterLongitude")),
            depth=round(float(info.get("FocalDepth", 0) or 0)),
            mag=float(mag.get("MagnitudeValue")),
            loc=str(epi.get("Location", "")) or str(
                eq.get("ReportContent", ""))[:40],
            maxI=best, url=str(eq.get("Web", "")) or None, kind=kind)
    except (TypeError, ValueError, KeyError):
        return None


def fetch_cwa(key: str, limit: int = 15) -> dict:
    events = []
    for ds, kind in DATASETS:
        url = (f"{API}/{ds}?" + urllib.parse.urlencode(
            dict(Authorization=key, limit=limit, format="JSON")))
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
        for eq in (data.get("records", {}) or {}).get("Earthquake", []):
            ev = parse_report(eq, kind)
            if ev:
                events.append(ev)
    events.sort(key=lambda e: e["t"], reverse=True)
    return dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                source="cwa-opendata", events=events[:30])


MOCK = dict(updated="(mock)", source="mock", events=[
    dict(t="2026-07-31 00:58", lat=23.31, lon=121.74, depth=20, mag=4.7,
         loc="臺東縣政府北北東方44.1公里(位於臺東縣近海)", maxI="4級",
         url=None, kind="significant"),
    dict(t="2026-07-30 08:10", lat=23.79, lon=121.57, depth=22, mag=3.8,
         loc="花蓮縣政府南方21.2公里(位於花蓮縣近海)", maxI="2級",
         url=None, kind="local"),
    dict(t="2026-07-27 10:14", lat=23.24, lon=120.52, depth=10, mag=4.8,
         loc="臺南市政府東北東方36.7公里(位於臺南市楠西區)", maxI="4級",
         url=None, kind="significant"),
])


def serve(webroot: Path, port: int):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(webroot), **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[monitor] http://localhost:{port}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("CWA_API_KEY", ""))
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "console_web"),
                    help="shared runtime dir (also used by run_live.py)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.mock and not args.key:
        raise SystemExit("no API key: set CWA_API_KEY or use --key "
                         "(register free at https://opendata.cwa.gov.tw), "
                         "or run --mock for offline testing")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # ONE source of truth: docs/index.html — always refresh the copy
    built = ROOT / "docs" / "index.html"
    if not built.exists():
        raise SystemExit("docs/index.html missing — run "
                         "scripts/build_dashboard.py first")
    shutil.copy(built, out / "index.html")
    if args.serve:
        serve(out, args.port)

    while True:
        try:
            payload = MOCK if args.mock else fetch_cwa(args.key)
            tmp = out / "cwa_events.json.tmp"
            # encoding must be explicit: Windows defaults to cp950 -> mojibake
            tmp.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(out / "cwa_events.json")
            print(f"[monitor] {payload['updated']}: "
                  f"{len(payload['events'])} events"
                  + (f" | latest M{payload['events'][0]['mag']} "
                     f"{payload['events'][0]['loc'][:24]}"
                     if payload["events"] else ""))
        except Exception as e:
            print(f"[monitor] fetch FAILED: {e}")
        if args.once:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break
    if args.serve and not args.once:
        return
    if args.serve:
        print("[monitor] serving final state; Ctrl-C to quit")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
