"""Run the EdgeQuake live engine + local web console (Phase 4 skeleton).

Sources:
    replay (default) — GDMS miniSEED replayed at wall-clock speed. This is
        the honest demo mode: the engine sees exactly what it would see live.
    seedlink — real-time SeedLink (no public Taiwan feed exists today; use
        GEOFON for client/latency validation, or a personal Raspberry Shake).

Usage (laptop, from the repo root; raw data folders live next to the repo):
    python scripts/run_live.py --event 0403 --speed 1
    python scripts/run_live.py --event dapu --speed 4 --max-stations 30
    python scripts/run_live.py --source seedlink --server geofon.gfz.de:18000 \
        --streams GE.WLF.BH?,GE.STU.BH?

Then open http://localhost:8600 — the console polls state.json twice a second.
"""
from __future__ import annotations

import argparse
import http.server
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVENTS = {
    "0403": ("raw_0403", ["hualien0403_HH.mseed", "hualien0403_EH.mseed",
                          "hualien0403_HL.mseed"]),
    "dapu": ("raw_dapu", ["dapu0121_HH.mseed", "dapu0121_HL.mseed"]),
}


def serve(webroot: Path, port: int):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(webroot), **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["replay", "seedlink"],
                    default="replay")
    ap.add_argument("--event", choices=list(EVENTS), default="0403")
    ap.add_argument("--base-dir", default=str(ROOT.parent))
    ap.add_argument("--dataless", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-stations", type=int, default=40)
    ap.add_argument("--weights", default="none")
    ap.add_argument("--state-dict",
                    default=str(ROOT / "outputs" / "phasenet_cwa_ft.pt"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--server", default="geofon.gfz.de:18000")
    ap.add_argument("--streams", default="",
                    help="seedlink: NET.STA.SEL comma list, e.g. GE.WLF.BH?")
    ap.add_argument("--webroot",
                    default=str(ROOT / "outputs" / "console_web"),
                    help="shared runtime dir (also used by poll_cwa.py)")
    ap.add_argument("--notify", action="store_true",
                    help="send email/Telegram on PWS alert (config via env: "
                         "EQ_SMTP_USER/PASS/MAIL_TO, EQ_TG_TOKEN/CHAT)")
    args = ap.parse_args()

    from edgequake.live.engine import LiveEngine
    from edgequake.live.sources import ReplaySource, SeedLinkSource, StationMeta
    from edgequake.pickers.seisbench_picker import SeisBenchPhaseNet

    webroot = Path(args.webroot)
    webroot.mkdir(parents=True, exist_ok=True)
    (webroot / "state.json").unlink(missing_ok=True)  # no stale state
    # ONE source of truth: docs/index.html (built by build_dashboard.py).
    # The runtime dir is a disposable cache — always refresh the copy.
    built = ROOT / "docs" / "index.html"
    if not built.exists():
        raise SystemExit("docs/index.html missing — run "
                         "scripts/build_dashboard.py first")
    shutil.copy(built, webroot / "index.html")

    if args.source == "replay":
        base = Path(args.base_dir)
        dirname, files = EVENTS[args.event]
        dataless = args.dataless or (base / "raw_resp" /
                                     "Dataless_CWASN.dataless")
        print(f"[live] loading replay source ({args.event}, "
              f"speed {args.speed}x)...")
        src = ReplaySource([base / dirname / f for f in files], dataless,
                           speed=args.speed, max_stations=args.max_stations)
        label = f"REPLAY {args.event} @ {args.speed:g}x"
    else:
        streams = [tuple(s.split(".")) for s in args.streams.split(",") if s]
        if not streams:
            raise SystemExit("--streams required for seedlink "
                             "(e.g. GE.WLF.BH?)")
        meta = [StationMeta(code=sta, lat=0.0, lon=0.0)
                for _, sta, _ in streams]
        src = SeedLinkSource(args.server, streams, meta)
        label = f"SEEDLINK {args.server}"

    notifier = None
    if args.notify:
        from edgequake.live.notify import Notifier

        notifier = Notifier.from_env()
        print(f"[live] notify channels: {notifier.channels or 'NONE'}")

    picker = SeisBenchPhaseNet(weights=args.weights,
                               state_dict_path=args.state_dict or None)
    engine = LiveEngine(picker, src.stations, threshold=args.threshold,
                        mode_label=label, notifier=notifier)
    print(f"[live] {len(src.stations)} stations | picker {picker.name}")

    serve(webroot, args.port)
    print(f"[live] console: http://localhost:{args.port}")

    t_wall0 = time.monotonic()
    n_ticks = 0
    try:
        for tick in src.ticks():
            engine.on_tick(tick)
            engine.write_state(webroot / "state.json")
            n_ticks += 1
            if n_ticks % 30 == 0:
                s = engine.state()
                evtxt = (f"event k={s['event']['k']} "
                         f"M{s['event'].get('mag', '–')}"
                         if s["event"] else "no event")
                print(f"[live] t+{n_ticks}s | {s['n_live']} feeding | "
                      f"{evtxt} | infer {s['infer_ms']:.0f} ms")
    except KeyboardInterrupt:
        pass
    print(f"[live] done ({n_ticks} ticks, "
          f"{time.monotonic() - t_wall0:.0f}s wall). Ctrl-C to quit server.")
    # keep serving the final state so the console stays viewable
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
