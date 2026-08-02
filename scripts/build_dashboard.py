"""Build the self-contained interactive replay dashboard (Phase 3).

Precomputes everything in Python (simulation frames, county intensities, PWS
alerts, map projection) and emits ONE offline-capable HTML file — no CDN, no
tiles, works from file:// and GitHub Pages alike.

Usage:
    python scripts/build_dashboard.py            # -> web/index.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# fixed equirectangular projection (matches the GIF view)
LON0, LON1, LAT0, LAT1 = 119.7, 122.5, 21.6, 25.6
W = 520
ASPECT = np.cos(np.radians(23.7))
H = int(W * (LAT1 - LAT0) / ((LON1 - LON0) * ASPECT))
KM_PER_DEG = 111.19
PX_PER_KM = W / ((LON1 - LON0) * KM_PER_DEG * ASPECT)

OFFICIAL = {
    "0403": [(9.0, "CWA report #1: M6.2 — no Taipei-area alert"),
             (15.0, "CWA report #2: M6.8")],
    "dapu": [(7.9, "CWA alert issued (+~5 s delivery)")],
}
TITLES = {"0403": "2024-04-03 Hualien M7.2 (offshore)",
          "dapu": "2025-01-21 Chiayi Dapu ML6.4 (inland)"}


def project(lon, lat):
    x = (np.asarray(lon) - LON0) / (LON1 - LON0) * W
    y = (LAT1 - np.asarray(lat)) / (LAT1 - LAT0) * H
    return x, y


def coastline_paths():
    gj = json.loads((ROOT / "assets" / "taiwan_coastline_ne50m.json").read_text())
    paths = []
    for f in gj["features"]:
        geom = f["geometry"]
        rings = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in rings:
            ring = np.array(poly[0])
            x, y = project(ring[:, 0], ring[:, 1])
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + " Z"
            paths.append(d)
    return paths


WF_FILES = {"0403": ("raw_0403", ["hualien0403_HH.mseed", "hualien0403_EH.mseed",
                                    "hualien0403_HL.mseed"]),
            "dapu": ("raw_dapu", ["dapu0121_HH.mseed", "dapu0121_HL.mseed"])}
WF_N_STATIONS = 6
WF_FS = 20.0          # downsampled rate for display
WF_PRE, WF_POST = 5.0, 75.0   # seconds around first trigger


def extract_waveforms(key, payload, base_dir):
    """Downsampled vertical-component strips for the earliest N stations."""
    import obspy

    dirname, files = WF_FILES[key]
    st = obspy.Stream()
    for f in files:
        p = Path(base_dir) / dirname / f
        if p.exists():
            st += obspy.read(str(p))
    if not len(st):
        return None
    st.merge(method=1, fill_value=0)
    t_ref_epoch = min(s0["tp"] for s0 in payload["stations"])  # rel=0 epoch? tp is rel
    # payload stations tp are relative to first trigger; need absolute epoch:
    # first trigger abs = origin_epoch - origin_rel? origin_rel = origin - t_ref
    origin_rel = payload["origin_rel"]
    strips = []
    for s0 in sorted(payload["stations"], key=lambda x: x["tp"])[:WF_N_STATIONS]:
        cand = [tr for tr in st if tr.stats.station == s0["code"]
                and tr.stats.channel[-1] == "Z"]
        if not cand:
            continue
        pref = {"HH": 0, "EH": 1, "HL": 2}
        tr = sorted(cand, key=lambda t: pref.get(t.stats.channel[:2], 9))[0].copy()
        fs0 = tr.stats.sampling_rate
        # absolute epoch of first trigger: use trace timing + station tp
        # station tp (rel) + t_ref(abs). t_ref(abs) unknown here -> derive:
        # origin(abs) known via truth origin_time; origin_rel = origin - t_ref
        origin_abs = obspy.UTCDateTime(payload["truth"]["origin_time"])
        t_ref_abs = origin_abs - origin_rel
        w0 = t_ref_abs - WF_PRE
        tr.trim(w0, t_ref_abs + WF_POST, pad=True, fill_value=0)
        tr.detrend("demean")
        dec = int(round(fs0 / WF_FS))
        tr.filter("lowpass", freq=WF_FS * 0.4)
        tr.decimate(dec, no_filter=True)
        data = tr.data.astype(float)
        peak = max(abs(data).max(), 1e-9)
        data = np.round(data / peak, 3)
        from edgequake.location.locator import haversine_km
        dist = haversine_km(payload["truth"]["lat"], payload["truth"]["lon"],
                            s0["lat"], s0["lon"])
        strips.append({"code": s0["code"], "ch": tr.stats.channel,
                       "tp": s0["tp"], "ts": s0["ts"],
                       "dist": round(dist, 1), "pga": s0.get("pga"),
                       "pprob": s0.get("pprob"),
                       "z": data.tolist()})
    return {"fs": WF_FS, "t0": -WF_PRE, "strips": strips}


def main() -> None:
    from demo_convergence import load_replay_json

    from edgequake.location.replay_sim import simulate

    events = {}
    for key in ("0403", "dapu"):
        p = ROOT / "outputs" / f"replay_{key}.json"
        if not p.exists():
            print(f"[dash] missing {p}, skipping {key}")
            continue
        ev, truth = load_replay_json(p)
        print(f"[dash] simulating {key} ({len(ev)} stations)...")
        from edgequake.location.site import load_site_terms
        site_terms = load_site_terms(ROOT / "outputs" / "site_terms.json")
        payload = simulate(ev, truth, max_stations=60,
                           site_terms=site_terms)
        payload["title"] = TITLES[key]
        payload["official"] = OFFICIAL.get(key, [])
        # projected coordinates baked in
        for s in payload["stations"]:
            x, y = project(s["lon"], s["lat"])
            s["x"], s["y"] = round(float(x), 1), round(float(y), 1)
        for c in payload["counties"]:
            x, y = project(c["lon"], c["lat"])
            c["x"], c["y"] = round(float(x), 1), round(float(y), 1)
        tx, ty = project(payload["truth"]["lon"], payload["truth"]["lat"])
        payload["truth"]["x"], payload["truth"]["y"] = round(float(tx), 1), round(float(ty), 1)
        wf = extract_waveforms(key, payload, ROOT.parent
                               if not len(sys.argv) > 1 else sys.argv[1])
        if wf:
            payload["wf"] = wf
            print(f"[dash]   waveform strips: {len(wf['strips'])}")
        events[key] = payload

    data = {"events": events, "map": {"w": W, "h": H, "pxPerKm": PX_PER_KM,
                                      "coast": coastline_paths()}}
    template = (ROOT / "web" / "app_template.html").read_text(
        encoding="utf-8")
    html = template.replace("/*__DATA__*/", "const DATA = " +
                            json.dumps(data, separators=(",", ":")) + ";")
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[dash] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    # the same unified file IS the Vercel deployment page
    vercel = ROOT / "vercel" / "index.html"
    if vercel.parent.exists():
        vercel.write_text(html, encoding="utf-8")
        print(f"[dash] synced {vercel}")


if __name__ == "__main__":
    main()
