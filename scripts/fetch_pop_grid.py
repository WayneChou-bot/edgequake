"""Phase 9 data: Taiwan 1 km population grid (WorldPop) for PAGER-style
exposure estimates.

Product: WorldPop **Global2** (2015-2030, release R2025A), constrained +
UN-adjusted, 1 km — population is only assigned to cells with mapped built
settlement, and the newest available year (e.g. 2026) is picked
automatically. Downloads the GeoTIFF (~0.2 MB), converts it to a
small npz asset the engine loads (assets/tw_pop_1km.npz), and stamps full
provenance (_meta) so every audit record can cite the population version
it used.

Update policy (population is a slowly-varying, VERSIONED dataset — no
webhooks exist): --check-update polls the WorldPop REST catalog and exits
1 if a newer product year than the local asset exists; a quarterly GitHub
Actions cron uses that to rebuild + notify. Historical audit records keep
the pop_version they were computed with — new data never rewrites the past.

Usage (any machine with internet):
    pip install tifffile imagecodecs numpy
    python scripts/fetch_pop_grid.py --download          # fetch + build
    python scripts/fetch_pop_grid.py --tif twn.tif       # build from file
    python scripts/fetch_pop_grid.py --check-update      # cron mode

Data: WorldPop (www.worldpop.org), School of Geography and Environmental
Science, University of Southampton. Licensed CC BY 4.0 — attribution is in
the console footer and README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# WorldPop Global2 (2015-2030) constrained, UN-adjusted, 1 km. Newer and
# better than the legacy wpgp 2000-2020 line: population is only assigned
# to cells with mapped built settlement.
RELEASE = "R2025A"
BASE = "https://data.worldpop.org/GIS/Population/Global_2015_2030"
ASSET = ROOT / "assets" / "tw_pop_1km.npz"


def url_for(release: str, year: int) -> str:
    return (f"{BASE}/{release}/{year}/TWN/v1/1km_ua/constrained/"
            f"twn_pop_{year}_CN_1km_{release}_UA_v1.tif")


def build(tif_path: Path) -> None:
    try:
        import tifffile
    except ImportError:
        raise SystemExit("pip install tifffile imagecodecs")
    # (imagecodecs is needed because WorldPop tifs are LZW-compressed)
    with tifffile.TiffFile(str(tif_path)) as tf:
        page = tf.pages[0]
        arr = page.asarray().astype(np.float32)
        tags = {t.name: t.value for t in page.tags.values()}
    # GeoTIFF georeferencing: pixel scale + upper-left tiepoint
    sx, sy = tags["ModelPixelScaleTag"][0], tags["ModelPixelScaleTag"][1]
    tp = tags["ModelTiepointTag"]          # (i,j,k, lon,lat,z) of tiepoint
    lon0 = tp[3] + (0.5 - tp[0]) * sx      # center of pixel (0,0)
    lat0 = tp[4] - (0.5 - tp[1]) * sy
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0                     # nodata sentinel -> uninhabited
    total = float(arr.sum())
    m = re.search(r"twn_pop_(\d{4})_CN_1km_(R\d{4}[A-Z])", tif_path.name)
    year = int(m.group(1)) if m else None
    release = m.group(2) if m else None
    meta = {
        "dataset": tif_path.name + " (WorldPop Global2, constrained UA)",
        "popyear": year,
        "release": release,
        "source": (url_for(release, year) if m else str(tif_path.name)),
        "license": "CC BY 4.0 — WorldPop, University of Southampton",
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_population": round(total),
        "shape": list(arr.shape),
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ASSET, pop=arr, lon0=lon0, lat0=lat0,
                        dlon=sx, dlat=sy, meta=json.dumps(meta))
    print(f"[pop] wrote {ASSET} ({ASSET.stat().st_size/1e3:.0f} KB)")
    print(f"[pop] grid {arr.shape[0]}x{arr.shape[1]} @ ~{sx*111.19:.2f} km"
          f" | total population {total/1e6:.2f} M")
    if not 15e6 < total < 30e6:
        print("[pop] WARNING: total looks wrong for Taiwan — check the tif")


def local_meta() -> dict:
    if not ASSET.exists():
        return {}
    try:
        return json.loads(str(np.load(ASSET)["meta"]))
    except Exception:
        return {}


def _exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def check_update() -> int:
    """Newer release folder OR a newer year in ours -> exit 1."""
    meta = local_meta()
    cur_rel, cur_year = meta.get("release"), meta.get("popyear")
    with urllib.request.urlopen(BASE + "/", timeout=60) as r:
        idx = r.read().decode("utf-8", "replace")
    releases = sorted(set(re.findall(r"R\d{4}[A-Z]", idx)))
    newest = releases[-1] if releases else None
    print(f"[pop] local: {cur_rel} year {cur_year} | newest release: "
          f"{newest} (all: {', '.join(releases)})")
    if not cur_rel or not cur_year:
        print("[pop] no local asset — run with --download")
        return 1
    if newest and newest > cur_rel:
        # only flag if it actually has a Taiwan file we can use
        for y in range(2030, 2014, -1):
            if _exists(url_for(newest, y)):
                print(f"[pop] UPDATE: {newest} has TWN {y} — rerun "
                      "--download and commit the new asset")
                return 1
        print(f"[pop] {newest} exists but has no TWN 1km file yet")
    if _exists(url_for(cur_rel, int(cur_year) + 1)):
        print(f"[pop] UPDATE: year {cur_year + 1} available in {cur_rel}")
        return 1
    print("[pop] up to date")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tif", default=None)
    ap.add_argument("--check-update", action="store_true")
    args = ap.parse_args()

    if args.check_update:
        sys.exit(check_update())
    if args.tif:
        build(Path(args.tif))
        return
    if args.download:
        year = time.gmtime().tm_year
        url = None
        for y in range(min(year, 2030), 2014, -1):   # newest available year
            if _exists(url_for(RELEASE, y)):
                url = url_for(RELEASE, y)
                break
        if url is None:
            raise SystemExit(f"[pop] no TWN file found under {RELEASE}")
        tmp = ROOT / "data" / "pop" / Path(url).name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pop] downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        print(f"[pop] saved {tmp} ({tmp.stat().st_size/1e6:.1f} MB)")
        build(tmp)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
