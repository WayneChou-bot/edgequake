"""Download CWA (Taiwan) chunks from Hugging Face and install them into the
SeisBench cache — bypassing the SeisBench repository entirely.

Why: the SeisBench repository serves CWA as UNCOMPRESSED per-year HDF5 files
(waveforms_2019.hdf5 alone is ~43 GB) and is slow from Taiwan (~1.3 MB/s).
Hugging Face serves one compressed tar (merge2019_2021.tar.gz, ~27 GB for all
three years) over a fast CDN, with resumable downloads.

Usage:
    python scripts/fetch_cwa_hf.py                       # 2019-2021 (dev+test)
    python scripts/fetch_cwa_hf.py --tar D:\merge2019_2021.tar.gz
        # if you downloaded the tar manually in a browser from
        # https://huggingface.co/datasets/NLPLabNTUST/Merged-CWA/tree/main
    python scripts/fetch_cwa_hf.py --file merge2015_2018.tar.gz   # other spans

Tip: `pip install hf_transfer` before running enables multi-connection
downloads (the script turns it on automatically if installed).

After this completes, evaluation runs entirely from the local cache:
    python scripts/eval_picking.py --dataset cwa --chunks _2020 _2021 --confirm-download --limit 1000
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

REPO = "NLPLabNTUST/Merged-CWA"
DEFAULT_FILE = "merge2019_2021.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE,
                    help="tar.gz on the HF repo (default: merge2019_2021.tar.gz)")
    ap.add_argument("--tar", default=None,
                    help="path to an already-downloaded tar.gz (skips download)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep the tar.gz after extraction")
    args = ap.parse_args()

    # enable accelerated HF downloads. Newer huggingface_hub uses the Xet
    # backend (HF_XET_HIGH_PERFORMANCE); older versions used hf_transfer.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass
    print("[fetch] high-performance HF transfer flags set "
          "(Xet / hf_transfer, whichever applies)")

    from seisbench.data import CWA

    target = Path(CWA._path_internal())
    target.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] SeisBench cache target: {target}")

    # sanity check: warn about suspicious leftovers from aborted downloads
    for f in target.glob("waveforms_*.hdf5"):
        print(f"[fetch] NOTE: {f.name} already present "
              f"({f.stat().st_size / 1e9:.1f} GB) — if a previous download was "
              "aborted, delete it before evaluating.")

    if args.tar:
        tar_path = Path(args.tar)
        if not tar_path.exists():
            raise SystemExit(f"tar not found: {tar_path}")
    else:
        from huggingface_hub import hf_hub_download

        print(f"[fetch] downloading {args.file} from {REPO} (resumable — "
              "re-run this script if interrupted)")
        tar_path = Path(hf_hub_download(
            repo_id=REPO, filename=args.file, repo_type="dataset",
            local_dir=target,
        ))

    print(f"[fetch] extracting {tar_path.name} -> {target} (takes a while)")
    try:
        import seisbench.util

        with tarfile.open(tar_path, "r:gz") as tar:
            seisbench.util.safe_extract_tar(tar, target)
    except AttributeError:  # older seisbench without safe_extract_tar
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(target)

    # some tars nest files in a subdirectory — flatten if needed
    for sub in [d for d in target.iterdir() if d.is_dir()]:
        for f in list(sub.glob("*.csv")) + list(sub.glob("*.hdf5")):
            dest = target / f.name
            if not dest.exists():
                shutil.move(str(f), dest)
        if not any(sub.iterdir()):
            sub.rmdir()

    # add the train/dev/test split column exactly like SeisBench would
    from seisbench.data.cwa import CWABase

    n_meta = 0
    for meta in sorted(target.glob("metadata*.csv")):
        CWABase._add_split(meta)
        n_meta += 1
    print(f"[fetch] split column ensured on {n_meta} metadata file(s)")

    # normalize mixed source_origin_time formats (missing fractional seconds
    # crash pandas>=2.0 format inference inside SeisBench)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fix_cwa_metadata import fix_file

    for meta in sorted(target.glob("metadata*.csv")):
        fix_file(meta)

    if not args.keep_tar and not args.tar:
        tar_path.unlink(missing_ok=True)
        # hf_hub_download may leave a .cache dir with the blob
        cache_dir = target / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        print("[fetch] removed tar to free space")

    chunks = sorted({p.stem.replace("metadata_", "_")
                     for p in target.glob("metadata_*.csv")})
    print(f"[fetch] installed chunks: {chunks}")
    print("[fetch] done. Evaluate with e.g.:\n"
          "  python scripts/eval_picking.py --dataset cwa "
          f"--chunks {' '.join(c for c in chunks if c in ('_2020','_2021'))} "
          "--confirm-download --limit 1000")


if __name__ == "__main__":
    main()
