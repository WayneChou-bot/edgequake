"""One-time fix for CWA metadata: normalize `source_origin_time` strings.

The CWA metadata mixes timestamp formats — most rows carry fractional seconds
("2020-05-29 08:20:12.170000") but some don't ("2020-05-29 08:20:12").
pandas >= 2.0 infers the format from the first value and then raises on the
inconsistent rows inside SeisBench's WaveformDataset init. Appending
".000000" where the fraction is missing makes the column uniform; nothing
else in the files is touched (the split column is preserved).

Usage:
    python scripts/fix_cwa_metadata.py            # fixes the SeisBench cwa cache
    python scripts/fix_cwa_metadata.py PATH_DIR   # or any dir with metadata*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COL = "source_origin_time"


def fix_file(path: Path) -> int:
    df = pd.read_csv(path, low_memory=False)
    if COL not in df.columns:
        print(f"[fix] {path.name}: no {COL} column, skipped")
        return 0
    s = df[COL].astype("string")
    mask = s.notna() & ~s.str.contains(".", regex=False)
    n = int(mask.sum())
    if n:
        df.loc[mask, COL] = s[mask] + ".000000"
        df.to_csv(path, index=False)
    print(f"[fix] {path.name}: normalized {n} of {len(df)} rows")
    return n


def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        from seisbench.data import CWA

        target = Path(CWA._path_internal())
    files = sorted(target.glob("metadata*.csv"))
    if not files:
        raise SystemExit(f"no metadata*.csv found in {target}")
    total = sum(fix_file(f) for f in files)
    print(f"[fix] done — {total} rows normalized across {len(files)} files")


if __name__ == "__main__":
    main()
