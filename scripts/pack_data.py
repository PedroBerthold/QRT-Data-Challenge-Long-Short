"""Pack the challenge data into a few small files that can be sent as chat attachments.

Run from the repo root, with the challenge CSVs in data/ (as for every other script):

    python scripts/pack_data.py

Output: data/pack/  (git-ignored, the challenge data must not be published)
    train_part01.parquet ...   X_train + TARGET, all rows, all columns
    test_part01.parquet ...    X_test, all rows, all columns
    sample_submission.csv.gz   if a sample submission is found in data/
    audit.json                 shapes, dtypes, missing values, counts (plain text)

Nothing is sampled: every row and every column is kept, in the original order.
The only change is float64 -> float32 (7 significant digits; the sign of every
value is preserved), which with zstd makes the files about 3x smaller than a
plain parquet. Each file stays under --max-mb so it fits in one attachment.

scripts/unpack_data.py does the reverse and rebuilds the parquet caches used by src/data.py.
"""
import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


def write_parts(df: pd.DataFrame, out: Path, stem: str, max_mb: float) -> list:
    """Split by rows (original order) into the fewest files that all stay under max_mb."""
    n = 1
    while True:
        for old in out.glob(f"{stem}_part*.parquet"):
            old.unlink()
        bounds = np.linspace(0, len(df), n + 1).astype(int)
        files = []
        for k in range(n):
            f = out / f"{stem}_part{k + 1:02d}.parquet"
            df.iloc[bounds[k]:bounds[k + 1]].to_parquet(f, compression="zstd", compression_level=19)
            files.append(f)
        biggest = max(f.stat().st_size for f in files) / 1e6
        if biggest <= max_mb:
            return files
        n = max(n + 1, math.ceil(n * biggest / max_mb * 1.05))


def describe(df: pd.DataFrame) -> dict:
    per_ts = df.groupby("TS").size()
    per_alloc = df.groupby("ALLOCATION").size()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "index_name": df.index.name,
        "index_dtype": str(df.index.dtype),
        "index_is_unique": bool(df.index.is_unique),
        "index_head": [str(v) for v in df.index[:3]],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v},
        "n_ts": int(df.TS.nunique()),
        "n_allocations": int(df.ALLOCATION.nunique()),
        "groups": {str(k): int(v) for k, v in df.GROUP.value_counts().items()},
        "allocations_per_ts": per_ts.describe().round(1).to_dict(),
        "ts_per_allocation": per_alloc.describe().round(1).to_dict(),
        "ts_head": [str(v) for v in df.TS.unique()[:5]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=D.DATA_DIR)
    ap.add_argument("--max-mb", type=float, default=20.0, help="maximum size of one output file")
    args = ap.parse_args()
    D.DATA_DIR = args.data_dir
    out = args.data_dir / "pack"
    out.mkdir(exist_ok=True)

    X, y = D.load_train()
    Xt = D.load_test()
    train = X.assign(TARGET=y)

    audit = {"train": describe(train), "test": describe(Xt)}
    audit["train"]["share_target_positive"] = float((y > 0).mean())
    audit["train"]["target_missing"] = int(y.isna().sum())
    audit["allocations_test_not_in_train"] = int((~Xt.ALLOCATION.isin(X.ALLOCATION.unique())).sum())
    audit["ts_in_both"] = int(len(set(X.TS) & set(Xt.TS)))

    files = write_parts(shrink(train), out, "train", args.max_mb)
    files += write_parts(shrink(Xt), out, "test", args.max_mb)

    sample = sorted(args.data_dir.glob("*sample*.csv"))
    if sample:
        dst = out / "sample_submission.csv.gz"
        with open(sample[0], "rb") as a, gzip.open(dst, "wb") as b:
            shutil.copyfileobj(a, b)
        files.append(dst)
        s = pd.read_csv(sample[0])
        audit["sample_submission"] = {
            "file": sample[0].name, "columns": list(s.columns), "rows": len(s),
            "same_order_as_X_test": bool(len(s) == len(Xt) and (s.iloc[:, 0].values == Xt.index.values).all()),
        }
    else:
        audit["sample_submission"] = None
        print("No *sample*.csv in data/ (not blocking, but add it if you have it).")

    audit["files"] = {f.name: round(f.stat().st_size / 1e6, 2) for f in files}
    (out / "audit.json").write_text(json.dumps(audit, indent=1, default=str))

    print(f"\nTrain {len(train):,} rows, test {len(Xt):,} rows. Files to send (all of them), in {out}:")
    for f in files + [out / "audit.json"]:
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:6.1f} MB")
    print(f"  total {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
