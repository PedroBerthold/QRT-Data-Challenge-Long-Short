"""Rebuild the caches used by src/data.py from the files written by scripts/pack_data.py.

    python scripts/unpack_data.py --pack-dir path/to/received/files

Writes data/X_train.parquet, data/y_train.parquet, data/X_test.parquet (and the sample
submission if present), then checks row counts against audit.json.
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, default=D.DATA_DIR / "pack")
    ap.add_argument("--data-dir", type=Path, default=D.DATA_DIR)
    args = ap.parse_args()
    args.data_dir.mkdir(exist_ok=True)

    def read(stem):
        parts = sorted(args.pack_dir.glob(f"{stem}_part*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no {stem}_part*.parquet in {args.pack_dir}")
        return pd.concat([pd.read_parquet(p) for p in parts]), len(parts)

    train, n_tr = read("train")
    test, n_te = read("test")
    audit_path = args.pack_dir / "audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for name, df in (("train", train), ("test", test)):
            want = audit[name]["rows"]
            assert len(df) == want, f"{name}: {len(df):,} rows received, {want:,} expected (a part is missing)"
    assert train.index.is_unique and test.index.is_unique

    train.drop(columns="TARGET").to_parquet(args.data_dir / "X_train.parquet")
    train[["TARGET"]].rename(columns={"TARGET": "target"}).to_parquet(args.data_dir / "y_train.parquet")
    test.to_parquet(args.data_dir / "X_test.parquet")
    gz = args.pack_dir / "sample_submission.csv.gz"
    if gz.exists():
        with gzip.open(gz, "rb") as a, open(args.data_dir / "sample_submission.csv", "wb") as b:
            shutil.copyfileobj(a, b)
    print(f"train {len(train):,} rows from {n_tr} parts, test {len(test):,} rows from {n_te} parts -> {args.data_dir}")


if __name__ == "__main__":
    main()
