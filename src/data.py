"""Data loading. Put the challenge files in data/ (X_train*.csv, y_train*.csv, X_test*.csv)."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RET = [f"RET_{i}" for i in range(1, 21)]
VOL = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]


def _find(pattern: str) -> Path:
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {DATA_DIR}")
    return files[0]


def _read(pattern: str, cache: str) -> pd.DataFrame:
    """Read the CSV once, then use a parquet cache (much faster)."""
    pq = DATA_DIR / cache
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(_find(pattern), index_col="ROW_ID")
    df.to_parquet(pq)
    return df


def load_train():
    X = _read("X_train*.csv", "X_train.parquet")
    y = _read("y_train*.csv", "y_train.parquet")["target"].reindex(X.index)
    return X, y


def load_test():
    return _read("X_test*.csv", "X_test.parquet")
