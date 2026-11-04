"""Shared setup for experiment scripts. Run scripts from the repo root:
    python experiments/01_benchmark.py
Set QUICK=1 to run on 20% of dates and 100 trees (smoke test, ~1 min).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_train, load_test  # noqa: E402

QUICK = os.environ.get("QUICK") == "1"
ROUNDS = 100 if QUICK else 500


def train_data():
    X, y = load_train()
    if QUICK:
        keep = X.TS.drop_duplicates().sample(frac=0.2, random_state=0)
        mask = X.TS.isin(keep)
        X, y = X[mask], y[mask]
    return X, y
