"""Shared setup for the leak scripts (10-12). Matches are cached in data/ (slow to compute)."""
import os
import numpy as np
import pandas as pd
from _common import *  # noqa
from src.data import DATA_DIR, load_train, load_test
from src.matching import RET_COLS, match

SHIFTS = range(1, 11)
MAX_LEN = 19   # long windows separate true matches (~0.99) from chance (~0.85) much better than 10 lags


def pool():
    """All rows of train + test: the place where the 'future' rows are searched."""
    X, y = load_train()
    Xt = load_test()
    P = pd.concat([X.assign(SRC="train"), Xt.assign(SRC="test")])
    return X, y, Xt, P


def cached_match(name, Q_df, P, use_group=False):
    path = DATA_DIR / f"matches_{name}{'_grp' if use_group else ''}.npz"
    if path.exists() and os.environ.get("RECOMPUTE") != "1":
        z = np.load(path)
        return {k: dict(idx=z[f"idx{k}"], sim=z[f"sim{k}"], sim2=z[f"sim2{k}"], read=z[f"read{k}"])
                for k in SHIFTS}
    print(f"Matching {len(Q_df):,} queries against {len(P):,} pool rows (cached afterwards in {path.name})")
    m = match(Q_df[RET_COLS].values, P[RET_COLS].values, SHIFTS, MAX_LEN,
              q_index=Q_df.index.values, p_index=P.index.values,
              q_group=Q_df.GROUP.values if use_group else None,
              p_group=P.GROUP.values if use_group else None)
    np.savez(path, **{f"{key}{k}": m[k][key] for k in SHIFTS for key in ("idx", "sim", "sim2", "read")})
    return m
