"""Step 5: train on the full train set and write a submission (ROW_ID, target in {0,1}).

MODEL = "B"        binary objective + target encoding (best single model, step 4)
MODEL = "D"        target / volatility + strong regularisation + target encoding
MODEL = "ensemble" average of B and D (step 6)
MODEL = "benchmark" the QRT benchmark (for reference)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from _common import *  # noqa
from src.data import load_train, load_test
from src.features import build, target_encoding
from src.cv import LGB_PARAMS
from src.segments import load_segments
from src.ensemble import combine

MODEL = "ensemble"
ROUNDS_FULL = 500
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)

X, y = load_train()
Xt = load_test()
print(f"Train {len(X):,} rows, test {len(Xt):,} rows. Building features...", flush=True)
F, Ft = build(X, ("raw", "benchmark")), build(Xt, ("raw", "benchmark"))

if MODEL != "benchmark":
    # Train rows: out-of-fold encoding (grouped by segment) so a row never sees its own target.
    seg_id = X.TS.map(load_segments()).values
    te_tr = pd.DataFrame(np.nan, index=X.index, columns=["TE_POS", "TE_MEAN"])
    for a, b in GroupKFold(5).split(X, groups=seg_id):
        te_tr.iloc[b] = target_encoding(X.ALLOCATION.values[a], y.values[a], X.ALLOCATION.values[b]).values
    # Test rows: encoding fitted on the whole train set.
    te_te = target_encoding(X.ALLOCATION, y, Xt.ALLOCATION).set_index(Xt.index)
    F, Ft = pd.concat([F, te_tr], axis=1), pd.concat([Ft, te_te], axis=1)
Ft = Ft[F.columns]


def fit_predict(params, label, rounds, name):
    print(f"Training {name}...", flush=True)
    model = lgb.train({**LGB_PARAMS, **params}, lgb.Dataset(F, label), rounds)
    return model.predict(Ft)


if MODEL == "benchmark":
    signal = fit_predict({}, y, ROUNDS_FULL, "benchmark")
else:
    vol = F["STD_PERF_20"].clip(lower=F["STD_PERF_20"].quantile(0.01)).values
    y_scaled = np.clip(y.values / vol, -5, 5)
    preds, centers = [], []
    if MODEL in ("B", "ensemble"):
        preds.append(fit_predict({"objective": "binary"}, (y > 0).astype(int), ROUNDS_FULL, "B (binary)"))
        centers.append(0.5)
    if MODEL in ("D", "ensemble"):
        preds.append(fit_predict(REG, y_scaled, ROUNDS_FULL * 2, "D (target/vol, strong reg)"))
        centers.append(0.0)
    signal = combine(preds, centers)

sub = pd.DataFrame({"target": (signal > 0).astype(int)}, index=Xt.index)
out = Path(__file__).resolve().parents[1] / "submissions"
out.mkdir(exist_ok=True)
sub.to_csv(out / f"{MODEL}.csv")
print(f"Saved submissions/{MODEL}.csv  ({len(sub)} rows, share of 1: {sub.target.mean():.3f})")
