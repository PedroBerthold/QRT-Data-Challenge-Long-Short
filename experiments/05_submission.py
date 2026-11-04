"""Step 5: train on the full train set and write a submission file (ROW_ID, target in {0,1}).
Set BLOCKS / USE_TE / BINARY to the best configuration found in steps 2-4."""
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from _common import *  # noqa
from src.data import load_train, load_test
from src.features import build, target_encoding
from src.cv import LGB_PARAMS

BLOCKS = ("raw", "benchmark")
USE_TE = False
BINARY = False
NAME = "lgbm_benchmark"

X, y = load_train()
Xt = load_test()
F, Ft = build(X, BLOCKS), build(Xt, BLOCKS)
if USE_TE:
    # train rows: leave-one-segment-group-out would be cleaner; full-sample TE is fine for the test side
    from sklearn.model_selection import GroupKFold
    from src.segments import load_segments
    seg_id = X.TS.map(load_segments()).values
    te_tr = pd.DataFrame(index=X.index, columns=["TE_POS", "TE_MEAN"], dtype=float)
    for a, b in GroupKFold(5).split(X, groups=seg_id):
        te_tr.iloc[b] = target_encoding(X.ALLOCATION.values[a], y.values[a], X.ALLOCATION.values[b]).values
    F = pd.concat([F, te_tr], axis=1)
    Ft = pd.concat([Ft, target_encoding(X.ALLOCATION, y, Xt.ALLOCATION).set_index(Xt.index)], axis=1)

params = {**LGB_PARAMS, "objective": "binary" if BINARY else "mse"}
label = (y > 0).astype(int) if BINARY else y
model = lgb.train(params, lgb.Dataset(F, label), 500)
pred = model.predict(Ft[F.columns])
sub = pd.DataFrame({"target": (pred > (0.5 if BINARY else 0)).astype(int)}, index=Xt.index)

out = Path(__file__).resolve().parents[1] / "submissions"
out.mkdir(exist_ok=True)
sub.to_csv(out / f"{NAME}.csv")
print(f"Saved submissions/{NAME}.csv  ({len(sub)} rows, share of 1: {sub.target.mean():.3f})")
