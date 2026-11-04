"""Step 4: regression on the return (MSE) vs classification of the sign (binary logloss),
plus a couple of regularisation settings. Set BLOCKS to the best set from step 3."""
from _common import *  # noqa
from src.features import build
from src.cv import segment_folds, cv_lgbm
from src.segments import load_segments
from src.log import log_result

BLOCKS = ("raw", "benchmark")

X, y = train_data()
folds = segment_folds(X.TS, load_segments())
F = build(X, BLOCKS)
cols = list(F.columns)

runs = [
    ("mse depth3", dict(), False),
    ("binary depth3", dict(), True),
    ("mse depth3 strong reg", dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1), False),
    ("binary depth3 strong reg", dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1), True),
]
for name, params, binary in runs:
    m, s, _ = cv_lgbm(F, y, cols, folds, params=params, rounds=ROUNDS, binary=binary, label=name)
    if not QUICK:
        log_result("04_objective", "segment", m, s, len(cols), name)
