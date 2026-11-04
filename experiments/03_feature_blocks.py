"""Step 3: ablation, add one feature block at a time (segment CV).
A gain below ~0.3 pt is noise: look at the fold-by-fold scores, not only the mean.
Edit USE_TE once step 2 tells you whether target encoding helps.
"""
from _common import *  # noqa
from src.features import build, make_te_fold_features
from src.cv import segment_folds, cv_lgbm
from src.segments import load_segments
from src.log import log_result

USE_TE = False

X, y = train_data()
seg = load_segments()
folds = segment_folds(X.TS, seg)

fold_features = make_te_fold_features(X, y, seg) if USE_TE else None

configs = [
    ("raw", "benchmark"),
    ("raw", "benchmark", "time_series"),
    ("raw", "benchmark", "cross_section"),
    ("raw", "benchmark", "time_series", "cross_section"),
    ("benchmark", "time_series", "cross_section"),   # without the 40 raw lags
]
for blocks in configs:
    F = build(X, blocks)
    name = "+".join(blocks)
    m, s, _ = cv_lgbm(F, y, list(F.columns), folds, rounds=ROUNDS,
                      fold_features=fold_features, label=name)
    if not QUICK:
        log_result("03_blocks", "segment", m, s, F.shape[1], name + (" +TE" if USE_TE else ""))
