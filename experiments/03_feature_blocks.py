"""Step 3: ablation, add one feature block at a time (segment CV), each compared with the benchmark.
A configuration is kept only if it wins on most folds AND the per-date t-stat is > ~2.
Set USE_TE once step 2 tells you whether target encoding helps.
"""
from _common import *  # noqa
from src.features import build, make_te_fold_features
from src.cv import segment_folds, cv_lgbm
from src.segments import load_segments
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "03_feature_blocks"
USE_TE = True

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
results = []
for blocks in configs:
    F = build(X, blocks)
    name = "+".join(blocks)
    res = cv_lgbm(F, y, list(F.columns), folds, rounds=ROUNDS, fold_features=fold_features, label=name)
    results.append(res)
    if not QUICK:
        log_result(EXP, "segment", res.mean, res.std, F.shape[1], name + (" +TE" if USE_TE else ""))

print("\nEach configuration vs the first one (benchmark):")
for res in results[1:]:
    compare(results[0], res, y, X.TS)

best = max(results, key=lambda r: r.mean)
plots.fold_scores(results, EXP)
plots.importance(best, EXP, kind="shap")
plots.confidence(best, y, EXP)
plots.errors_by(best, y, X, seg, EXP)
plots.show()
