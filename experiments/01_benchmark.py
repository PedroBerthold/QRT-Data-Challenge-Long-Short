"""Step 1: reproduce the QRT benchmark and compare two CV schemes (same model, two ways to measure).

Expected: ~52.2% with both, but much larger spread across folds with segment CV.
Figures in figures/01_benchmark/: fold scores, feature importance, confidence, errors, market timing.
"""
from _common import *  # noqa
from src.features import build
from src.cv import date_folds, segment_folds, cv_lgbm
from src.segments import load_segments
from src.log import log_result
from src import plots

EXP = "01_benchmark"
X, y = train_data()
F = build(X, ("raw", "benchmark"))
cols = list(F.columns)
seg = load_segments()

res_date = cv_lgbm(F, y, cols, date_folds(X.TS), rounds=ROUNDS, label="benchmark | date CV")
res_seg = cv_lgbm(F, y, cols, segment_folds(X.TS, seg), rounds=ROUNDS, label="benchmark | segment CV")
if not QUICK:
    log_result(EXP, "date", res_date.mean, res_date.std, len(cols))
    log_result(EXP, "segment", res_seg.mean, res_seg.std, len(cols))

plots.fold_scores([res_date, res_seg], EXP)
plots.importance(res_seg, EXP, kind="shap")
plots.confidence(res_seg, y, EXP)
plots.errors_by(res_seg, y, X, seg, EXP)
plots.market_timing(res_seg, y, X, EXP)
plots.show()
