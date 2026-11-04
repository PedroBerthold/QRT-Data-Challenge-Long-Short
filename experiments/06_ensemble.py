"""Step 6: combine the two best settings of step 4.
B = binary objective + TE
D = target / volatility + strong regularisation + TE
Ensemble = average of their standardised predictions (src/ensemble.combine), computed fold by fold.
"""
import numpy as np
from _common import *  # noqa
from src.features import build, make_te_fold_features
from src.cv import segment_folds, cv_lgbm, CVResult, accuracy
from src.segments import load_segments
from src.ensemble import combine
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "06_ensemble"
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)

X, y = train_data()
seg = load_segments()
folds = segment_folds(X.TS, seg)
F = build(X, ("raw", "benchmark"))
cols = list(F.columns)
te = make_te_fold_features(X, y, seg)
vol = F["STD_PERF_20"].clip(lower=F["STD_PERF_20"].quantile(0.01)).values
y_scaled = np.clip(y.values / vol, -5, 5)

B = cv_lgbm(F, y, cols, folds, rounds=ROUNDS, binary=True, fold_features=te, label="B: binary", shap_sample=0)
D = cv_lgbm(F, y, cols, folds, params=REG, rounds=ROUNDS * 2, fold_features=te, train_target=y_scaled,
            label="D: target/vol strong reg", shap_sample=0)

oof = np.zeros(len(y))
scores = []
for tr, va in folds:
    oof[va] = combine([B.oof[va], D.oof[va]], [0.5, 0.0])
    scores.append(accuracy(oof[va], y.values[va], 0.0))
E = CVResult("Ensemble B+D", scores, oof, 0.0, folds=folds)
print(f"{E.label:45s} acc {E.mean:.2f} ± {E.std:.2f}  folds {[round(s * 100, 2) for s in scores]}")
agree = ((B.oof > 0.5) == (D.oof > 0)).mean()
print(f"B and D agree on {agree:.1%} of rows")

for r in (B, D, E):
    if not QUICK:
        log_result(EXP, "segment", r.mean, r.std, len(cols) + 2, r.label + " +TE")
print("\nComparisons:")
compare(B, E, y, X.TS)
compare(D, E, y, X.TS)

plots.fold_scores([B, D, E], EXP)
plots.confidence(E, y, EXP)
plots.errors_by(E, y, X, seg, EXP)
plots.market_timing(E, y, X, EXP)
plots.show()
