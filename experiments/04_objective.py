"""Step 4: model settings. Base = best set from step 3 (raw + benchmark + target encoding).

Hypotheses tested (each compared with the base with the paired test):
A. Target scaling: the base regresses the raw return, so volatile allocations dominate the
   squared error while accuracy counts every row equally. Divide the target by the allocation's
   20-day volatility (and clip outliers) so every row weighs the same.
B. Binary objective (predict the sign directly).
C. Capacity: the base is very constrained (depth 3, lr 0.01, 500 trees). Try more trees / deeper trees.
D. Stronger regularisation.
E. A linear model (logistic regression) for comparison, with its coefficients.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from _common import *  # noqa
from src.features import build, make_te_fold_features
from src.cv import segment_folds, cv_lgbm, CVResult, accuracy
from src.segments import load_segments
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "04_objective"
BLOCKS = ("raw", "benchmark")
USE_TE = True

X, y = train_data()
seg = load_segments()
folds = segment_folds(X.TS, seg)
F = build(X, BLOCKS)
cols = list(F.columns)
te = make_te_fold_features(X, y, seg) if USE_TE else None

vol = F["STD_PERF_20"].clip(lower=F["STD_PERF_20"].quantile(0.01)).values
y_scaled = np.clip(y.values / vol, -5, 5)

REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)
runs = [
    ("base: mse raw target", dict(), False, None, ROUNDS),
    ("A: mse target / vol", dict(), False, y_scaled, ROUNDS),
    ("B: binary", dict(), True, None, ROUNDS),
    (f"C1: target / vol, {ROUNDS * 3} trees", dict(), False, y_scaled, ROUNDS * 3),
    (f"C2: target / vol, depth 5, {ROUNDS * 2} trees", dict(max_depth=5, num_leaves=31), False, y_scaled, ROUNDS * 2),
    (f"D: target / vol, strong reg, {ROUNDS * 2} trees", REG, False, y_scaled, ROUNDS * 2),
]
results = []
for name, params, binary, target, rounds in runs:
    r = cv_lgbm(F, y, cols, folds, params=params, rounds=rounds, binary=binary,
                fold_features=te, train_target=target, label=name, shap_sample=0)
    results.append(r)

# E. linear model
oof, scores, coefs = np.zeros(len(y)), [], []
print("[E: logistic] training 5 folds...", flush=True)
for tr, va in folds:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.01, max_iter=500))
    model.fit(F.iloc[tr][cols], (y.values[tr] > 0).astype(int))
    p = model.predict_proba(F.iloc[va][cols])[:, 1]
    oof[va] = p
    scores.append(accuracy(p, y.values[va], 0.5))
    coefs.append(model[-1].coef_[0])
lin = CVResult("E: logistic (C=0.01)", scores, oof, 0.5, folds=folds)
print(f"{lin.label:45s} acc {lin.mean:.2f} ± {lin.std:.2f}  folds {[round(x * 100, 2) for x in scores]}")
results.append(lin)

for r in results:
    if not QUICK:
        log_result(EXP, "segment", r.mean, r.std, len(cols), r.label + (" +TE" if USE_TE else ""))
print("\nEach setting vs the base:")
for r in results[1:]:
    compare(results[0], r, y, X.TS)

plots.fold_scores(results, EXP)
plots.coefficients(pd.Series(np.mean(coefs, 0), index=cols), "logistic regression coefficients", EXP)
plots.confidence(max(results, key=lambda r: r.mean), y, EXP)
plots.show()
