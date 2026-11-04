"""Step 4: which model / objective? Set BLOCKS to the best set from step 3.
- LightGBM regression on the return (MSE) vs classification of the sign (binary)
- stronger regularisation
- a linear model (logistic regression on standardised features), with its coefficients
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from _common import *  # noqa
from src.features import build
from src.cv import segment_folds, cv_lgbm, CVResult, accuracy
from src.segments import load_segments
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "04_objective"
BLOCKS = ("raw", "benchmark")
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)

X, y = train_data()
folds = segment_folds(X.TS, load_segments())
F = build(X, BLOCKS)
cols = list(F.columns)

results = [
    cv_lgbm(F, y, cols, folds, rounds=ROUNDS, label="lgbm mse"),
    cv_lgbm(F, y, cols, folds, rounds=ROUNDS, binary=True, label="lgbm binary"),
    cv_lgbm(F, y, cols, folds, params=REG, rounds=ROUNDS, label="lgbm mse strong reg"),
    cv_lgbm(F, y, cols, folds, params=REG, rounds=ROUNDS, binary=True, label="lgbm binary strong reg"),
]

# Linear model
oof, scores, coefs = np.zeros(len(y)), [], []
for tr, va in folds:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.01, max_iter=500))
    model.fit(F.iloc[tr][cols], (y.values[tr] > 0).astype(int))
    p = model.predict_proba(F.iloc[va][cols])[:, 1]
    oof[va] = p
    scores.append(accuracy(p, y.values[va], 0.5))
    coefs.append(model[-1].coef_[0])
lin = CVResult("logistic (C=0.01)", scores, oof, 0.5, folds=folds)
print(f"{lin.label:45s} acc {lin.mean:.2f} ± {lin.std:.2f}  folds {[round(x * 100, 2) for x in scores]}")
results.append(lin)

for r in results:
    if not QUICK:
        log_result(EXP, "segment", r.mean, r.std, len(cols), r.label)
print("\nEach model vs lgbm mse:")
for r in results[1:]:
    compare(results[0], r, y, X.TS)

plots.fold_scores(results, EXP)
plots.coefficients(pd.Series(np.mean(coefs, 0), index=cols), "logistic regression coefficients", EXP)
plots.confidence(lin, y, EXP)
plots.show()
