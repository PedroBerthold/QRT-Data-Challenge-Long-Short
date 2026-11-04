"""Step 7: separate the market move from the allocation-specific move.

1) Market model: one row per date (~2,500 rows), predicts the average next-day return of all
   allocations. Diagnostic: how often it gets the sign of the day right, and what accuracy
   we would get on rows by predicting the market sign for every allocation.
2) V1 additive: total = beta_alloc * market_pred + relative_pred, where the relative model
   predicts (return - market mean) / vol from allocation features + cross-section + TE.
3) V2 stacked: model B (binary + TE) with the market prediction (and market_pred x beta) added
   as features. Market predictions for training rows are out-of-fold (no leakage).
Each variant is compared with B.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from _common import *  # noqa
from sklearn.model_selection import GroupKFold
from src.features import build, make_te_fold_features
from src.cv import segment_folds, cv_lgbm, CVResult, accuracy, LGB_PARAMS
from src.segments import load_segments
from src.market import date_features, date_target, market_predictions
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "07_two_stage"
MKT_KIND = "lgbm"   # or "ridge"

X, y = train_data()
seg = load_segments()
folds = segment_folds(X.TS, seg)
print("Building features...", flush=True)
F = build(X, ("raw", "benchmark", "cross_section"))
base_cols = [c for c in build(X.iloc[:5], ("raw", "benchmark")).columns]
te = make_te_fold_features(X, y, seg)
D = date_features(X)
yD = date_target(X, y)
ts = X.TS.values
beta = F["BETA_MKT"].values
vol = F["STD_PERF_20"].clip(lower=F["STD_PERF_20"].quantile(0.01)).values
date_mean_y = pd.Series(ts).map(yD).values
y_rel = np.clip((y.values - date_mean_y) / vol, -5, 5)

# Market predictions for every fold (validation dates + OOF for training dates)
print(f"Market model ({MKT_KIND}) on {len(D):,} dates...", flush=True)
mkt = []
date_acc = []
for tr, va in folds:
    d_tr, d_va = np.unique(ts[tr]), np.unique(ts[va])
    p_tr, p_va = market_predictions(D, yD, d_tr, d_va, seg, MKT_KIND)
    mkt.append((pd.Series(ts[tr]).map(p_tr).values, pd.Series(ts[va]).map(p_va).values))
    date_acc.append(((p_va > 0) == (yD.loc[d_va] > 0)).mean())
print(f"Market model: sign of the day right on {np.mean(date_acc) * 100:.2f}% of validation dates "
      f"{[round(a * 100, 1) for a in date_acc]}")
mkt_only = [accuracy(m_va, y.values[va]) for (m_tr, m_va), (tr, va) in zip(mkt, folds)]
print(f"Predicting the market sign for every allocation: {np.mean(mkt_only) * 100:.2f}% "
      f"{[round(a * 100, 2) for a in mkt_only]}")

# Reference: B
B = cv_lgbm(F, y, base_cols, folds, rounds=ROUNDS, binary=True, fold_features=te, label="B: binary + TE",
            shap_sample=0)

# V1 additive
print("[V1 additive] training 5 folds...", flush=True)
oof, scores = np.zeros(len(y)), []
rel_cols = list(F.columns)
for k, ((tr, va), (m_tr, m_va)) in enumerate(zip(folds, mkt)):
    etr, eva = te(tr, va)
    Ftr = pd.concat([F.iloc[tr][rel_cols], etr], axis=1)
    Fva = pd.concat([F.iloc[va][rel_cols], eva], axis=1)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(Ftr, y_rel[tr]), ROUNDS)
    rel_pred = model.predict(Fva) * vol[va]
    oof[va] = beta[va] * m_va + rel_pred
    scores.append(accuracy(oof[va], y.values[va]))
    print(f"  fold {k + 1}/5: acc {scores[-1] * 100:.2f}%", flush=True)
V1 = CVResult("V1: market x beta + relative", scores, oof, 0.0, folds=folds)
print(f"{V1.label:45s} acc {V1.mean:.2f} ± {V1.std:.2f}  folds {[round(s * 100, 2) for s in scores]}")


# V2 stacked: B + market prediction features
fold_map = {id(f[0]): m for f, m in zip(folds, mkt)}


def te_plus_market(tr, va):
    etr, eva = te(tr, va)
    m_tr, m_va = fold_map[id(tr)]
    etr = etr.assign(MKT_PRED=m_tr, MKT_PRED_X_BETA=m_tr * beta[tr])
    eva = eva.assign(MKT_PRED=m_va, MKT_PRED_X_BETA=m_va * beta[va])
    return etr, eva


V2 = cv_lgbm(F, y, base_cols + ["BETA_MKT"], folds, rounds=ROUNDS, binary=True, fold_features=te_plus_market,
             label="V2: B + market prediction", shap_sample=20000)

for r in (B, V1, V2):
    if not QUICK:
        log_result(EXP, "segment", r.mean, r.std, len(base_cols), r.label)
print("\nComparisons with B:")
compare(B, V1, y, X.TS)
compare(B, V2, y, X.TS)

plots.fold_scores([B, V1, V2], EXP)
plots.importance(V2, EXP, kind="shap")
plots.market_timing(V2, y, X, EXP)
plots.show()
