"""Cross-validation by groups of dates. All rows of a date stay in the same fold."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold


def date_folds(ts: pd.Series, n_splits=5, seed=0):
    """Random KFold on dates (what the benchmark does)."""
    ud = ts.unique()
    out = []
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(ud):
        out.append((np.where(ts.isin(ud[a]))[0], np.where(ts.isin(ud[b]))[0]))
    return out


def segment_folds(ts: pd.Series, seg: pd.Series, n_splits=5):
    """GroupKFold on chronological segments (block time CV, stricter)."""
    groups = ts.map(seg).values
    return list(GroupKFold(n_splits).split(np.zeros(len(ts)), groups=groups))


LGB_PARAMS = dict(objective="mse", learning_rate=0.01, max_depth=3,
                  num_threads=-1, verbosity=-1, seed=42)


def accuracy(pred, y, thr=0.0):
    return float(((pred > thr) == (y > 0)).mean())


def cv_lgbm(F: pd.DataFrame, y: pd.Series, cols, folds, params=None,
            rounds=500, binary=False, fold_features=None, label=""):
    """Train one LightGBM per fold, return (mean acc, std acc, oof predictions).

    fold_features: optional function (train_idx, valid_idx) -> (F_train_extra, F_valid_extra)
    to add features that must be computed inside the fold (e.g. target encoding).
    """
    p = {**LGB_PARAMS, **(params or {})}
    if binary:
        p["objective"] = "binary"
    yv = y.values
    oof = np.zeros(len(y))
    scores = []
    for tr, va in folds:
        Ftr, Fva = F.iloc[tr][cols], F.iloc[va][cols]
        if fold_features is not None:
            etr, eva = fold_features(tr, va)
            Ftr = pd.concat([Ftr, etr], axis=1)
            Fva = pd.concat([Fva, eva], axis=1)
        lab = (yv[tr] > 0).astype(int) if binary else yv[tr]
        model = lgb.train(p, lgb.Dataset(Ftr, lab), rounds)
        pred = model.predict(Fva)
        oof[va] = pred
        scores.append(accuracy(pred, yv[va], 0.5 if binary else 0.0))
    m, s = np.mean(scores) * 100, np.std(scores) * 100
    print(f"{label:45s} acc {m:.2f} ± {s:.2f}  folds {[round(x * 100, 2) for x in scores]}", flush=True)
    return m, s, oof
