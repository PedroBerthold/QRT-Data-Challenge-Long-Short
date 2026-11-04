"""Two-stage approach: a date-level 'market' model + an allocation-level 'relative' model.

The target of a row = market move of the day (shared by all allocations) + the allocation's
own deviation. A row-level model sees 527k rows but only ~2,500 independent market moves.
Here the market part is learnt on one row per date.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from .data import RET, VOL

MKT_PARAMS = dict(objective="mse", learning_rate=0.02, max_depth=2, min_data_in_leaf=40,
                  feature_fraction=0.7, num_threads=-1, verbosity=-1, seed=42)
MKT_ROUNDS = 300


def date_features(X):
    """One row per date: cross-sectional averages and dispersion of the history."""
    g = X.groupby("TS")
    D = g[RET].mean().add_prefix("MKT_")
    D["MKT_DISP_1"] = g["RET_1"].std()
    D["MKT_DISP_5"] = X.assign(m5=X[RET[:5]].mean(1)).groupby("TS")["m5"].std()
    D["MKT_VOL_20"] = X.assign(v=X[RET].std(1)).groupby("TS")["v"].mean()
    D["MKT_SV_2"] = g["SIGNED_VOLUME_2"].mean()
    D["MKT_SV_MEAN_5"] = X.assign(s=X[VOL[1:6]].mean(1)).groupby("TS")["s"].mean()
    D["MKT_UPFRAC_1"] = X.assign(u=(X.RET_1 > 0)).groupby("TS")["u"].mean()
    for k in (5, 20):
        D[f"MKT_CUM_{k}"] = D[[f"MKT_RET_{i}" for i in range(1, k + 1)]].sum(1)
        D[f"MKT_SR_{k}"] = D[f"MKT_CUM_{k}"] / (D[[f"MKT_RET_{i}" for i in range(1, 21)]].std(1) + 1e-9)
    for grp in sorted(X.GROUP.unique()):
        sub = X[X.GROUP == grp].groupby("TS")
        for i in (1, 2, 3):
            D[f"G{grp}_RET_{i}"] = sub[f"RET_{i}"].mean()
    D["N_ALLOC"] = g.size()
    return D


def date_target(X, y):
    return pd.Series(np.asarray(y), index=X.index).groupby(X.TS.values).mean()


def fit_market(Dtr, ytr, kind="lgbm"):
    if kind == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
        m.fit(Dtr.fillna(0), ytr)
        return lambda D: m.predict(D.fillna(0))
    m = lgb.train(MKT_PARAMS, lgb.Dataset(Dtr, ytr), MKT_ROUNDS)
    return lambda D: m.predict(D)


def market_predictions(D, yD, dates_tr, dates_va, seg, kind="lgbm", inner_splits=4):
    """Market prediction for validation dates (model fitted on training dates) and
    out-of-fold predictions for training dates (inner CV grouped by segment)."""
    pred_va = pd.Series(fit_market(D.loc[dates_tr], yD.loc[dates_tr], kind)(D.loc[dates_va]), index=dates_va)
    pred_tr = pd.Series(np.nan, index=dates_tr)
    groups = pd.Index(dates_tr).map(seg).values
    for a, b in GroupKFold(inner_splits).split(dates_tr, groups=groups):
        f = fit_market(D.loc[dates_tr[a]], yD.loc[dates_tr[a]], kind)
        pred_tr.iloc[b] = f(D.loc[dates_tr[b]])
    return pred_tr, pred_va
