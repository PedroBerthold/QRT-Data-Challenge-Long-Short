"""Feature blocks. Each function takes the raw X (train or test) and returns a DataFrame."""
import numpy as np
import pandas as pd
from .data import RET, VOL


def raw(X):
    return X[RET + VOL + ["MEDIAN_DAILY_TURNOVER"]].copy()


def benchmark(X):
    """Same features as the QRT benchmark notebook."""
    o = pd.DataFrame(index=X.index)
    for k in [3, 5, 10, 15, 20]:
        o[f"AVERAGE_PERF_{k}"] = X[RET[:k]].mean(1)
        o[f"ALLOCATIONS_AVERAGE_PERF_{k}"] = o[f"AVERAGE_PERF_{k}"].groupby(X.TS).transform("mean")
    o["STD_PERF_20"] = X[RET].std(1)
    o["ALLOCATIONS_STD_PERF_20"] = o["STD_PERF_20"].groupby(X.TS).transform("mean")
    return o


def time_series(X):
    """Per-row features from the 20-day history: risk-adjusted momentum, shape, volumes."""
    r = X[RET].values
    v = X[VOL].values
    vol = np.nanstd(r, 1) + 1e-6
    o = {"VOL20": vol}
    for k in [1, 2, 3, 5, 10, 20]:
        o[f"SR_{k}"] = np.nanmean(r[:, :k], 1) / vol
    o["UPFRAC_5"] = (r[:, :5] > 0).mean(1)
    o["UPFRAC_20"] = (r > 0).mean(1)
    o["SKEW_20"] = X[RET].skew(1).values
    cum = np.nancumsum(r[:, ::-1], 1)          # oldest -> newest
    o["MAX_DD_20"] = (cum - np.maximum.accumulate(cum, 1)).min(1)
    o["SV1_MISSING"] = np.isnan(v[:, 0]).astype(float)   # SIGNED_VOLUME_1 is ~73% missing
    o["SV_MEAN_5"] = np.nanmean(v[:, 1:6], 1)
    o["SV_MEAN_20"] = np.nanmean(v, 1)
    o["SV_Z_2"] = (v[:, 1] - o["SV_MEAN_20"]) / (np.nanstd(v, 1) + 1e-9)
    return pd.DataFrame(o, index=X.index)


def cross_section(X):
    """Features relative to the other allocations of the same date (and same GROUP)."""
    g = X.TS
    o = pd.DataFrame(index=X.index)
    r = X[RET].values
    sr5 = np.nanmean(r[:, :5], 1) / (np.nanstd(r, 1) + 1e-6)
    tmp = pd.DataFrame({"RET_1": X.RET_1, "RET_2": X.RET_2, "SR_5": sr5}, index=X.index)
    for c in tmp.columns:
        mean = tmp[c].groupby(g).transform("mean")
        std = tmp[c].groupby(g).transform("std")
        o[f"CS_MEAN_{c}"] = mean
        o[f"CS_Z_{c}"] = (tmp[c] - mean) / (std + 1e-9)
        o[f"CS_RANK_{c}"] = tmp[c].groupby(g).rank(pct=True)
        o[f"GRP_MEAN_{c}"] = tmp[c].groupby([g, X.GROUP]).transform("mean")
    # beta of the allocation to the "market" = average allocation path that day
    mkt = X[RET].groupby(g).transform("mean").values
    a = r - np.nanmean(r, 1, keepdims=True)
    b = mkt - np.nanmean(mkt, 1, keepdims=True)
    o["BETA_MKT"] = np.nansum(a * b, 1) / (np.nansum(b * b, 1) + 1e-12)
    o["CORR_MKT"] = np.nansum(a * b, 1) / np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1) + 1e-12)
    o["N_ALLOC_DATE"] = g.map(g.value_counts())
    return o


def target_encoding(alloc_fit, y_fit, alloc_apply, prior=200):
    """Shrunk per-allocation stats of the target, fitted on (alloc_fit, y_fit) only.

    TE_POS  = share of positive next-day returns
    TE_MEAN = mean next-day return
    prior = number of pseudo-observations pulling towards the global mean.
    """
    d = pd.DataFrame({"a": np.asarray(alloc_fit), "y": np.asarray(y_fit)})
    d["p"] = (d.y > 0).astype(float)
    g = d.groupby("a")
    n = g.size()
    te_pos = (g.p.sum() + prior * d.p.mean()) / (n + prior)
    te_mean = (g.y.sum() + prior * d.y.mean()) / (n + prior)
    a = pd.Series(np.asarray(alloc_apply))
    return pd.DataFrame({"TE_POS": a.map(te_pos).fillna(d.p.mean()).values,
                         "TE_MEAN": a.map(te_mean).fillna(d.y.mean()).values})


BLOCKS = {"raw": raw, "benchmark": benchmark, "time_series": time_series,
          "cross_section": cross_section}


def build(X, blocks=("raw", "benchmark")):
    return pd.concat([BLOCKS[b](X) for b in blocks], axis=1)


def make_te_fold_features(X, y, seg, inner_splits=4):
    """Return a fold_features(tr, va) function for cv_lgbm that adds target encoding.

    Validation rows: encoding fitted on all training rows of the fold.
    Training rows: inner out-of-fold encoding (grouped by segment), so a row never
    sees its own target.
    """
    from sklearn.model_selection import GroupKFold
    alloc = X.ALLOCATION.values
    yv = np.asarray(y)
    seg_id = X.TS.map(seg).values

    def fold_features(tr, va):
        te_va = target_encoding(alloc[tr], yv[tr], alloc[va])
        te_tr = pd.DataFrame(np.nan, index=range(len(tr)), columns=te_va.columns)
        for a, b in GroupKFold(inner_splits).split(tr, groups=seg_id[tr]):
            te_tr.iloc[b] = target_encoding(alloc[tr[a]], yv[tr[a]], alloc[tr[b]]).values
        te_tr.index = X.index[tr]
        te_va.index = X.index[va]
        return te_tr, te_va

    return fold_features
