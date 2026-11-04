"""Is model B really better than model A? Tools to separate signal from noise."""
from math import comb
import numpy as np
import pandas as pd


def per_date_accuracy(pred_label, y, ts):
    return pd.Series((np.asarray(pred_label) == (np.asarray(y) > 0)), index=np.asarray(ts)).groupby(level=0).mean()


def compare(res_a, res_b, y, ts):
    """Paired comparison of two CVResults evaluated on the same rows.

    - fold wins: on how many folds B beats A, and the probability of doing at least as well by pure chance
    - paired t-stat on per-date accuracy differences (each date = one observation, ~2500 dates)
    """
    wins = int(sum(b > a for a, b in zip(res_a.scores, res_b.scores)))
    n = len(res_a.scores)
    p_folds = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
    da = per_date_accuracy(res_a.pred_label, y, ts)
    db = per_date_accuracy(res_b.pred_label, y, ts)
    diff = (db - da).dropna()
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    out = dict(delta_pts=round((res_b.mean - res_a.mean), 3), fold_wins=f"{wins}/{n}",
               p_folds_by_chance=round(p_folds, 3), t_stat_per_date=round(float(t), 2))
    print(f"{res_b.label} vs {res_a.label}: {out}")
    return out, diff
