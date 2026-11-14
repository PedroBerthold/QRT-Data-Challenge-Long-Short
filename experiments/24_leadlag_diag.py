"""Step 24 (diagnostic): where does the RET_1 effect come from? Is next-day performance of a
group or of a statistical factor predictable from yesterday's performance of the others?

Date-level only (one observation per date), so every number here has ~2,500 observations.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from _common import *  # noqa
from src.data import load_train, RET
from src.segments import load_segments

X, y = load_train()
seg = X.TS.map(load_segments())
yb = y > 0
print("trivial rules, accuracy on all train rows:")
for name, s in (("always up", np.ones(len(X))), ("sign(RET_1)", X.RET_1), ("sign(mean RET_1..3)", X[RET[:3]].mean(axis=1)),
                ("sign(mean RET_1..20)", X[RET].mean(axis=1)), ("-sign(RET_2)", -X.RET_2)):
    print(f"  {name:22s} {((np.asarray(s) > 0) == yb.values).mean() * 100:.2f}")

# group-level lead-lag: corr( mean target of group g , mean RET_j of group h ) across dates
gt = y.groupby([X.TS, X.GROUP]).mean().unstack()
for j in (1, 2):
    gr = X[f"RET_{j}"].groupby([X.TS, X.GROUP]).mean().unstack()
    C = pd.DataFrame({h: [gt[g].corr(gr[h]) for g in gt.columns] for h in gr.columns}, index=[f"target G{g}" for g in gt.columns])
    C.columns = [f"RET_{j} G{h}" for h in gr.columns]
    print(f"\ncorrelation across dates, group mean target vs group mean RET_{j}:\n{C.round(3).to_string()}")
print("\nsame-day correlation between group mean targets:\n", gt.corr().round(2).to_string())
print("std of group mean target (bp):", (gt.std() * 1e4).round(2).to_dict(),
      "| std of a single allocation target (bp): %.1f" % (y.std() * 1e4))

# statistical factors: fixed loadings over allocations (PCA of the target cross-sections), next-day factor
# return regressed on yesterday's factor returns, out of sample by segment
Y = y.groupby([X.TS, X.ALLOCATION]).mean().unstack()
R1 = X.RET_1.groupby([X.TS, X.ALLOCATION]).mean().unstack().reindex_like(Y)
R2 = X.RET_2.groupby([X.TS, X.ALLOCATION]).mean().unstack().reindex_like(Y)
full = Y.notna().mean(axis=1) > 0.95                     # dates where the whole universe trades
Yf, R1f, R2f = Y[full].fillna(0), R1[full].fillna(0), R2[full].fillna(0)
sd = Yf.std() + 1e-12
g = seg.groupby(X.TS).first().reindex(Yf.index).values
K = 12
r2 = np.zeros((K, 3)); acc = np.zeros(K); n = 0
for a, b in GroupKFold(5).split(Yf, groups=g):
    U, S, Vt = np.linalg.svd((Yf.iloc[a] / sd).values, full_matrices=False)
    V = Vt[:K].T
    fy, f1, f2 = (Yf / sd).values @ V, (R1f / sd).values @ V, (R2f / sd).values @ V
    Z = np.c_[f1, f2]
    for k in range(K):
        own = RidgeCV(alphas=[1, 10, 100, 1000]).fit(f1[a][:, [k]], fy[a, k]).predict(f1[b][:, [k]])
        allp = RidgeCV(alphas=[1, 10, 100, 1000]).fit(Z[a], fy[a, k]).predict(Z[b])
        tot = ((fy[b, k] - fy[a, k].mean()) ** 2).sum()
        r2[k, 0] += 1 - ((fy[b, k] - own) ** 2).sum() / tot
        r2[k, 1] += 1 - ((fy[b, k] - allp) ** 2).sum() / tot
        r2[k, 2] += (S[k] ** 2) / (S ** 2).sum()
        acc[k] += ((allp > 0) == (fy[b, k] > 0)).mean()
print(f"\nfactors on the {full.sum()} full-universe dates (out-of-sample by segment, mean of 5 folds):")
print(pd.DataFrame({"var_share": r2[:, 2] / 5, "R2_own_lag1": r2[:, 0] / 5, "R2_all_lags12": r2[:, 1] / 5,
                    "sign_acc": acc / 5}, index=[f"F{k + 1}" for k in range(K)]).round(4).T.to_string())
