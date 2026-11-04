"""Step 2: does each allocation have a persistent drift?

a) Persistence: correlate per-allocation mean return between two disjoint halves of the segments.
b) Pure rule: predict 1 if the allocation's historical share of positive days > 50%.
c) Benchmark features + target encoding in LightGBM, compared with the benchmark alone.

Target encoding is fitted ONLY on the training folds (inner out-of-fold for training rows).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import *  # noqa
from src.features import build, target_encoding, make_te_fold_features
from src.cv import segment_folds, cv_lgbm, accuracy
from src.segments import load_segments
from src.stats import compare
from src.log import log_result
from src import plots

EXP = "02_target_encoding"
X, y = train_data()
seg = load_segments()
seg_id = X.TS.map(seg).values
alloc = X.ALLOCATION.values
yv = y.values

# a) persistence
d = pd.DataFrame({"a": alloc, "y": yv, "h": seg_id % 2})
mean_h = d.groupby(["a", "h"]).y.mean().unstack()
pos_h = d.assign(p=d.y > 0).groupby(["a", "h"]).p.mean().unstack()
print(f"Per-allocation mean return, corr between halves: {mean_h.corr().iloc[0, 1]:.3f}")
print(f"Per-allocation share of positive days, corr between halves: {pos_h.corr().iloc[0, 1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(mean_h[0] * 1e4, mean_h[1] * 1e4, s=10)
ax[0].set(title=f"Mean return per allocation (corr {mean_h.corr().iloc[0, 1]:.2f})",
          xlabel="half A (bp)", ylabel="half B (bp)")
ax[1].scatter(pos_h[0], pos_h[1], s=10, color="#55a868")
ax[1].set(title=f"Share of positive days (corr {pos_h.corr().iloc[0, 1]:.2f})", xlabel="half A", ylabel="half B")
plots._save(fig, EXP, "persistence")

folds = segment_folds(X.TS, seg)

# b) pure rule
scores = []
for tr, va in folds:
    te = target_encoding(alloc[tr], yv[tr], alloc[va])
    scores.append(accuracy(te.TE_POS.values, yv[va], thr=0.5))
m, s = np.mean(scores) * 100, np.std(scores) * 100
print(f"{'rule: TE_POS > 0.5 | segment CV':45s} acc {m:.2f} ± {s:.2f}  folds {[round(x * 100, 2) for x in scores]}")
if not QUICK:
    log_result("02_rule_TE_POS", "segment", m, s, 1)

# c) benchmark vs benchmark + TE
F = build(X, ("raw", "benchmark"))
cols = list(F.columns)
res_a = cv_lgbm(F, y, cols, folds, rounds=ROUNDS, label="benchmark")
res_b = cv_lgbm(F, y, cols, folds, rounds=ROUNDS, fold_features=make_te_fold_features(X, y, seg),
                label="benchmark + TE")
if not QUICK:
    log_result("02_benchmark_TE", "segment", res_b.mean, res_b.std, len(cols) + 2)

_, diff = compare(res_a, res_b, y, X.TS)
plots.compare_dates(diff, "benchmark + TE vs benchmark", EXP)
plots.fold_scores([res_a, res_b], EXP)
plots.importance(res_b, EXP, kind="shap")
plots.show()
