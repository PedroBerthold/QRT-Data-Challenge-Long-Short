"""Step 2: does each allocation have a persistent drift?

a) Persistence check: correlate per-allocation mean return between two disjoint halves of the segments.
b) Pure rule: predict sign = 1 if the allocation's historical share of positive days > 50%.
c) Benchmark features + target encoding inside LightGBM.

Target encoding must be fitted ONLY on the training folds. For training rows we use an
inner out-of-fold encoding, otherwise the model sees its own target and overfits.
"""
import numpy as np
import pandas as pd
from _common import *  # noqa
from src.features import build, target_encoding, make_te_fold_features
from src.cv import segment_folds, cv_lgbm, accuracy
from src.segments import load_segments
from src.log import log_result

X, y = train_data()
seg = load_segments()
seg_id = X.TS.map(seg).values
alloc = X.ALLOCATION.values
yv = y.values

# a) persistence
half = seg_id % 2
d = pd.DataFrame({"a": alloc, "y": yv, "h": half})
stats = d.groupby(["a", "h"]).y.mean().unstack()
print(f"Per-allocation mean return, corr between halves: {stats.corr().iloc[0, 1]:.3f}")
pos = d.assign(p=d.y > 0).groupby(["a", "h"]).p.mean().unstack()
print(f"Share of positive days per allocation: min {pos[0].min():.3f}, max {pos[0].max():.3f}")

folds = segment_folds(X.TS, seg)

# b) pure rule
scores = []
for tr, va in folds:
    te = target_encoding(alloc[tr], yv[tr], alloc[va])
    scores.append(accuracy(te.TE_POS.values, yv[va], thr=0.5))
m, s = np.mean(scores) * 100, np.std(scores) * 100
print(f"{'rule: TE_POS > 0.5 | segment CV':45s} acc {m:.2f} ± {s:.2f}")
if not QUICK:
    log_result("02_rule_TE_POS", "segment", m, s, 1)


# c) LightGBM with TE
F = build(X, ("raw", "benchmark"))
cols = list(F.columns)
m, s, _ = cv_lgbm(F, y, cols, folds, rounds=ROUNDS,
                  fold_features=make_te_fold_features(X, y, seg),
                  label="benchmark + TE | segment CV")
if not QUICK:
    log_result("02_benchmark_TE", "segment", m, s, len(cols) + 2)
