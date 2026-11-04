"""Step 11: run the matching on the TEST rows and check whether they behave like train rows.

Result found so far (sample of 800 rows, 19 lags): test rows reach the same best similarity
as queries that cannot have a true match (median ~0.85, none above 0.95), while train rows
reach ~0.99. The continuation of the test rows is NOT in the files: the leak cannot be used
on the test set. This script lets you confirm it on all test rows.

No truth here, so we look at indirect signals:
- similarity distribution vs train queries (step 10) and vs the null test
- coverage: share of test rows with a confident match, per threshold
- agreement: do the copies from different shifts agree on the sign? (high agreement = reliable)
- where the matches come from (train or test rows), and whether the GROUP is preserved
Also tests H8: restricting matches to the same GROUP (USE_GROUP = True).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _leak_common import *  # noqa
from src.matching import to_frames, aggregate
from src import plots

EXP = "11_leak_test"
USE_GROUP = False

X, y, Xt, P = pool()
Q = Xt.sample(2000, random_state=0) if QUICK else Xt
m = cached_match("test" + ("_quick" if QUICK else ""), Q, P, use_group=USE_GROUP)
ks, read, sim, sim2 = to_frames(m)

from src.matching import match
Qnull = Q[RET_COLS].values[:1000, ::-1].copy()
null_sim = match(Qnull, P[RET_COLS].values, [1], MAX_LEN, verbose=False)[1]["sim"]
print(f"\nNull queries (no true match possible): median best sim {np.median(null_sim):.4f}")
print(f"Test queries, k=1:                     median best sim {np.median(sim[:, 0]):.4f}")
print("\nMedian best similarity per shift:", dict(zip(ks, np.median(sim, 0).round(4))))
src = P.SRC.values[m[1]["idx"]]
print("k=1 matches found in:", pd.Series(src).value_counts(normalize=True).round(3).to_dict())
print(f"k=1 match has the same GROUP: {(P.GROUP.values[m[1]['idx']] == Q.GROUP.values).mean():.1%}")

print("\nCoverage and sign agreement between shifts, per threshold:")
for thr in [0.9, 0.95, 0.98, 0.99]:
    ok = sim > thr
    n_ok = ok.sum(1)
    s = np.where(ok, np.sign(read), 0)
    agree = np.abs(s.sum(1)) / np.maximum(n_ok, 1)
    multi = n_ok >= 2
    print(f"  thr {thr}: covered (>=1 copy) {np.mean(n_ok >= 1):.1%}, median copies {np.median(n_ok):.0f}, "
          f"sign agreement among rows with >=2 copies {agree[multi].mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(sim[:, 0], bins=np.linspace(0.6, 1, 120), color="#c44e52")
ax[0].set(title="Test queries: best similarity, k=1", xlabel="cosine similarity")
ax[1].plot(ks, (sim > 0.99).mean(0) * 100, "o-")
ax[1].set(title="Share of test rows matched with sim > 0.99, per shift", xlabel="k", ylabel="%")
plots._save(fig, EXP, "test_matching")
plots.show()
