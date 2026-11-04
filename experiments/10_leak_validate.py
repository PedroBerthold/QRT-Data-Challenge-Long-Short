"""Step 10: measure the leak on TRAIN rows, where we know the true answer.

Inside the train set, the row of day t+k contains the target of day t (RET_k), up to noise.
This script quantifies how much accuracy that gives. Step 11 checks whether the same works
for test rows (so far: it does not).

We take a sample of train rows as "fake test", search their future rows in train + test,
read the noisy copies of the target, and measure the accuracy. Hypotheses tested:

H1  Reading the next-day row (shift k=1) gives ~80% accuracy.
H2  Accuracy depends on the similarity of the match: we need a threshold.
H3  Later rows (k=2..10) also contain the target, with similar accuracy.
H4  The noise is independent between copies: then averaging copies beats a single read.
H5  The best way to combine copies (k1 / best / mean / wmean / vote) and the threshold.
H6  False matches: what similarity does a query with NO true match reach? (null test)
H7  Errors come from returns close to zero, whose sign the noise can flip.

Set N_QUERIES below (QUICK=1 uses 2,000). Figures in figures/10_leak_validate/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _leak_common import *  # noqa
from src.matching import to_frames, aggregate, match
from src import plots

EXP = "10_leak_validate"
N_QUERIES = 2000 if QUICK else 20000

X, y, Xt, P = pool()
Q = X.sample(N_QUERIES, random_state=0)
truth = y.loc[Q.index].values
m = cached_match(f"train{N_QUERIES}", Q, P)
ks, read, sim, sim2 = to_frames(m)
correct = np.sign(read) == np.sign(truth)[:, None]

# sanity: in train, the matched row should carry the same ALLOCATION label
same_alloc = P.ALLOCATION.values[m[1]["idx"]] == Q.ALLOCATION.values
print(f"\nSanity: k=1 match has the same ALLOCATION label in {same_alloc.mean():.1%} of cases")

# H1 + H3: accuracy per shift
print("\nH1/H3 accuracy of a single read, per shift (all rows, no threshold):")
acc_k = correct.mean(0)
for k, a, s in zip(ks, acc_k, np.median(sim, 0)):
    print(f"  k={k:2d}: acc {a:.4f}   median sim {s:.4f}")

# H2: accuracy by similarity bin (k=1)
bins = [-1, .9, .95, .98, .99, .995, .998, 1.01]
b = pd.cut(sim[:, 0], bins)
tab = pd.DataFrame({"bin": b, "ok": correct[:, 0]}).groupby("bin", observed=True).ok.agg(["mean", "size"])
print("\nH2 accuracy by similarity (k=1):\n", tab.round(4).to_string())

# H6: null similarity (queries with lags reversed have no true continuation)
Qnull = Q[RET_COLS].values[:, ::-1].copy()
mnull = match(Qnull[:1000], P[RET_COLS].values, [1], MAX_LEN, verbose=False)
null_sim = mnull[1]["sim"]
print(f"\nH6 null test: best sim of queries WITHOUT true match: median {np.median(null_sim):.4f}, "
      f"99th pct {np.quantile(null_sim, .99):.4f}")

# H4: are errors independent across shifts?
err = read - truth[:, None]
ok_all = (sim > 0.95).all(1)
C = pd.DataFrame(err[ok_all], columns=[f"k{k}" for k in ks]).corr()
off = C.values[~np.eye(len(ks), dtype=bool)]
print(f"\nH4 correlation of read errors between shifts (rows matched on all shifts): mean {off.mean():.3f}")
print("    ~0 -> independent noise, averaging helps;  ~1 -> same noise, averaging useless")

# H5: strategies x thresholds
print("\nH5 strategies (accuracy on covered rows | coverage | overall with fallback 'predict 1'):")
rows = []
for strat in ["k1", "best", "mean", "wmean", "vote"]:
    for thr in [0.9, 0.95, 0.98, 0.99]:
        est, cov = aggregate(read, sim, sim2, strat, thr)
        pred = np.where(cov, est > 0, True)
        rows.append(dict(strategy=strat, thr=thr, acc_covered=(np.sign(est[cov]) == np.sign(truth[cov])).mean(),
                         coverage=cov.mean(), acc_overall=(pred == (truth > 0)).mean()))
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
best = res.loc[res.acc_overall.idxmax()]
print(f"\nBest: {best.strategy} thr={best.thr} -> overall acc {best.acc_overall:.4f}")

# H7: errors vs size of the true return
est, cov = aggregate(read, sim, sim2, best.strategy, best.thr)
mag = pd.qcut(np.abs(truth), 10, labels=False)
acc_mag = pd.Series(np.sign(est) == np.sign(truth))[cov].groupby(mag[cov]).mean()
print("\nH7 accuracy by decile of |true return| (0 = closest to zero):", acc_mag.round(3).tolist())

# figures
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(ks, acc_k * 100, "o-")
ax[0, 0].set(title="H1/H3 accuracy of a single read per shift k", xlabel="k (row k days later)", ylabel="%")
ax[0, 1].hist(sim[:, 0], bins=np.linspace(0.6, 1, 120), alpha=0.7, label="train queries (true match exists)")
ax[0, 1].hist(null_sim, bins=np.linspace(0.6, 1, 120), alpha=0.7, label="null queries (no true match)")
ax[0, 1].set(title="H2/H6 best similarity, k=1", xlabel="cosine similarity")
ax[0, 1].legend(fontsize=8)
im = ax[1, 0].imshow(C.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax[1, 0].set(title="H4 correlation of read errors between shifts", xticks=range(len(ks)), yticks=range(len(ks)),
             xticklabels=ks, yticklabels=ks)
fig.colorbar(im, ax=ax[1, 0])
ax[1, 1].bar(acc_mag.index + 1, acc_mag.values * 100, color="#55a868")
ax[1, 1].set(title="H7 accuracy by decile of |true return|", xlabel="decile (1 = closest to 0)", ylabel="%",
             ylim=(40, 101))
plots._save(fig, EXP, "leak_validation")
plots.show()
