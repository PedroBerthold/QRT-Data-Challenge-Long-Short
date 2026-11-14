"""Step 21: where do the test dates sit in time, and can the 40 segments be chained?

src/segments.py links a date to its successor with a 1-day shift only. Here every pair of
dates is compared at the date level for shifts k = 1..19: if S = T + k trading days, then
RET_{j+k} at S equals RET_j at T for j = 1..20-k, for all ~278 allocations at once.
Even k = 19 leaves a 278-dimensional overlap, far stronger than any row-level match.

Also: universe size and turnover of each segment and of the test dates (slow variables
that place a date in time without any overlap in returns).

Output: data/date_links.csv (pairs above threshold), printed summary.
"""
import numpy as np
import pandas as pd
from _common import *  # noqa
from src.data import DATA_DIR, load_train, load_test, RET
from src.segments import DATA_DIR as _D

X, y = load_train()
Xt = load_test()
A = pd.concat([X, Xt])
segs = pd.read_csv(DATA_DIR / "date_segments.csv").set_index("TS")
dates = segs.index.values
allocs = np.sort(A.ALLOCATION.unique())
is_test = segs.IS_TEST.values
D = len(dates)
R = np.stack([A.pivot(index="TS", columns="ALLOCATION", values=c).reindex(index=dates, columns=allocs).values
              for c in RET]).astype(np.float32)          # (20, dates, allocations)
M = ~np.isnan(R)
R0 = np.nan_to_num(R)
seg, pos = segs.SEG.values, segs.POS.values
same_seg_gap = np.where(seg[:, None] == seg[None, :], pos[None, :] - pos[:, None], 0)   # S after T by k

links, rows = [], []
for k in range(1, 20):
    n = 20 - k
    a, ma = np.concatenate(R0[:n], 1), np.concatenate(M[:n], 1).astype(np.float32)        # RET_j at T
    b, mb = np.concatenate(R0[k:], 1), np.concatenate(M[k:], 1).astype(np.float32)        # RET_{j+k} at S
    num = a @ b.T
    den = np.sqrt(((a ** 2) @ mb.T) * (ma @ (b ** 2).T)) + 1e-30
    S = num / den
    S[(ma @ mb.T) < 20 * 1] = -1
    np.fill_diagonal(S, -1)
    true = same_seg_gap == k
    null = S[~true & ~is_test[:, None] & ~is_test[None, :]]
    thr = 0.9 if k <= 8 else 0.95     # true k-shifted pairs sit at 0.94-0.99, unrelated pairs below 0.6
    hit = np.argwhere(S > thr)
    new = [(i, j) for i, j in hit if not true[i, j]]
    for i, j in new:
        links.append((dates[i], dates[j], k, float(S[i, j]), bool(is_test[i]), bool(is_test[j]),
                      int(seg[i]), int(seg[j])))
    rows.append(dict(k=k, true_pairs=int(true.sum()), true_med=np.median(S[true]) if true.any() else np.nan,
                     true_q01=np.quantile(S[true], .01) if true.any() else np.nan,
                     null_q999=float(np.quantile(null, .999)), thr=thr, new_links=len(new),
                     new_test=sum(1 for i, j in new if is_test[i] or is_test[j]),
                     test_best_med=float(np.median(np.maximum(S[is_test].max(1), S[:, is_test].max(0))))))
print(pd.DataFrame(rows).round(3).to_string(index=False))
L = pd.DataFrame(links, columns=["T", "S", "k", "score", "T_test", "S_test", "T_seg", "S_seg"])
L.to_csv(DATA_DIR / "date_links.csv", index=False)
print(f"\nnew links (not already inside a segment): {len(L)}; involving a test date: {(L.T_test | L.S_test).sum()}")
if len(L):
    cross = L[(L.T_seg != L.S_seg)]
    print("segment-to-segment links (T_seg -> S_seg, min shift):")
    print(cross.groupby(["T_seg", "S_seg"]).agg(k_min=("k", "min"), n=("k", "size"), score=("score", "max")).reset_index().head(80).to_string(index=False))
    print("same-segment unlabeled links:", int((L.T_seg == L.S_seg).sum()))

# ---- slow variables: universe size and turnover -----------------------------------------
turn = A.pivot(index="TS", columns="ALLOCATION", values="MEDIAN_DAILY_TURNOVER").reindex(index=dates, columns=allocs)
n_alloc = turn.notna().sum(1).values
info = pd.DataFrame({"seg": seg, "n": n_alloc, "turn": turn.mean(1).values, "test": is_test})
tr = info[~info.test].groupby("seg").agg(days=("n", "size"), n_min=("n", "min"), n_max=("n", "max"),
                                         turn=("turn", "mean")).sort_values("turn")
print("\nsegments sorted by mean turnover:\n", tr.round(4).T.to_string())
te = info[info.test]
print("\ntest dates: n allocations quantiles", te.n.quantile([0, .1, .5, .9, 1]).tolist(),
      "| turnover quantiles", te.turn.quantile([0, .1, .5, .9, 1]).round(4).tolist())
# nearest train date for each test date by per-allocation turnover profile
Tm = turn.values
tr_idx, te_idx = np.where(~is_test)[0], np.where(is_test)[0]
near = []
for i in te_idx:
    d = np.nanmean(np.abs(np.log(Tm[tr_idx] + 1e-9) - np.log(Tm[i] + 1e-9)), 1)
    j = tr_idx[np.nanargmin(d)]
    near.append((seg[j], float(np.nanmin(d)), float(np.nanmedian(d))))
near = pd.DataFrame(near, columns=["seg", "dist", "median_dist"])
print("nearest train segment of each test date by turnover profile (segment: count):",
      near.seg.value_counts().to_dict())
print("distance to nearest train date: median %.4f (median distance to any train date %.4f)" %
      (near.dist.median(), near.median_dist.median()))
# same for consecutive train dates as a reference
ref = [np.nanmean(np.abs(np.log(Tm[i] + 1e-9) - np.log(Tm[j] + 1e-9)))
       for i, j in zip(tr_idx[:-1], tr_idx[1:]) if seg[i] == seg[j] and pos[j] - pos[i] == 1]
print("reference, distance between consecutive train dates: median %.4f" % np.median(ref))
# ---- how much of the sign is common to all allocations of a date? -----------------------
up = (y > 0).groupby(X.TS).mean()
print("\nshare of rows agreeing with the majority sign of their date: %.3f; dates with >70%% same sign: %.3f"
      % (np.maximum(up, 1 - up).mean(), (np.maximum(up, 1 - up) > .7).mean()))
