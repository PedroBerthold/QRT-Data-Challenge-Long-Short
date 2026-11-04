"""Reconstruct the chronological order of the (shuffled) dates.

Row (date d, allocation a) holds RET_1..RET_20. If d' is the trading day after d,
then RET_2..RET_7 at d' should match RET_1..RET_6 at d (up to the small noise added
by the organisers). We compare these cross-sectional vectors for every pair of dates
and link each date to its best successor. Chains of linked dates = "segments".

Use: segments as CV groups (block time CV, avoids neighbouring dates leaking
through the 19 overlapping days of history).

Output: data/date_segments.csv with columns TS, SEG, POS, IS_TEST.
"""
import numpy as np
import pandas as pd
from .data import DATA_DIR, load_train, load_test

N_LAGS = 6          # number of lags used to match a date with its successor
MIN_SCORE = 0.9     # similarity threshold to accept a link
MIN_OVERLAP = 20    # minimum number of common non-missing values


def _stack(A, dates, allocs, cols):
    mats = [A.pivot(index="TS", columns="ALLOCATION", values=c)
             .reindex(index=dates, columns=allocs).values for c in cols]
    return np.concatenate(mats, axis=1)


def build_segments():
    X, _ = load_train()
    Xt = load_test()
    A = pd.concat([X.assign(IS_TEST=False), Xt.assign(IS_TEST=True)])
    dates = A["TS"].unique()
    allocs = np.sort(A["ALLOCATION"].unique())

    cur = _stack(A, dates, allocs, [f"RET_{i}" for i in range(1, N_LAGS + 1)])
    nxt = _stack(A, dates, allocs, [f"RET_{i}" for i in range(2, N_LAGS + 2)])

    # Uncentered cosine similarity on common non-missing entries
    m1, m2 = ~np.isnan(cur), ~np.isnan(nxt)
    c0, n0 = np.nan_to_num(cur), np.nan_to_num(nxt)
    num = c0 @ n0.T
    overlap = m1.astype(float) @ m2.T.astype(float)
    den = np.sqrt(((c0 ** 2) @ m2.T.astype(float)) * (m1.astype(float) @ (n0 ** 2).T)) + 1e-30
    S = num / den
    S[overlap < MIN_OVERLAP] = -1
    np.fill_diagonal(S, -1)

    best, succ = S.max(1), S.argmax(1)
    print("Best-match score quantiles (1%, 10%, 50%):", np.quantile(best, [.01, .1, .5]).round(3))

    # Greedy linking: each date gets at most one successor and one predecessor
    nxt_of, taken = {}, set()
    for i in np.argsort(-best):
        if best[i] > MIN_SCORE and succ[i] not in taken:
            nxt_of[i] = succ[i]
            taken.add(succ[i])
    has_prev = set(nxt_of.values())

    seg = np.full(len(dates), -1)
    pos = np.zeros(len(dates), dtype=int)
    sid = 0
    starts = [i for i in range(len(dates)) if i not in has_prev] + list(range(len(dates)))
    for i in starts:
        if seg[i] >= 0:
            continue
        j, p = i, 0
        while j is not None and seg[j] < 0:
            seg[j], pos[j] = sid, p
            j, p = nxt_of.get(j), p + 1
        sid += 1

    is_test = A.groupby("TS")["IS_TEST"].first().reindex(dates).values
    out = pd.DataFrame({"TS": dates, "SEG": seg, "POS": pos, "IS_TEST": is_test})
    out.to_csv(DATA_DIR / "date_segments.csv", index=False)

    tr = out[~out.IS_TEST]
    print(f"Train: {tr.SEG.nunique()} segments, sizes: {tr.groupby('SEG').size().sort_values(ascending=False).head(10).tolist()}")
    linked_test = np.isin(np.where(is_test)[0], list(nxt_of) + list(has_prev)).sum()
    print(f"Test dates linked to any other date: {linked_test} / {is_test.sum()}")
    return out


def load_segments() -> pd.Series:
    path = DATA_DIR / "date_segments.csv"
    if not path.exists():
        build_segments()
    return pd.read_csv(path).set_index("TS")["SEG"]


if __name__ == "__main__":
    build_segments()
