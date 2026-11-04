"""Leak exploitation: find, for each row, the rows of the same allocation k days later.

Row of day t:   RET_1 = return of day t, ..., target = return of day t+1.
Row of day t+k: RET_k = return of day t+1  (= the target of the row of day t),
                RET_{k+1..k+L} = returns of days t..t-L+1  (= RET_1..RET_L of the row of day t).

So for each query row q and each shift k, we look for the pool row c whose
RET_{k+1..k+L} is closest to q's RET_1..RET_L (cosine similarity, values are noised,
never identical). Then c.RET_k is a noisy copy of q's target.
"""
import numpy as np

RET_COLS = [f"RET_{i}" for i in range(1, 21)]


def _normalize(M):
    M = np.nan_to_num(M.astype(np.float32))
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


def match(Q, P, shifts=range(1, 11), max_len=10, block=256,
          q_index=None, p_index=None, q_group=None, p_group=None, verbose=True):
    """Q: query returns (n_q x 20, column 0 = RET_1). P: pool returns (n_p x 20).

    q_index / p_index: optional global row ids, a query is never matched with itself.
    q_group / p_group: optional GROUP arrays, if given only rows of the same GROUP can match.
    Returns dict k -> dict(idx, sim, sim2, read) where
        idx  = index in P of the best match,
        sim  = its cosine similarity, sim2 = second best (margin = sim - sim2),
        read = P[idx, k-1] = noisy copy of the query's target.
    """
    out = {}
    for k in shifts:
        L = min(max_len, 20 - k)
        Qn = _normalize(Q[:, :L])
        Pn = _normalize(P[:, k:k + L])
        idx = np.empty(len(Q), dtype=np.int64)
        s1 = np.empty(len(Q), dtype=np.float32)
        s2 = np.empty(len(Q), dtype=np.float32)
        for a in range(0, len(Q), block):
            b = min(a + block, len(Q))
            S = Qn[a:b] @ Pn.T
            if q_index is not None and p_index is not None:
                same = q_index[a:b, None] == p_index[None, :]
                S[same] = -2
            if q_group is not None and p_group is not None:
                S[q_group[a:b, None] != p_group[None, :]] = -2
            j = S.argmax(1)
            r = np.arange(b - a)
            s1[a:b] = S[r, j]
            S[r, j] = -2
            s2[a:b] = S.max(1)
            idx[a:b] = j
        out[k] = dict(idx=idx, sim=s1, sim2=s2, read=P[idx, k - 1])
        if verbose:
            print(f"  shift k={k:2d} (L={L}): median best sim {np.median(s1):.4f}", flush=True)
    return out


def to_frames(matches):
    """Stack per-shift results into arrays of shape (n_q, n_shifts)."""
    ks = sorted(matches)
    stack = lambda key: np.column_stack([matches[k][key] for k in ks])
    return ks, stack("read"), stack("sim"), stack("sim2")


def aggregate(read, sim, sim2, strategy="mean", thr=0.99, min_margin=0.0):
    """Combine the noisy copies of the target. Returns (estimate, covered mask).

    k1    : only the next-day row (shift 1)
    best  : the copy from the shift with the highest similarity
    mean  : average of all copies with sim > thr and margin > min_margin
    wmean : same, weighted by 1 / (1 - sim)
    vote  : majority of signs among accepted copies
    """
    ok = (sim > thr) & ((sim - sim2) > min_margin) & ~np.isnan(read)
    if strategy == "k1":
        est, cov = read[:, 0], ok[:, 0]
    elif strategy == "best":
        s = np.where(ok, sim, -np.inf)
        j = s.argmax(1)
        est, cov = read[np.arange(len(read)), j], ok.any(1)
    elif strategy in ("mean", "wmean", "vote"):
        w = ok.astype(float)
        if strategy == "wmean":
            w = w / np.maximum(1 - sim, 1e-4)
        vals = np.sign(read) if strategy == "vote" else read
        num = np.nansum(np.where(ok, vals * w, 0), 1)
        den = w.sum(1)
        est = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        cov = den > 0
    else:
        raise ValueError(strategy)
    return est, cov
