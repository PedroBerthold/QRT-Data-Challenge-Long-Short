"""Step 22: chain the 40 segments into one timeline with the multi-shift links of step 21,
place the test dates on it, and measure how allocation-specific the return dynamics are.

Output: data/timeline.csv (TS, COMP, T) where T is a trading-day coordinate inside component COMP.
"""
import numpy as np
import pandas as pd
from _common import *  # noqa
from src.data import DATA_DIR, load_train, load_test, RET

X, y = load_train()
Xt = load_test()
segs = pd.read_csv(DATA_DIR / "date_segments.csv")
L = pd.read_csv(DATA_DIR / "date_links.csv")
tr = segs[~segs.IS_TEST].reset_index(drop=True)
idx = {d: i for i, d in enumerate(tr.TS)}

# weighted union-find: off[i] = time of i minus time of its root
parent, off = list(range(len(tr))), [0] * len(tr)
def find(i):
    path = []
    while parent[i] != i:
        path.append(i); i = parent[i]
    for j in reversed(path):
        off[j] += off[parent[j]] if parent[j] != i else 0
        parent[j] = i
    return i
def find_off(i):
    r = find(i); return r, off[i]

edges = []
for s, g in tr.groupby("SEG"):
    g = g.sort_values("POS")
    edges += [(idx[a], idx[b], 1, 2.0) for a, b in zip(g.TS[:-1], g.TS[1:])]
edges += [(idx[a], idx[b], k, sc) for a, b, k, sc in zip(L["T"], L["S"], L.k, L.score)]
edges.sort(key=lambda e: (e[2], -e[3]))
conflict = []
for a, b, k, _ in edges:                     # time[b] = time[a] + k
    (ra, oa), (rb, ob) = find_off(a), find_off(b)
    if ra == rb:
        conflict.append(ob - oa - k)
    else:
        parent[rb] = ra; off[rb] = oa + k - ob
comp = np.array([find(i) for i in range(len(tr))]); T = np.array([find_off(i)[1] for i in range(len(tr))])
tr["COMP"] = pd.factorize(comp)[0]
tr["T"] = T - pd.Series(T).groupby(tr.COMP).transform("min").values
conflict = np.abs(conflict)
print(f"edges {len(edges)}, redundant {len(conflict)}, of which inconsistent by >2 days: {(conflict > 2).sum()}")

n = X.groupby("TS").size(); turn = X.groupby("TS").MEDIAN_DAILY_TURNOVER.mean()
tr["n"], tr["turn"] = tr.TS.map(n).values, tr.TS.map(turn).values
def edge(g, col, last):
    g = g.sort_values("T"); return (g[col].iloc[-20:] if last else g[col].iloc[:20]).median()
summ = tr.groupby("COMP").apply(lambda g: pd.Series(dict(
    dates=len(g), span=g["T"].max() + 1, segs=" ".join(map(str, g.sort_values("T").SEG.unique())),
    n_start=edge(g, "n", 0), n_end=edge(g, "n", 1), turn_start=edge(g, "turn", 0), turn_end=edge(g, "turn", 1))))
print(summ.to_string())

# ---- test dates: nearest train date by per-allocation turnover profile -----------------
A = pd.concat([X, Xt])
P = np.log(A.pivot(index="TS", columns="ALLOCATION", values="MEDIAN_DAILY_TURNOVER") + 1e-9)
Ptr, Pte = P.loc[tr.TS].values, P.loc[Xt.TS.unique()].values
near = np.array([np.nanargmin(np.nanmean(np.abs(Ptr - v), 1)) for v in Pte])
place = tr.iloc[near]
print("\ntest dates, nearest train date: component counts", place.COMP.value_counts().to_dict())
for c, g in place.groupby("COMP"):
    span = tr[tr.COMP == c]["T"].max()
    print(f"  comp {c}: {len(g)} test dates, position in the component (0=start, 1=end) quantiles 10/50/90%: "
          f"{(g['T'] / span).quantile([.1, .5, .9]).round(2).tolist()}")
tr[["TS", "SEG", "POS", "COMP", "T"]].to_csv(DATA_DIR / "timeline.csv", index=False)

# ---- is the dynamics allocation-specific, and is that stable? ---------------------------
d = X[["TS", "ALLOCATION", "GROUP"] + RET[:5]].assign(y=y.values)
d = d.merge(tr[["TS", "COMP", "T"]], on="TS")
order = d.COMP.map(summ.n_start.rank(method="first")) * 1e5 + d["T"]      # rough global order
half = order > order.median()
vol = X[RET].std(axis=1).values[d.index] if False else None
def slopes(g):
    out = {"mean_y": g.y.mean(), "up": (g.y > 0).mean()}
    for c in RET[:3]:
        out[f"corr_{c}"] = np.corrcoef(g[c].fillna(0), g.y)[0, 1]
    return pd.Series(out)
s1, s2 = d[~half].groupby("ALLOCATION").apply(slopes), d[half].groupby("ALLOCATION").apply(slopes)
both = s1.join(s2, lsuffix="_early", rsuffix="_late").dropna()
print("\nper-allocation statistics, first half of the timeline vs second half (correlation across allocations):")
for c in s1.columns:
    print(f"  {c:12s} early mean {s1[c].mean():+.4f}  late mean {s2[c].mean():+.4f}  "
          f"sd across allocations {s1[c].std():.4f}  early-late corr {both[c + '_early'].corr(both[c + '_late']):+.2f}")
g1, g2 = d[~half].groupby("GROUP"), d[half].groupby("GROUP")
print("by GROUP, corr(RET_1, target) early / late:",
      {k: (round(np.corrcoef(a.RET_1.fillna(0), a.y)[0, 1], 3), round(np.corrcoef(b.RET_1.fillna(0), b.y)[0, 1], 3))
       for (k, a), (_, b) in zip(g1, g2)})
