"""Step 26: forward-in-time validation, and how the signal drifts along the timeline.

Step 22 puts the test dates late on the timeline, and isolated test dates 20+ days apart cannot
fit inside the train period: 120 dates x ~21 days is about ten years, the length of the train set
itself. The test is most likely a later period sampled about once a month. Segment CV mixes past
and future; here models are trained on the past and scored on the future only.

Assumed order of the four components (growing universe, falling turnover): 2, 1, 0, 3.
Lockbox rows are excluded from both sides.
"""
from _research import *  # noqa

order = {2: 0, 1: 1, 0: 2, 3: 3}
span = tl.groupby("COMP")["T"].max() + 1
start = {c: sum(span[k] for k in order if order[k] < order[c]) for c in order}
G_date = tl.COMP.map(start) + tl["T"]
G = pd.Series(ts).map(G_date).values
in_res = np.zeros(len(X), bool); in_res[research] = True
print(f"timeline length {int(G.max()) + 1} trading days")

# ---- drift of the raw effect along the timeline ----------------------------------------
oof = pd.read_parquet(DATA_DIR / "oof_23.parquet")["base"].reindex(X.index).values
bucket = pd.cut(G, 8, labels=False)
d = pd.DataFrame({"b": bucket, "g": X.GROUP.values, "r1": X.RET_1.values, "y": y.values,
                  "hit": np.where(np.isnan(oof), np.nan, (oof > 0.5) == yb), "up": yb,
                  "rule": (X.RET_1.values > 0) == yb, "turn": X.MEDIAN_DAILY_TURNOVER.values})
t = d.groupby("b").agg(rows=("y", "size"), up=("up", "mean"), rule_RET1=("rule", "mean"), model_oof=("hit", "mean"),
                       turnover=("turn", "mean"))
c = d.groupby(["b", "g"]).apply(lambda q: q.r1.corr(q.y)).unstack()
c.columns = [f"corr_G{k}" for k in c.columns]
print("along the timeline (8 equal buckets, oldest first):")
print(pd.concat([t, c], axis=1).round(3).to_string())

# ---- forward splits ----------------------------------------------------------------------
B = ("raw", "benchmark")
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)
small_cols = [f"RET_{i}" for i in range(1, 6)] + list(F_all["benchmark"].columns)
VAR = {
    "base": dict(blocks=B),
    "noTE": dict(blocks=B, te=False),
    "decay_hl500": dict(blocks=B, hl=500),
    "decay_hl250": dict(blocks=B, hl=250),
    "last_1000d": dict(blocks=B, last=1000),
    "small": dict(blocks=B, cols=small_cols),
    "strong_reg": dict(blocks=B, params=REG),
    "+volume": dict(blocks=B + ("volume",)),
    "+context": dict(blocks=B + ("context",)),
}
rows = []
for split, cut, end in (("A: past -> second half of comp 0 + comp 3", start[0] + span[0] * 0.5, G.max() + 1),
                        ("B: past -> comp 3", start[3], G.max() + 1),
                        ("C: comp 2 + comp 1 -> first half of comp 0", start[0], start[0] + span[0] * 0.5)):
    tr = np.where(in_res & (G < cut - 20))[0]
    va = np.where(in_res & (G >= cut) & (G < end))[0]
    print(f"\nsplit {split}: train {len(tr):,} rows, valid {len(va):,} rows / {len(set(ts[va]))} dates", flush=True)
    preds = {}
    for name, cfg in VAR.items():
        Fd = feats(cfg["blocks"])
        Fv = (Fd[cfg["cols"]] if "cols" in cfg else Fd).values
        w, sub = None, tr
        if "hl" in cfg:
            w = np.power(0.5, (cut - G) / cfg["hl"])
        if "last" in cfg:
            sub = tr[G[tr] >= cut - cfg["last"]]
        preds[name] = fit_predict(Fv, sub, Fv[va], alloc[va], params=cfg.get("params"), weight=w, te=cfg.get("te", True))
    preds["rule sign(RET_1)"] = (X.RET_1.values[va] > 0).astype(float)
    preds["always up"] = np.ones(len(va))
    pdx = lambda p: pd.Series((p > 0.5) == yb[va], index=ts[va]).groupby(level=0).mean()
    b = pdx(preds["base"])
    for name, p in preds.items():
        rows.append(dict(split=split[0], variant=name, acc=((p > 0.5) == yb[va]).mean() * 100,
                         t_vs_base=tstat(pdx(p) - b), share_up=(p > 0.5).mean()))
res = pd.DataFrame(rows)
print("\n" + res.pivot(index="variant", columns="split", values=["acc", "t_vs_base"]).round(2).loc[list(VAR) + ["rule sign(RET_1)", "always up"]].to_string())
res.round(3).to_csv(Path(__file__).resolve().parents[1] / "results" / "26_forward.csv", index=False)
