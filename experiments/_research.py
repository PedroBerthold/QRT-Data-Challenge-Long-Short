"""Shared protocol for steps 23+: research set / lockbox, late-era slice, fold runner, paired statistics."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from _common import *  # noqa
from pathlib import Path
from src.data import DATA_DIR, load_train, load_test
from src.features import build, target_encoding, BLOCKS
from src.cv import LGB_PARAMS
from src.segments import load_segments

X, y = load_train()
Xt = load_test()
seg_id = X.TS.map(load_segments()).values
yb = (y.values > 0).astype(int)
ts, alloc = X.TS.values, X.ALLOCATION.values

# same split as step 20: lockbox = last of 6 segment folds, 5-fold segment CV on the rest
research, lock = list(GroupKFold(6).split(X, groups=seg_id))[-1]
folds = [(research[a], research[b]) for a, b in GroupKFold(5).split(research, groups=seg_id[research])]

# late era = where the test dates sit on the timeline of step 22 (component 3, second half of component 0)
tl = pd.read_csv(DATA_DIR / "timeline.csv").set_index("TS")
rel = tl["T"] / tl.groupby("COMP")["T"].transform("max")
late_date = (tl.COMP == 3) | ((tl.COMP == 0) & (rel > 0.5))
late = pd.Series(ts).map(late_date).values

ALL_BLOCKS = list(BLOCKS)
F_all = {b: BLOCKS[b](X) for b in ALL_BLOCKS}
Ft_all = {b: BLOCKS[b](Xt) for b in ALL_BLOCKS}


def feats(blocks, test=False):
    src = Ft_all if test else F_all
    return pd.concat([src[b] for b in blocks], axis=1)


def fit_predict(Fv, tr, Fva, va_alloc, params=None, rounds=500, weight=None, te=True, seeds=(42,)):
    """Binary LightGBM; target encoding fitted on tr only (inner out-of-fold for the train rows)."""
    Ftr = Fv[tr]
    if te:
        te_tr = np.full((len(tr), 2), np.nan)
        for a, b in GroupKFold(4).split(tr, groups=seg_id[tr]):
            te_tr[b] = target_encoding(alloc[tr[a]], y.values[tr[a]], alloc[tr[b]]).values
        Ftr = np.c_[Ftr, te_tr]
        Fva = np.c_[Fva, target_encoding(alloc[tr], y.values[tr], va_alloc).values]
    pred = np.zeros(len(Fva))
    for s in seeds:
        p = {**LGB_PARAMS, "objective": "binary", **(params or {}), "seed": s}
        pred += lgb.train(p, lgb.Dataset(Ftr, yb[tr], weight=None if weight is None else weight[tr]), rounds).predict(Fva)
    return pred / len(seeds)


def run_cv(name, blocks=("raw", "benchmark"), **kw):
    t0 = time.time()
    Fv = feats(blocks).values
    oof = np.full(len(X), np.nan)
    for tr, va in folds:
        oof[va] = fit_predict(Fv, tr, Fv[va], alloc[va], **kw)
    print(f"  [{name}] done in {time.time() - t0:.0f}s", flush=True)
    return oof


def per_date(pred, idx, thr=0.5):
    return pd.Series((pred[idx] > thr) == yb[idx], index=ts[idx]).groupby(level=0).mean()


def tstat(d):
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else 0.0


def report(oofs, base="base"):
    """Accuracy on all research dates and on late-era dates, paired per-date t against the baseline."""
    r_late = research[late[research]]
    b_all, b_late = per_date(oofs[base], research), per_date(oofs[base], r_late)
    rows = []
    for k, o in oofs.items():
        d_all, d_late = per_date(o, research), per_date(o, r_late)
        fold_acc = [((o[va] > 0.5) == yb[va]).mean() * 100 for _, va in folds]
        rows.append(dict(config=k, acc=((o[research] > 0.5) == yb[research]).mean() * 100,
                         folds=" ".join(f"{a:.1f}" for a in fold_acc),
                         wins=sum(a > b for a, b in zip(fold_acc, rows[0]["_f"])) if rows else 0,
                         t=tstat(d_all - b_all), late_acc=((o[r_late] > 0.5) == yb[r_late]).mean() * 100,
                         late_t=tstat(d_late - b_late), share_up=(o[research] > 0.5).mean(), _f=fold_acc))
    out = pd.DataFrame(rows).drop(columns="_f").round(3)
    print(out.to_string(index=False), flush=True)
    return out


# ---- forward-in-time splits (steps 26+) ----------------------------------------------------
_order = {2: 0, 1: 1, 0: 2, 3: 3}                      # assumed chronological order of the components
_span = tl.groupby("COMP")["T"].max() + 1
T_START = {c: sum(_span[k] for k in _order if _order[k] < _order[c]) for c in _order}
G = pd.Series(ts).map(tl.COMP.map(T_START) + tl["T"]).values          # global trading-day coordinate
_in_res = np.zeros(len(X), bool); _in_res[research] = True
FORWARD = {}
for _n, _cut, _end in (("A", T_START[0] + _span[0] * 0.5, G.max() + 1), ("B", T_START[3], G.max() + 1),
                       ("C", T_START[0], T_START[0] + _span[0] * 0.5)):
    FORWARD[_n] = (np.where(_in_res & (G < _cut - 20))[0], np.where(_in_res & (G >= _cut) & (G < _end))[0])


def forward_eval(variants, base):
    """variants: name -> function(tr, va) returning probabilities for va. Paired per-date t against `base`."""
    rows, keep = [], {}
    for s, (tr, va) in FORWARD.items():
        t0 = time.time()
        preds = {n: f(tr, va) for n, f in variants.items()}
        keep[s] = preds
        pdx = lambda p: pd.Series((p > 0.5) == yb[va], index=ts[va]).groupby(level=0).mean()
        b = pdx(preds[base])
        for n, p in preds.items():
            rows.append(dict(split=s, variant=n, acc=((p > 0.5) == yb[va]).mean() * 100, t=tstat(pdx(p) - b),
                             share_up=(p > 0.5).mean()))
        print(f"  split {s} done in {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    tab = res.pivot(index="variant", columns="split", values=["acc", "t"]).loc[list(variants)]
    tab[("acc", "mean")] = tab["acc"].mean(axis=1)
    tab[("up", "A")] = res[res.split == "A"].set_index("variant").share_up
    print(tab.round(2).to_string(), flush=True)
    return res, keep
