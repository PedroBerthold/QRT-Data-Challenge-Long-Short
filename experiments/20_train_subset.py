"""Step 20: does a well-chosen part of the training dates beat training on everything?

Model = final model of step 4 (binary LightGBM + allocation target encoding).
Training-set variants, always validated on ALL rows of the validation segments:
    full      every training date
    rand50/25 random 50% / 25% of the training dates (control: what does less data cost?)
    like50/25 the 50% / 25% of training dates whose market state looks most like the test dates
    wfull     every date, weighted by its resemblance to the test dates

"Looks like the test" = a date-level classifier train-date vs test-date on features only
(cross-sectional mean return path, dispersion, volatility, turnover). No label is used.

Two scores per variant: plain accuracy, and accuracy weighted towards test-like dates.
One sixth of the segments is a lockbox, never used for selection, scored once at the end.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from _common import *  # noqa
from src.data import load_train, load_test, RET
from src.features import build, target_encoding
from src.cv import LGB_PARAMS
from src.segments import load_segments

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {**LGB_PARAMS, "objective": "binary"}
X, y = load_train()
Xt = load_test()
seg = load_segments()
seg_id = X.TS.map(seg).values
yb = (y.values > 0).astype(int)
ts = X.TS.values
alloc = X.ALLOCATION.values
F, Ft = build(X), build(Xt)
print(f"train {len(X):,} rows / {X.TS.nunique()} dates / {len(set(seg_id))} segments; "
      f"test {len(Xt):,} rows / {Xt.TS.nunique()} dates", flush=True)


# ---- 1. how test-like is each training date? (features only) -------------------------
def date_state(A):
    g = A.groupby("TS")
    d = g[RET].mean()
    d["DISP_1"] = g["RET_1"].std()
    d["DISP_5"] = A[RET[:5]].mean(1).groupby(A.TS).std()
    d["VOL"] = A[RET].std(1).groupby(A.TS).mean()
    d["TURN"] = g["MEDIAN_DAILY_TURNOVER"].mean()
    d["MOM_5"], d["MOM_20"] = d[RET[:5]].mean(1), d[RET].mean(1)
    return d

S = pd.concat([date_state(X), date_state(Xt)])
is_test = np.r_[np.zeros(X.TS.nunique()), np.ones(Xt.TS.nunique())].astype(int)
grp = np.r_[S.index[:X.TS.nunique()].map(seg).values, 10_000 + np.arange(Xt.TS.nunique())]
p = np.zeros(len(S))
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(S, is_test, grp):
    p[b] = clf.fit(S.fillna(0).values[a], is_test[a]).predict_proba(S.fillna(0).values[b])[:, 1]
auc = roc_auc_score(is_test, p)
w_date = pd.Series(p / (1 - p), index=S.index)[is_test == 0]
w_date = (w_date / w_date.mean()).clip(0.2, 5)
w_row = pd.Series(ts).map(w_date).values
print(f"domain classifier train-date vs test-date: out-of-fold AUC {auc:.3f} (0.5 = no measurable shift)")
print(f"test-likeness weight quantiles 5/50/95%: {w_date.quantile([.05, .5, .95]).round(2).tolist()}", flush=True)

# ---- 2. folds: lockbox = 1/6 of the segments, 5-fold CV on the rest -------------------
research, lock = list(GroupKFold(6).split(X, groups=seg_id))[-1]
folds = [(research[a], research[b]) for a, b in GroupKFold(5).split(research, groups=seg_id[research])]
print(f"research {len(research):,} rows, lockbox {len(lock):,} rows ({len(set(seg_id[lock]))} segments)", flush=True)


def pick(tr, variant):
    """Rows of tr kept for training, and their weights."""
    dates = pd.Series(ts[tr]).unique()
    if variant.startswith("rand"):
        keep = np.random.default_rng(0).choice(dates, int(len(dates) * int(variant[4:]) / 100), replace=False)
    elif variant.startswith("like"):
        keep = w_date[dates].sort_values(ascending=False).index[:int(len(dates) * int(variant[4:]) / 100)]
    else:
        keep = dates
    sub = tr[np.isin(ts[tr], keep)]
    return sub, (w_row[sub] if variant == "wfull" else None)


def fit_predict(tr, va_F, va_alloc, weight=None, rounds=500):
    """Binary LightGBM + target encoding fitted on tr only (inner out-of-fold for the train rows)."""
    te_tr = np.full((len(tr), 2), np.nan)
    for a, b in GroupKFold(4).split(tr, groups=seg_id[tr]):
        te_tr[b] = target_encoding(alloc[tr[a]], y.values[tr[a]], alloc[tr[b]]).values
    te_va = target_encoding(alloc[tr], y.values[tr], va_alloc).values
    Ftr = np.c_[F.values[tr], te_tr]
    model = lgb.train(PARAMS, lgb.Dataset(Ftr, yb[tr], weight=weight), rounds)
    return model.predict(np.c_[va_F.values, te_va])


VARIANTS = ["full", "rand50", "rand25", "like50", "like25", "wfull"]
oof = {v: np.full(len(X), np.nan) for v in VARIANTS}
fold_acc = {v: [] for v in VARIANTS}
t0 = time.time()
for k, (tr, va) in enumerate(folds):
    for v in VARIANTS:
        sub, w = pick(tr, v)
        oof[v][va] = fit_predict(sub, F.iloc[va], alloc[va], w)
        fold_acc[v].append(((oof[v][va] > 0.5) == yb[va]).mean() * 100)
    print(f"fold {k + 1}/5 done ({time.time() - t0:.0f}s): " +
          "  ".join(f"{v} {fold_acc[v][-1]:.2f}" for v in VARIANTS), flush=True)


# ---- 3. compare (per-date paired statistics, plain and test-like weighted) ------------
def per_date(pred, idx):
    return pd.Series((pred[idx] > 0.5) == yb[idx], index=ts[idx]).groupby(level=0).mean()

base = per_date(oof["full"], research)
wd = w_date[base.index]
rows = []
for v in VARIANTS:
    d = per_date(oof[v], research)
    diff = d - base
    wdiff = diff * wd / wd.mean()
    rows.append(dict(variant=v, cv_acc=np.mean(fold_acc[v]), fold_std=np.std(fold_acc[v]),
                     folds=" ".join(f"{s:.2f}" for s in fold_acc[v]),
                     delta=diff.mean() * 100,
                     t=diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if v != "full" else 0.0,
                     testlike_acc=np.average(d, weights=wd) * 100,
                     testlike_delta=wdiff.mean() * 100,
                     testlike_t=wdiff.mean() / (wdiff.std(ddof=1) / np.sqrt(len(wdiff))) if v != "full" else 0.0,
                     share_up=(oof[v][research] > 0.5).mean()))
res = pd.DataFrame(rows).round(3)
print("\n" + res.to_string(index=False), flush=True)

# ---- 4. lockbox, once: full, plus the best variant only if it clearly beats full ------
cand = res[(res.variant != "full") & ((res.t >= 2) | (res.testlike_t >= 2))].sort_values("testlike_t")
final = ["full"] + ([cand.variant.iloc[-1]] if len(cand) else [])
lock_pred = {}
for v in final:
    sub, w = pick(research, v)
    lock_pred[v] = fit_predict(sub, F.iloc[lock], alloc[lock], w)
    d = pd.Series((lock_pred[v] > 0.5) == yb[lock], index=ts[lock]).groupby(level=0).mean()
    wl = w_date[d.index]
    print(f"LOCKBOX {v:7s} acc {((lock_pred[v] > 0.5) == yb[lock]).mean() * 100:.2f}  "
          f"test-like weighted {np.average(d, weights=wl) * 100:.2f}  (always-up {yb[lock].mean() * 100:.2f})", flush=True)
    res.loc[res.variant == v, "lockbox_acc"] = round(((lock_pred[v] > 0.5) == yb[lock]).mean() * 100, 3)

(ROOT / "results").mkdir(exist_ok=True)
res.assign(domain_auc=round(auc, 3)).to_csv(ROOT / "results" / "20_train_subset.csv", index=False)

# ---- 5. submission with the retained training set, trained on every train segment -----
chosen = final[-1]
sub, w = pick(np.arange(len(X)), chosen)
pred = fit_predict(sub, Ft[F.columns], Xt.ALLOCATION.values, w)
(ROOT / "submissions").mkdir(exist_ok=True)
out = pd.DataFrame({"target": (pred > 0.5).astype(int)}, index=Xt.index)
out.to_csv(ROOT / "submissions" / f"20_{chosen}.csv")
print(f"saved submissions/20_{chosen}.csv ({len(out)} rows, share of 1: {out.target.mean():.3f}, "
      f"trained on {len(sub):,} rows)")
