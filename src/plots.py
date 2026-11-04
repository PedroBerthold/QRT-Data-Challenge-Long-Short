"""Figures. Every function saves a PNG in figures/<experiment>/ and returns the figure.
Call show() at the end of a script to open all windows (set SHOW=0 to only save)."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("SHOW", "1") == "0":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
BASE_RATE = 50.7      # always predicting 1 on train
BENCH_LB = 50.79      # benchmark on the public leaderboard
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, exp, name):
    d = FIG_DIR / exp
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
    return fig


def show():
    if os.environ.get("SHOW", "1") != "0":
        plt.show()


# ---------- data exploration ----------

def target_overview(y, X, exp="00_eda"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.clip(y, -0.015, 0.015), bins=120, color="#4c72b0")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set(title=f"Next-day return (clipped)  | share > 0: {(y > 0).mean():.3f}", xlabel="return")
    pos = (y > 0).groupby(X.ALLOCATION).mean().sort_values()
    ax[1].bar(range(len(pos)), pos.values, width=1.0, color="#55a868")
    ax[1].axhline(0.5, color="k", lw=1)
    ax[1].set(title="Share of positive days per allocation (sorted)", xlabel="allocation", ylim=(0.35, 0.7))
    day = (y > 0).groupby(X.TS).mean()
    ax[2].hist(day, bins=60, color="#c44e52")
    ax[2].set(title="Share of allocations positive, per date", xlabel="share positive that day")
    return _save(fig, exp, "target_overview")


def feature_ic(F, y, exp="00_eda", top=30):
    """Spearman correlation of each feature with the next-day return (information coefficient)."""
    ic = F.corrwith(pd.Series(np.asarray(y), index=F.index).rank(), method="spearman")
    ic = ic.reindex(ic.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(ic) + 1))
    ax.barh(ic.index, ic.values, color=np.where(ic.values > 0, "#55a868", "#c44e52"))
    ax.axvline(0, color="k", lw=1)
    ax.set(title="Rank correlation with next-day return (IC), top features", xlabel="IC")
    return _save(fig, exp, "feature_ic")


def missing_values(X, exp="00_eda"):
    na = X.isna().mean()
    na = na[na > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.25 * len(na) + 1))
    ax.barh(na.index, na.values * 100, color="#8172b2")
    ax.set(title="Missing values (%)", xlabel="%")
    return _save(fig, exp, "missing_values")


# ---------- model results ----------

def fold_scores(results, exp, name="fold_scores"):
    """Accuracy per fold for several CVResults, with the 'always 1' line and the CV mean."""
    n = len(results)
    k = len(results[0].scores)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * k), 4.5))
    for i, r in enumerate(results):
        x = np.arange(k) + i * width
        ax.bar(x, np.array(r.scores) * 100, width, label=f"{r.label}  ({r.mean:.2f} ± {r.std:.2f})")
    ax.axhline(BASE_RATE, color="k", ls="--", lw=1, label=f"always predict 1 ({BASE_RATE}%)")
    ax.set_xticks(np.arange(k) + width * (n - 1) / 2, [f"fold {i + 1}" for i in range(k)])
    lo = min(min(r.scores) for r in results) * 100
    hi = max(max(r.scores) for r in results) * 100
    ax.set(ylim=(min(lo, BASE_RATE) - 0.5, hi + 0.5), ylabel="accuracy (%)", title="Accuracy per CV fold")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, exp, name)


def importance(res, exp, top=25, kind="shap"):
    """Feature importance averaged over folds, with min-max across folds as error bars.
    kind='shap': mean |SHAP| (how much each feature moves predictions)
    kind='gain': LightGBM split gain."""
    df = res.shap if (kind == "shap" and res.shap is not None) else res.importance
    df = df.div(df.sum(0), axis=1) * 100
    m = df.mean(1).sort_values(ascending=False)[:top][::-1]
    lo, hi = df.loc[m.index].min(1), df.loc[m.index].max(1)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(m) + 1.2))
    ax.barh(m.index, m.values, xerr=[m - lo, hi - m], color="#4c72b0", ecolor="grey", capsize=2)
    ax.set(title=f"{res.label}: feature importance ({kind}, % of total, bars = min/max over folds)",
           xlabel="% of total importance")
    ax.title.set_fontsize(9)
    return _save(fig, exp, f"importance_{kind}_{_slug(res.label)}")


def confidence(res, y, exp, n_bins=10):
    """Does the model know when it is right? Accuracy by decile of prediction strength."""
    yv = np.asarray(y)
    strength = np.abs(res.oof - res.threshold)
    q = pd.qcut(strength, n_bins, labels=False, duplicates="drop")
    acc = pd.Series((res.pred_label == (yv > 0))).groupby(q).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(acc.index + 1, acc.values, color="#55a868")
    ax.axhline(50, color="k", lw=1)
    ax.axhline(res.mean, color="#4c72b0", ls="--", lw=1, label=f"overall {res.mean:.2f}%")
    ax.set(title=f"{res.label}: accuracy by confidence decile (1 = weakest prediction)",
           xlabel="decile of |prediction|", ylabel="accuracy (%)", ylim=(min(acc.min(), 49) - 1, acc.max() + 1))
    ax.title.set_fontsize(9)
    ax.legend()
    return _save(fig, exp, f"confidence_{_slug(res.label)}")


def errors_by(res, y, X, seg, exp):
    """Where does the model fail? Accuracy per segment (period), per GROUP, per allocation."""
    yv = np.asarray(y)
    ok = pd.Series(res.pred_label == (yv > 0), index=X.index)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    s = X.TS.map(seg)
    by_seg = ok.groupby(s.values).agg(["mean", "size"])
    by_seg = by_seg[by_seg["size"] > 2000].sort_values("mean")
    ax[0].bar(range(len(by_seg)), by_seg["mean"] * 100, color="#4c72b0")
    ax[0].axhline(50, color="k", lw=1)
    ax[0].set(title="Accuracy per chronological segment (period)", xlabel="segment (sorted)", ylabel="%",
              ylim=(by_seg["mean"].min() * 100 - 1, by_seg["mean"].max() * 100 + 1))
    by_grp = ok.groupby(X.GROUP).mean() * 100
    ax[1].bar(by_grp.index.astype(str), by_grp.values, color="#55a868")
    ax[1].axhline(50, color="k", lw=1)
    ax[1].set(title="Accuracy per GROUP", ylim=(by_grp.min() - 1, by_grp.max() + 1))
    by_alloc = ok.groupby(X.ALLOCATION).agg(["mean", "size"])
    ax[2].hist(by_alloc["mean"] * 100, bins=40, color="#c44e52", alpha=0.8, label="observed")
    n = by_alloc["size"].median()
    ax[2].axvspan(50 - 196 * np.sqrt(0.25 / n), 50 + 196 * np.sqrt(0.25 / n), color="grey", alpha=0.2,
                  label="95% range if pure chance")
    ax[2].set(title="Accuracy per allocation", xlabel="%")
    ax[2].legend(fontsize=8)
    fig.suptitle(res.label, fontsize=10)
    return _save(fig, exp, f"errors_{_slug(res.label)}")


def market_timing(res, y, X, exp):
    """Per date: share of 1 predicted vs share actually positive. Shows whether the model
    mostly bets on the direction of the whole market."""
    yv = np.asarray(y)
    d = pd.DataFrame({"pred": res.pred_label, "true": (yv > 0).astype(int), "TS": X.TS.values})
    g = d.groupby("TS").mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(g.true, g.pred, s=4, alpha=0.4)
    ax[0].set(xlabel="share of allocations actually positive", ylabel="share predicted 1",
              title=f"Per date (corr {g.true.corr(g.pred):.2f})")
    ax[1].hist(g.pred, bins=50, color="#8172b2")
    ax[1].set(title="Share predicted 1 per date", xlabel="share")
    fig.suptitle(res.label, fontsize=10)
    return _save(fig, exp, f"market_timing_{_slug(res.label)}")


def compare_dates(diff, label, exp):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(diff * 100, bins=80, color="#4c72b0")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(diff.mean() * 100, color="#c44e52", lw=2, label=f"mean {diff.mean() * 100:+.3f} pt")
    ax.set(title=f"{label}: accuracy difference per date", xlabel="B - A (pts)")
    ax.legend()
    return _save(fig, exp, f"compare_{_slug(label)}")


def coefficients(coefs: pd.Series, title, exp, top=25):
    c = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(c) + 1.2))
    ax.barh(c.index, c.values, color=np.where(c.values > 0, "#55a868", "#c44e52"))
    ax.axvline(0, color="k", lw=1)
    ax.set(title=title, xlabel="coefficient (standardised features)")
    return _save(fig, exp, f"coefs_{_slug(title)}")


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")[:60]
