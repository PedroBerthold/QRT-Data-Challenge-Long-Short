"""Step 23: one change at a time against the final model of step 4, scored on all research
dates and on the late era (where the test dates sit, step 22).

Hypotheses
  noTE        allocation drift is half as stable across time as across random halves (step 22): is TE still useful late?
  late_x3     the dynamics drifts (group 3 loses its RET_1 effect late): weight the late era more
  context     GROUP and the part of the universe trading that day
  decomp      common vs specific part of the recent return
  normalised  level-free features, robust to the lower-turnover regime of the test
  volume      signed-volume summaries
  capacity    more capacity with stronger regularisation
"""
from _research import *  # noqa

CAP = dict(learning_rate=0.03, max_depth=5, num_leaves=24, min_data_in_leaf=1500, feature_fraction=0.5,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0)
B = ("raw", "benchmark")
w_late = np.where(late, 3.0, 1.0)
print(f"research {len(research):,} rows of which late era {late[research].mean():.1%}; "
      f"lockbox late share {late[lock].mean():.1%}", flush=True)
oofs = {"base": run_cv("base", B)}
oofs["noTE"] = run_cv("noTE", B, te=False)
oofs["late_x3"] = run_cv("late_x3", B, weight=w_late)
for blk in ("context", "decomp", "normalised", "volume"):
    oofs[blk] = run_cv(blk, B + (blk,))
oofs["capacity"] = run_cv("capacity", B, params=CAP, rounds=600)
res = report(oofs)
res.to_csv(ROOT / "results" / "23_feature_and_era.csv", index=False) if "ROOT" in globals() else \
    res.to_csv(Path(__file__).resolve().parents[1] / "results" / "23_feature_and_era.csv", index=False)
pd.DataFrame(oofs).iloc[research].to_parquet(DATA_DIR / "oof_23.parquet")

# decision threshold: is 0.5 the right cut on out-of-fold probabilities?
o = oofs["base"][research]
for name, thr in (("0.5", 0.5), ("median", np.median(o)), ("base-rate quantile", np.quantile(o, 1 - yb[research].mean()))):
    print(f"threshold {name:18s} ({thr:.4f}): acc {((o > thr) == yb[research]).mean() * 100:.3f}")
