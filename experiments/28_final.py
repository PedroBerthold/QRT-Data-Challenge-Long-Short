"""Step 28: final model = binary LightGBM + allocation target encoding with the strong regularisation
retained in step 27. Segment CV for comparison with earlier steps, the lockbox (scored once, paired
against the step-4 model), then the submission trained on every train row with 5 bagging seeds.
"""
from _research import *  # noqa

B = ("raw", "benchmark")
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)
Fv, Ftv = feats(B).values, feats(B, test=True)[feats(B).columns].values

oofs = {"base": run_cv("step4_model", B), "strong_reg": run_cv("strong_reg", B, params=REG)}
res = report(oofs)
res.to_csv(Path(__file__).resolve().parents[1] / "results" / "28_final.csv", index=False)

p4 = fit_predict(Fv, research, Fv[lock], alloc[lock])
pr = fit_predict(Fv, research, Fv[lock], alloc[lock], params=REG, seeds=(1, 2, 3, 4, 5))
pdx = lambda p: pd.Series((p > 0.5) == yb[lock], index=ts[lock]).groupby(level=0).mean()
print(f"LOCKBOX ({len(lock):,} rows, {len(set(ts[lock]))} dates): step4 {((p4 > 0.5) == yb[lock]).mean() * 100:.2f}  "
      f"strong_reg {((pr > 0.5) == yb[lock]).mean() * 100:.2f}  paired t {tstat(pdx(pr) - pdx(p4)):.2f}  "
      f"rule sign(RET_1) {((X.RET_1.values[lock] > 0) == yb[lock]).mean() * 100:.2f}  always up {yb[lock].mean() * 100:.2f}")

every = np.arange(len(X))
t4 = fit_predict(Fv, every, Ftv, Xt.ALLOCATION.values)
tr_ = fit_predict(Fv, every, Ftv, Xt.ALLOCATION.values, params=REG, seeds=(1, 2, 3, 4, 5))
sub = pd.DataFrame({"target": (tr_ > 0.5).astype(int)}, index=Xt.index)
out = Path(__file__).resolve().parents[1] / "submissions"
out.mkdir(exist_ok=True)
sub.to_csv(out / "28_strong_reg.csv")
print(f"saved submissions/28_strong_reg.csv: {len(sub)} rows, share of 1 {sub.target.mean():.3f}; "
      f"agreement with the step-4 model {((t4 > 0.5) == (tr_ > 0.5)).mean():.3f}; "
      f"agreement with sign(RET_1) {((Xt.RET_1.values > 0) == (tr_ > 0.5)).mean():.3f}")
