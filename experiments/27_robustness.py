"""Step 27: which model survives the passage of time?

Step 26: trained on the past only, the step-4 model loses a full point (51.2% on split A against
52.2% in segment CV) and no longer beats the rule sign(RET_1) there; stronger regularisation is the
only change that helps forward. This step walks the regularisation / simplicity frontier on the
three forward splits. Reference = strong_reg (the REG parameters of experiments/05_submission.py).
"""
from sklearn.linear_model import LogisticRegression
from _research import *  # noqa

B = ("raw", "benchmark")
REG = dict(min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1)
Fb = feats(B)
cols = list(Fb.columns)
no_date = [c for c in cols if not c.startswith("ALLOCATIONS_")]
no_sv = [c for c in cols if not c.startswith("SIGNED_VOLUME")]
mono = [1 if c == "RET_1" else 0 for c in cols] + [0, 0]
Fn = F_all["normalised"].fillna(0).values
Fn = np.clip(Fn / (Fn.std(0) + 1e-12), -5, 5)


def lgbm(c=cols, te=True, rounds=500, seeds=(42,), **p):
    Fv = Fb[c].values
    return lambda tr, va: fit_predict(Fv, tr, Fv[va], alloc[va], params={**REG, **p}, rounds=rounds, te=te, seeds=seeds)


def logit(tr, va):
    return LogisticRegression(C=0.01, max_iter=300).fit(Fn[tr], yb[tr]).predict_proba(Fn[va])[:, 1]


def blend(f, g, w=0.5):
    return lambda tr, va: w * f(tr, va) + (1 - w) * g(tr, va)


base_f = lambda tr, va: fit_predict(Fb.values, tr, Fb.values[va], alloc[va])
VAR = {
    "step4_model": base_f,
    "strong_reg": lgbm(),
    "very_strong_reg": lgbm(min_data_in_leaf=6000, feature_fraction=0.3, lambda_l2=50.0),
    "reg_250_trees": lgbm(rounds=250),
    "reg_1000_trees": lgbm(rounds=1000),
    "reg_depth2": lgbm(max_depth=2),
    "reg_no_date_feats": lgbm(no_date),
    "reg_no_raw_volume": lgbm(no_sv),
    "reg_noTE": lgbm(te=False),
    "reg_monotone_RET1": lgbm(monotone_constraints=mono),
    "reg_3seeds": lgbm(seeds=(1, 2, 3)),
    "logit_normalised": logit,
    "blend_reg+logit": blend(lgbm(), logit),
    "rule sign(RET_1)": lambda tr, va: (X.RET_1.values[va] > 0).astype(float),
}
res, _ = forward_eval(VAR, base="strong_reg")
res.round(3).to_csv(Path(__file__).resolve().parents[1] / "results" / "27_robustness.csv", index=False)
