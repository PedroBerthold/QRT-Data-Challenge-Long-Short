"""Step 25: use the other allocations of the date to denoise yesterday's return.

Step 24: at date level, the mean target of a group is predictable from the group's own mean RET_1
(correlation 0.19 for group 1, 0.10 for group 3), and several statistical factors have an
out-of-sample R2 of 1-2% on their own lag. A single allocation's RET_1 is a noisy proxy for that.
The factor features are refitted inside every fold (loadings and 2 coefficients per factor).
"""
from _research import *  # noqa
from src.factors import GroupFactors, group_context, NAMES
import src.features as SF

SF.BLOCKS["group_context"] = group_context
F_all["group_context"], Ft_all["group_context"] = group_context(X), group_context(Xt)
GF = GroupFactors(X, y, Xt, k=8)
B = ("raw", "benchmark")


def run(name, blocks, factors=False, k=8, **kw):
    t0 = time.time()
    GF.k = k
    Fv = feats(blocks).values
    oof, fp = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for tr, va in folds:
        Fx = np.c_[Fv, GF.features(tr)[0]] if factors else Fv
        oof[va] = fit_predict(Fx, tr, Fx[va], alloc[va], **kw)
        if factors:
            fp[va] = Fx[va, Fv.shape[1]]
    print(f"  [{name}] done in {time.time() - t0:.0f}s", flush=True)
    return oof, fp


oofs = {"base": run("base", B)[0]}
oofs["factors_k8"], fp8 = run("factors_k8", B, True)
oofs["factors_k3"], fp3 = run("factors_k3", B, True, k=3)
oofs["factors_k20"], fp20 = run("factors_k20", B, True, k=20)
oofs["group_context"] = run("group_context", B + ("group_context",))[0]
oofs["factors_k8+ctx+vol"] = run("factors_k8+ctx+vol", B + ("group_context", "volume"), True)[0]
for n, f in (("k3", fp3), ("k8", fp8), ("k20", fp20)):
    oofs[f"sign(FP_{n}) alone"] = np.where(np.isnan(f), 0.5, (f > 0).astype(float))
res = report(oofs)
res.to_csv(Path(__file__).resolve().parents[1] / "results" / "25_group_factors.csv", index=False)
pd.DataFrame(oofs).iloc[research].to_parquet(DATA_DIR / "oof_25.parquet")
