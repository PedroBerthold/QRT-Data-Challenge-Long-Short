"""Group factor model (step 25).

Inside each GROUP, the cross-section of yesterday's returns is projected on K fixed statistical
factors (PCA over allocations). Each factor's next-day return is regressed on its own last two
returns (2 coefficients per factor, fitted on the training dates only). For every row this gives

    FP        predicted next-day return = sum_k loading_k * (c1_k f_k[t] + c2_k f_k[t-1])
    FP_1      same with the first factor only
    COMMON_1  part of RET_1 explained by the group factors (denoised by the other allocations)
    IDIO_1    RET_1 - COMMON_1

Works on isolated dates: it only needs the cross-section of the date.
"""
import numpy as np
import pandas as pd

NAMES = ["FP", "FP_1", "COMMON_1", "IDIO_1"]


class GroupFactors:
    def __init__(self, X, y, Xt, k=8):
        A = pd.concat([X[["TS", "ALLOCATION", "GROUP", "RET_1", "RET_2"]],
                       Xt[["TS", "ALLOCATION", "GROUP", "RET_1", "RET_2"]]])
        self.n_train, self.k = len(X), k
        self.dc, dates = pd.factorize(A.TS)
        self.ac, allocs = pd.factorize(A.ALLOCATION)
        shape = (len(dates), len(allocs))
        self.R1, self.R2, self.Y = (np.full(shape, np.nan) for _ in range(3))
        self.R1[self.dc, self.ac] = A.RET_1.values
        self.R2[self.dc, self.ac] = A.RET_2.values
        self.Y[self.dc[:len(X)], self.ac[:len(X)]] = y.values
        grp = pd.Series(A.GROUP.values).groupby(self.ac).first()
        self.members = [np.where(grp.values == g)[0] for g in np.unique(grp.values)]

    def features(self, fit_rows):
        """fit_rows: positions of the train rows allowed for fitting. Returns (train_features, test_features)."""
        fit_dates = np.zeros(self.R1.shape[0], bool)
        fit_dates[np.unique(self.dc[fit_rows])] = True
        out = {n: np.full(self.R1.shape, np.nan) for n in NAMES}
        for cols in self.members:
            r1, r2, yy = self.R1[:, cols], self.R2[:, cols], self.Y[:, cols]
            present = np.isfinite(r1).mean(1) > 0.5
            fit = fit_dates & present & (np.isfinite(yy).mean(1) > 0.5)
            sd = np.nanstd(r1[fit], 0) + 1e-12
            z1, z2, zy = np.nan_to_num(r1 / sd), np.nan_to_num(r2 / sd), np.nan_to_num(yy / sd)
            V = np.linalg.svd(z1[fit], full_matrices=False)[2][:self.k].T
            f1, f2, fy = z1 @ V, z2 @ V, zy @ V
            fp = np.zeros_like(f1)
            for j in range(V.shape[1]):
                Z = np.c_[f1[fit, j], f2[fit, j]]
                c = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(2) * len(Z) * Z.var(), Z.T @ fy[fit, j])
                fp[:, j] = c[0] * f1[:, j] + c[1] * f2[:, j]
            common = (f1 @ V.T) * sd
            vals = {"FP": (fp @ V.T) * sd, "FP_1": np.outer(fp[:, 0], V[:, 0]) * sd,
                    "COMMON_1": common, "IDIO_1": r1 - common}
            for n in NAMES:
                v = vals[n].copy()
                v[~present] = np.nan
                out[n][:, cols] = v
        F = np.c_[tuple(out[n][self.dc, self.ac] for n in NAMES)]
        return F[:self.n_train], F[self.n_train:]


def group_context(X):
    """Yesterday's and the day before's mean return of each of the four groups, on the row's date."""
    o = pd.DataFrame(index=X.index)
    o["GROUP_ID"] = pd.to_numeric(X.GROUP.astype(str).str.extract(r"(\d+)")[0], errors="coerce").values
    for j in (1, 2):
        m = X[f"RET_{j}"].groupby([X.TS, X.GROUP]).mean().unstack()
        for g in m.columns:
            o[f"G{g}_RET_{j}"] = X.TS.map(m[g]).values
    return o
