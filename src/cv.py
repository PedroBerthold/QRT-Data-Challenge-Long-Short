"""Cross-validation by groups of dates. All rows of a date stay in the same fold."""
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold


def date_folds(ts: pd.Series, n_splits=5, seed=0):
    """Random KFold on dates (what the benchmark does)."""
    ud = ts.unique()
    out = []
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(ud):
        out.append((np.where(ts.isin(ud[a]))[0], np.where(ts.isin(ud[b]))[0]))
    return out


def segment_folds(ts: pd.Series, seg: pd.Series, n_splits=5):
    """GroupKFold on chronological segments (block time CV, stricter)."""
    groups = ts.map(seg).values
    return list(GroupKFold(n_splits).split(np.zeros(len(ts)), groups=groups))


LGB_PARAMS = dict(objective="mse", learning_rate=0.01, max_depth=3,
                  num_threads=-1, verbosity=-1, seed=42)


def accuracy(pred, y, thr=0.0):
    return float(((pred > thr) == (np.asarray(y) > 0)).mean())


@dataclass
class CVResult:
    label: str
    scores: list            # accuracy per fold (0-1)
    oof: np.ndarray         # out-of-fold prediction for every train row
    threshold: float        # 0 for regression, 0.5 for binary
    importance: pd.DataFrame = None   # gain per feature (rows) x fold (columns)
    shap: pd.DataFrame = None         # mean |SHAP| per feature x fold
    folds: list = field(default_factory=list)

    @property
    def mean(self):
        return float(np.mean(self.scores) * 100)

    @property
    def std(self):
        return float(np.std(self.scores) * 100)

    @property
    def pred_label(self):
        return (self.oof > self.threshold).astype(int)


def cv_lgbm(F: pd.DataFrame, y: pd.Series, cols, folds, params=None,
            rounds=500, binary=False, fold_features=None, label="",
            shap_sample=20000):
    """Train one LightGBM per fold. Returns a CVResult.

    fold_features: optional function (train_idx, valid_idx) -> (F_train_extra, F_valid_extra)
    for features that must be computed inside the fold (e.g. target encoding).
    shap_sample: number of validation rows used to compute SHAP values per fold (0 = skip).
    """
    p = {**LGB_PARAMS, **(params or {})}
    if binary:
        p["objective"] = "binary"
    thr = 0.5 if binary else 0.0
    yv = np.asarray(y)
    oof = np.zeros(len(yv))
    scores, imps, shaps = [], {}, {}
    t0 = time.time()
    print(f"[{label}] training {len(folds)} folds, {len(cols)} features, {rounds} trees...", flush=True)
    for k, (tr, va) in enumerate(folds):
        Ftr, Fva = F.iloc[tr][cols], F.iloc[va][cols]
        if fold_features is not None:
            etr, eva = fold_features(tr, va)
            Ftr = pd.concat([Ftr, etr], axis=1)
            Fva = pd.concat([Fva, eva], axis=1)
        lab = (yv[tr] > 0).astype(int) if binary else yv[tr]
        model = lgb.train(p, lgb.Dataset(Ftr, lab), rounds)
        pred = model.predict(Fva)
        oof[va] = pred
        scores.append(accuracy(pred, yv[va], thr))
        print(f"  fold {k + 1}/{len(folds)}: acc {scores[-1] * 100:.2f}%  ({time.time() - t0:.0f}s elapsed)", flush=True)
        imps[f"fold{k + 1}"] = pd.Series(model.feature_importance("gain"), index=Ftr.columns)
        if shap_sample:
            sample = Fva.sample(min(shap_sample, len(Fva)), random_state=0)
            contrib = model.predict(sample, pred_contrib=True)[:, :-1]   # last column = bias
            shaps[f"fold{k + 1}"] = pd.Series(np.abs(contrib).mean(0), index=Ftr.columns)
    res = CVResult(label, scores, oof, thr, pd.DataFrame(imps),
                   pd.DataFrame(shaps) if shaps else None, folds)
    print(f"{label:45s} acc {res.mean:.2f} ± {res.std:.2f}  folds {[round(x * 100, 2) for x in scores]}",
          flush=True)
    return res
