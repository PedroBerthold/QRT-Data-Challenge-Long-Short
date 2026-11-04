"""Step 1: reproduce the QRT benchmark and compare two CV schemes.

Expected: ~52.2% with random date KFold (the notebook reports 52.22%).
Question: is segment CV (block time) more pessimistic? More variable across folds?
"""
from _common import *  # noqa
from src.features import build
from src.cv import date_folds, segment_folds, cv_lgbm
from src.segments import load_segments
from src.log import log_result

X, y = train_data()
F = build(X, ("raw", "benchmark"))
cols = list(F.columns)
seg = load_segments()

for name, folds in [("date", date_folds(X.TS)), ("segment", segment_folds(X.TS, seg))]:
    m, s, _ = cv_lgbm(F, y, cols, folds, rounds=ROUNDS, label=f"benchmark | {name} CV")
    if not QUICK:
        log_result("01_benchmark", name, m, s, len(cols))
