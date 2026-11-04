"""Step 0b: look at the data before modelling.
Figures in figures/00_eda/:
- target_overview: distribution of the next-day return, share of positive days per allocation,
  share of allocations positive per date (how much dates move together)
- feature_ic: rank correlation of each feature with the next-day return
- missing_values: which columns are missing, and how often
"""
from _common import *  # noqa
from src.features import build
from src import plots

X, y = train_data()
print(f"{len(X):,} rows, {X.TS.nunique():,} dates, {X.ALLOCATION.nunique()} allocations, groups {sorted(X.GROUP.unique())}")
print(f"Share of positive next-day returns: {(y > 0).mean():.4f}")

plots.target_overview(y, X)
plots.missing_values(X)
F = build(X, ("raw", "benchmark", "time_series", "cross_section"))
plots.feature_ic(F, y)
plots.show()
