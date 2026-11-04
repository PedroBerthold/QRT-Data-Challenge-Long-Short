"""Step 12: submission built ONLY from the leak method, to measure its accuracy on the test
through the leaderboard (we have no test labels locally).

For every test row: find the rows 1 and 2 days later (shifts k=1, 2, the most accurate on train),
read the noisy copies of the target and combine them (weighted mean). NO similarity threshold:
every row uses its best match, so the score measures the method alone (on train this gives ~89%,
step 10). With a threshold, most test rows would fall back to a default and the score would only
reflect that default.

Expected: if step 11 is right (no true match for test rows), the public score should be ~50%.
If the leak worked on test, it would be ~85-89% (what step 10 measures on train).
Runtime: ~5 min (2 shifts x 31,870 rows). Output: submissions/leak_only.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from _leak_common import *  # noqa
from src.matching import match, to_frames, aggregate, RET_COLS

SHIFTS_SUB = [1, 2]
STRATEGY, THR = "wmean", -1.0   # no threshold: every row uses its best match

X, y, Xt, P = pool()
path = DATA_DIR / "matches_test_sub.npz"
if path.exists():
    z = np.load(path)
    m = {k: dict(idx=z[f"idx{k}"], sim=z[f"sim{k}"], sim2=z[f"sim2{k}"], read=z[f"read{k}"]) for k in SHIFTS_SUB}
else:
    print(f"Matching {len(Xt):,} test rows against {len(P):,} pool rows, shifts {SHIFTS_SUB}")
    m = match(Xt[RET_COLS].values, P[RET_COLS].values, SHIFTS_SUB, MAX_LEN,
              q_index=Xt.index.values, p_index=P.index.values)
    np.savez(path, **{f"{key}{k}": m[k][key] for k in SHIFTS_SUB for key in ("idx", "sim", "sim2", "read")})

ks, read, sim, sim2 = to_frames(m)
est, cov = aggregate(read, sim, sim2, STRATEGY, THR)
pred = np.where(cov, est > 0, True).astype(int)

print(f"Median best similarity k=1: {np.median(sim[:, 0]):.4f}  (train ~0.99, no-match ~0.85)")
print(f"Rows with a match above {THR}: {cov.mean():.1%}  | share of 1: {pred.mean():.3f}")

out = Path(__file__).resolve().parents[1] / "submissions"
out.mkdir(exist_ok=True)
pd.DataFrame({"target": pred}, index=Xt.index).to_csv(out / "leak_only.csv")
print("Saved submissions/leak_only.csv -> upload it on the challenge page and read the public score")
