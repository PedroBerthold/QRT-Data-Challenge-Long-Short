# QRT Data Challenge: sign of next-day allocation returns

[Challenge page](https://challengedata.ens.fr/challenges/167). For each (date, allocation), predict whether the allocation's next-day return is positive from 20 days of returns, signed volumes and median turnover. Metric: accuracy. Benchmark (LightGBM): 0.5079 on the public leaderboard.

## Setup

```bash
pip install -r requirements.txt
```

Download the challenge files into `data/` (`X_train*.csv`, `y_train*.csv`, `X_test*.csv`). The folder is git-ignored: the data must not be published. The first run caches them as parquet.

## Running

From the repo root, in order:

| Script | What it does |
|---|---|
| `experiments/00_build_segments.py` | Rebuilds the chronological order of the shuffled dates (needed for segment CV) |
| `experiments/00b_eda.py` | Looks at the data: target distribution, per-allocation drift, missing values, feature correlations |
| `experiments/01_benchmark.py` | Reproduces the benchmark, compares random-date CV vs segment CV |
| `experiments/02_target_encoding.py` | Tests whether each allocation has a persistent drift |
| `experiments/03_feature_blocks.py` | Adds feature blocks one at a time |
| `experiments/04_objective.py` | Compares MSE regression vs binary classification, plus regularisation |
| `experiments/05_submission.py` | Trains the chosen model (`MODEL` = B, D, ensemble or benchmark) on the full train set and writes `submissions/<MODEL>.csv` |
| `experiments/07_two_stage.py` | Market model on one row per date + relative model per allocation, additive (V1) or stacked into B (V2) |
| `experiments/06_ensemble.py` | Combines the two best settings of step 4 (binary + TE, target/vol + strong reg + TE) |
| `experiments/10_leak_validate.py` | Measures the look-ahead leak inside the train set (see findings) |
| `experiments/11_leak_test.py` | Checks whether the same leak exists for test rows |
| `experiments/12_leak_submission.py` | Submission built only from the leak method, to read its real accuracy on the leaderboard |

Full-mode scores are appended to `results/results.csv`. Figures are saved in `figures/<experiment>/` and opened at the end of each script (`SHOW=0` to only save them).

## Figures

| Figure | Question it answers |
|---|---|
| `fold_scores` | Accuracy per fold for each model, against the "always predict 1" line. Is a gain consistent across periods? |
| `importance_shap` | Which features move the predictions, averaged over folds. Bars = min/max across folds: a long bar means the feature matters only in some periods |
| `confidence` | Accuracy by decile of prediction strength. If the model knows when it is right, the right-hand deciles are higher |
| `errors` | Accuracy per period, per GROUP and per allocation (grey band = what pure chance would give) |
| `market_timing` | Per date, share predicted 1 vs share actually positive. Shows whether the model mostly bets on the whole market's direction |
| `compare` | Accuracy difference between two models, date by date |
| `coefs` | Coefficients of the logistic regression on standardised features |

## Is B really better than A?

`src/stats.compare` prints three numbers for two models evaluated on the same folds:

- `delta_pts`: difference in mean CV accuracy.
- `fold_wins` and `p_folds_by_chance`: on how many folds B beats A, and how likely that is if both models were equally good. 5/5 happens by chance 3% of the time, 4/5 or better 19%.
- `t_stat_per_date`: paired t-statistic on the per-date accuracy difference (about 2,500 dates). Above ~2, the gain is unlikely to be noise.

Keep a change only if the gain is positive on most folds and the t-stat is around 2 or more.

`QUICK=1 python experiments/01_benchmark.py` runs on 20% of dates with 100 trees as a smoke test. Those numbers are not meaningful.

## Code

- `src/data.py`: loading and parquet cache
- `src/segments.py`: date-order reconstruction
- `src/cv.py`: CV folds grouped by date or by segment, `cv_lgbm`
- `src/features.py`: feature blocks (`raw`, `benchmark`, `time_series`, `cross_section`) and target encoding
- `src/log.py`: results log

## Findings so far

- **Date order is recoverable inside the train set.** Dates are shuffled, but `RET_2..RET_7` at day t+1 matches `RET_1..RET_6` at day t up to small noise (median similarity 0.98). The train rebuilds into 40 chronological segments.
- **Look-ahead leak inside train, not usable on test.** The row of day t+k holds the target of day t (in `RET_k`). Inside the train, finding that row by history matching recovers the sign of the target about 85% of the time. Test rows behave exactly like queries with no true match (best similarity ~0.85 with 19 lags, vs ~0.99 for train rows): their continuation is not in the files. The 0.78-0.82 public scores therefore come from something not identified yet.
- **Segment CV vs random-date CV.** The benchmark scores about 52.1-52.2% under both schemes. Segment CV is much less stable across folds (±0.56 vs ±0.07), so the gap to the 50.8% leaderboard score is mostly a regime effect, not CV leakage.
- **The leaderboard is noisy.** With 31,870 test rows, the standard error of the accuracy is about 0.28 pt. Select models on CV, not on public LB.
- **Date-level timing has a low ceiling.** Even knowing the majority sign of each date perfectly only gives 57% accuracy.

## Validation rules

- All rows of a date stay in the same fold (rows of one date share the same market move).
- Anything fitted on the target (target encoding, scaling fitted on y) is computed inside the training fold only.
- Use `src/stats.compare` before keeping a change (see above).
