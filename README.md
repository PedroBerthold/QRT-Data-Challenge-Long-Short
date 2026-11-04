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
| `experiments/01_benchmark.py` | Reproduces the benchmark, compares random-date CV vs segment CV |
| `experiments/02_target_encoding.py` | Tests whether each allocation has a persistent drift |
| `experiments/03_feature_blocks.py` | Adds feature blocks one at a time |
| `experiments/04_objective.py` | Compares MSE regression vs binary classification, plus regularisation |
| `experiments/05_submission.py` | Trains on the full train set and writes `submissions/<name>.csv` |

Full-mode scores are appended to `results/results.csv`.

`QUICK=1 python experiments/01_benchmark.py` runs on 20% of dates with 100 trees as a smoke test. Those numbers are not meaningful.

## Code

- `src/data.py`: loading and parquet cache
- `src/segments.py`: date-order reconstruction
- `src/cv.py`: CV folds grouped by date or by segment, `cv_lgbm`
- `src/features.py`: feature blocks (`raw`, `benchmark`, `time_series`, `cross_section`) and target encoding
- `src/log.py`: results log

## Findings so far

- **Date order is recoverable, test is not linked to train.** Dates are shuffled, but `RET_2..RET_7` at day t+1 matches `RET_1..RET_6` at day t up to small noise (median similarity 0.98). The train rebuilds into 40 chronological segments. None of the 120 test dates links to any other date, so there is no target leakage and the test covers a separate period.
- **Segment CV vs random-date CV.** The benchmark scores about 52.1-52.2% under both schemes. Segment CV is much less stable across folds (±0.56 vs ±0.07), so the gap to the 50.8% leaderboard score is mostly a regime effect, not CV leakage.
- **The leaderboard is noisy.** With 31,870 test rows, the standard error of the accuracy is about 0.28 pt. Select models on CV, not on public LB.
- **Date-level timing has a low ceiling.** Even knowing the majority sign of each date perfectly only gives 57% accuracy.

## Validation rules

- All rows of a date stay in the same fold (rows of one date share the same market move).
- Anything fitted on the target (target encoding, scaling fitted on y) is computed inside the training fold only.
- A gain below ~0.3 pt is within noise: check that it holds on most folds.
