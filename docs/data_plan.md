# Data plan (branch 2026-11-14)

The challenge files are not in the repository (they must not be published), so they
travel as chat attachments and stay git-ignored on both sides.

## What is needed

Everything: the 527,073 train rows with their target, the 31,870 test rows, the sample
submission. No sampling.

Reason: the unit of independent information is the date, not the row (about 2,500 dates,
40 chronological segments once the timeline is rebuilt). Any row-level or date-level
extract breaks the segment chains that the validation relies on, and with an edge of
half a point of accuracy, a subsample cannot separate a real gain from noise.

## How it is sliced for transport

`python scripts/pack_data.py` writes `data/pack/`:

| File | Content |
|---|---|
| `train_partNN.parquet` | X_train + `TARGET`, consecutive row blocks in the original order |
| `test_partNN.parquet` | X_test |
| `sample_submission.csv.gz` | as provided |
| `audit.json` | shapes, dtypes, missing values, counts |

Floats are stored as float32 with zstd (signs and 7 significant digits kept), each file
under 20 MB (`--max-mb` to change). Expect about 8 files, 130 MB in total.
`python scripts/unpack_data.py --pack-dir ...` rebuilds the caches read by `src/data.py`.

## How it is sliced for research

- Timeline: `src/segments.py`, unchanged (shifted-window matching, about 40 segments).
- Model selection: 5 folds of whole segments, paired per-date test, as in the existing CV.
- New: the segments are cut in two before any new experiment. A research set (about
  80% of the dates, where the 5-fold CV runs) and a lockbox of whole segments (about 20%)
  that is scored once, on the final candidates only. About fifteen configurations have
  already been selected on the full CV, so its 52.50% is optimistic by an unknown amount.
- Test-like slice: the test dates are isolated (no neighbour in the files) and their
  recent history is more positive than the train average. Train dates are reweighted
  to the test distribution of market state (features only, no label) to get a second
  score that looks like the test period.
