from datetime import datetime
from pathlib import Path
import csv

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.csv"


def log_result(experiment, cv, mean, std, n_features, notes=""):
    RESULTS.parent.mkdir(exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "experiment", "cv", "acc_mean", "acc_std", "n_features", "notes"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), experiment, cv,
                    f"{mean:.2f}", f"{std:.2f}", n_features, notes])
