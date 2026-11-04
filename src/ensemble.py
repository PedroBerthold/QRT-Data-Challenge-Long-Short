"""Combine models that output predictions on different scales (probabilities, scaled returns)."""
import numpy as np


def combine(preds, centers, weights=None):
    """Average of standardised signals: (pred - center) / std(pred).
    center = decision threshold of each model (0.5 for binary, 0 for regression).
    Returns a signal whose sign is the combined prediction (threshold 0)."""
    weights = weights or [1.0] * len(preds)
    out = np.zeros(len(preds[0]))
    for p, c, w in zip(preds, centers, weights):
        z = (np.asarray(p) - c)
        out += w * z / (z.std() + 1e-12)
    return out / sum(weights)
