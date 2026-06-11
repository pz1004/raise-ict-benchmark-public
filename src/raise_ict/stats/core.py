"""Small descriptive statistics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_array(values: object) -> np.ndarray:
    """Coerce tabular values to numeric and drop missing entries."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(numeric).dropna().to_numpy(dtype=float)


def mean_ci(values: object, confidence: float = 0.95) -> dict[str, float]:
    """Return a quantile confidence interval for non-empty numeric values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("mean_ci requires at least one value")
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(arr, [alpha, 1.0 - alpha])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high)}


def normal_mean_ci(values: object) -> dict[str, float]:
    """Return mean, sample standard deviation, and normal-approximation 95% CI."""
    arr = numeric_array(values)
    if arr.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ci_low": nan, "ci_high": nan}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(arr.std(ddof=1))
    half_width = 1.96 * std / float(np.sqrt(arr.size))
    return {"mean": mean, "std": std, "ci_low": mean - half_width, "ci_high": mean + half_width}
