"""Benchmark metrics and scoring."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import balanced_accuracy_score, f1_score


def _label_array(values: ArrayLike | Iterable[object]) -> np.ndarray:
    """Return a one-dimensional label array without forcing Python-list copies."""
    array = np.asarray(values)
    if array.ndim == 0 and not isinstance(values, (str, bytes)):
        array = np.asarray(tuple(values))
    return array


def classification_summary(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return clean utility components used by the benchmark result schema."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "clean_macro_f1": macro_f1,
        "clean_bal_acc": bal_acc,
        "utility": 0.5 * (macro_f1 + bal_acc),
    }


def service_cost(
    y_true: ArrayLike | Iterable[object],
    y_pred: ArrayLike | Iterable[object],
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
) -> float:
    """Compute normalized false-alarm and missed-detection service cost."""
    true = _label_array(y_true)
    pred = _label_array(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    benign_mask = true == 0
    malicious_mask = true != 0
    benign = int(benign_mask.sum()) or 1
    malicious = int(malicious_mask.sum()) or 1
    fp = int((benign_mask & (pred != 0)).sum())
    fn = int((malicious_mask & (pred == 0)).sum())
    return false_positive_cost * fp / benign + false_negative_cost * fn / malicious


def raise_score(
    clean_utility: float,
    robust_utility: float,
    p95_latency_ms: float,
    peak_mem_mb: float,
    service_cost_value: float,
    energy_per_flow_j: float = 0.0,
    latency_budget_ms: float = 10.0,
    latency_cap_ms: float = 100.0,
    memory_budget_mb: float = 256.0,
    memory_cap_mb: float = 2048.0,
    energy_budget_j: float = 0.001,
    energy_cap_j: float = 1.0,
    alpha: float = 0.35,
    beta: float = 0.65,
    latency_weight: float = 0.05,
    energy_weight: float = 0.05,
    memory_weight: float = 0.05,
    service_weight: float = 0.02,
) -> float:
    """Combine utility and manuscript-defined normalized edge/service penalties."""
    latency_penalty = edge_penalty(p95_latency_ms, latency_budget_ms, latency_cap_ms)
    energy_penalty = edge_penalty(energy_per_flow_j, energy_budget_j, energy_cap_j)
    memory_penalty = edge_penalty(peak_mem_mb, memory_budget_mb, memory_cap_mb)
    return (
        alpha * clean_utility
        + beta * robust_utility
        - latency_weight * latency_penalty
        - energy_weight * energy_penalty
        - memory_weight * memory_penalty
        - service_weight * service_cost_value
    )


def edge_penalty(value: float, threshold: float, cap: float) -> float:
    """Log-normalized edge penalty psi(z; z0, zmax) from the manuscript."""
    if value <= 0.0:
        return 0.0
    if threshold <= 0.0 or cap <= 0.0:
        raise ValueError("threshold and cap must be positive")
    denominator = math.log1p(cap / threshold)
    if denominator <= 0.0:
        return 0.0
    return min(1.0, math.log1p(value / threshold) / denominator)
