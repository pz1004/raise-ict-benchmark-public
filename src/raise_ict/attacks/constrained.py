"""Simple constrained feature perturbations for tabular smoke tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstrainedAttackConfig:
    """Configuration for feature-space constrained perturbations."""

    threat_id: str = "t1_constrained_feature"
    epsilon: float = 0.15
    mutable_features: list[str] = field(default_factory=list)
    nonnegative_features: list[str] = field(default_factory=list)
    seed: int = 0
    strategy: str = "random"
    n_candidates: int = 1
    budget_norm: str = "inf"


@dataclass(frozen=True)
class AttackValidityReport:
    """Validity evidence for one feature-space adversarial evaluation."""

    valid_mask: np.ndarray
    valid_count: int
    invalid_count: int
    budget_pass_rate: float
    bounds_pass_rate: float
    immutable_pass_rate: float
    relation_pass_rate: float
    validity_rate: float


@dataclass(frozen=True)
class AttackEvaluation:
    """Generated adversarial frame plus the validity checks applied to it."""

    x_adv: pd.DataFrame
    report: AttackValidityReport


def generate_constrained_perturbations(
    features: pd.DataFrame,
    config: ConstrainedAttackConfig,
    labels: pd.Series | None = None,
    score_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
    lower_bounds: Mapping[str, float] | None = None,
    upper_bounds: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Perturb mutable numeric features and keep nonnegative features valid."""
    rng = np.random.default_rng(config.seed)
    attacked = features.copy()
    mutable = _mutable_columns(attacked, config.mutable_features)

    target_index = _target_index(attacked, labels)
    if len(target_index) == 0:
        return attacked

    base_values = attacked.loc[target_index, mutable].to_numpy()
    scale = _scale_array(mutable, scales or {})
    if config.strategy == "score_search" and score_fn is not None and config.n_candidates > 1:
        best_frame = attacked.copy()
        best_scores = np.full(len(target_index), -np.inf, dtype=float)
        for _ in range(config.n_candidates):
            candidate = attacked.copy()
            noise = _scaled_uniform_noise(rng, config, mutable, len(target_index), scale)
            candidate.loc[target_index, mutable] = base_values + noise
            _project_budget_inplace(features, candidate, mutable, target_index, config, scale)
            _project_feature_bounds_inplace(candidate, lower_bounds, upper_bounds, mutable)
            _project_nonnegative_inplace(candidate, config, lower_bounds)
            scores = np.asarray(score_fn(candidate.loc[target_index]), dtype=float)
            improve = scores > best_scores
            if improve.any():
                improved_index = target_index[improve]
                best_frame.loc[improved_index, mutable] = candidate.loc[improved_index, mutable]
                best_scores[improve] = scores[improve]
        attacked = best_frame
    else:
        noise = _scaled_uniform_noise(rng, config, mutable, len(target_index), scale)
        attacked.loc[target_index, mutable] = base_values + noise

    _project_budget_inplace(features, attacked, mutable, target_index, config, scale)
    _project_feature_bounds_inplace(attacked, lower_bounds, upper_bounds, mutable)
    _project_nonnegative_inplace(attacked, config, lower_bounds)
    return attacked


def evaluate_constrained_perturbations(
    features: pd.DataFrame,
    config: ConstrainedAttackConfig,
    labels: pd.Series | None = None,
    score_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
    lower_bounds: Mapping[str, float] | None = None,
    upper_bounds: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
    relationship_check: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> AttackEvaluation:
    """Generate perturbations and return the full manuscript validity report."""
    mutable = _mutable_columns(features, config.mutable_features)
    x_adv = generate_constrained_perturbations(
        features,
        config,
        labels=labels,
        score_fn=score_fn,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        scales=scales,
    )
    report = evaluate_validity(
        clean=features,
        attacked=x_adv,
        config=config,
        mutable_features=mutable,
        labels=labels,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        scales=scales,
        relationship_check=relationship_check,
    )
    return AttackEvaluation(x_adv=x_adv, report=report)


def evaluate_validity(
    clean: pd.DataFrame,
    attacked: pd.DataFrame,
    config: ConstrainedAttackConfig,
    mutable_features: list[str] | None = None,
    labels: pd.Series | None = None,
    lower_bounds: Mapping[str, float] | None = None,
    upper_bounds: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
    relationship_check: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> AttackValidityReport:
    """Evaluate budget, bounds, immutable-field, and relation validity masks."""
    mutable = mutable_features or _mutable_columns(clean, config.mutable_features)
    target_index = _target_index(clean, labels)
    budget_mask = _budget_mask(clean, attacked, mutable, target_index, config, _scale_array(mutable, scales or {}))
    bounds_mask = _bounds_mask(attacked, config, lower_bounds or {}, upper_bounds or {})
    immutable_mask = _immutable_mask(clean, attacked, mutable)
    relation_mask = _relation_mask(attacked, relationship_check)
    valid_mask = budget_mask & bounds_mask & immutable_mask & relation_mask
    valid_count = int(valid_mask.sum())
    invalid_count = int(len(valid_mask) - valid_count)
    return AttackValidityReport(
        valid_mask=valid_mask,
        valid_count=valid_count,
        invalid_count=invalid_count,
        budget_pass_rate=float(budget_mask.mean()) if len(budget_mask) else 1.0,
        bounds_pass_rate=float(bounds_mask.mean()) if len(bounds_mask) else 1.0,
        immutable_pass_rate=float(immutable_mask.mean()) if len(immutable_mask) else 1.0,
        relation_pass_rate=float(relation_mask.mean()) if len(relation_mask) else 1.0,
        validity_rate=float(valid_mask.mean()) if len(valid_mask) else 1.0,
    )


def _mutable_columns(features: pd.DataFrame, requested: list[str]) -> list[str]:
    """Return requested mutable columns or all numeric columns when none match."""
    mutable = [col for col in requested if col in features.columns]
    if mutable:
        return mutable
    return features.select_dtypes(include=[np.number]).columns.tolist()


def _target_index(features: pd.DataFrame, labels: pd.Series | None) -> pd.Index:
    """Return the rows eligible for perturbation under the existing label rule."""
    if labels is None:
        return features.index
    return labels[labels.astype(int) == 1].index


def _scale_array(mutable: list[str], scales: Mapping[str, float]) -> np.ndarray:
    """Return feature scales in mutable-column order."""
    return np.asarray([float(scales.get(col, 1.0) or 1.0) for col in mutable], dtype=float)


def _project_nonnegative_inplace(
    features: pd.DataFrame,
    config: ConstrainedAttackConfig,
    lower_bounds: Mapping[str, float] | None = None,
) -> None:
    bounded = set((lower_bounds or {}).keys())
    for col in config.nonnegative_features:
        if col in features.columns and col not in bounded:
            features[col] = features[col].clip(lower=0.0)


def _project_feature_bounds_inplace(
    features: pd.DataFrame,
    lower_bounds: Mapping[str, float] | None,
    upper_bounds: Mapping[str, float] | None,
    columns: list[str],
) -> None:
    lower = lower_bounds or {}
    upper = upper_bounds or {}
    for col in columns:
        if col not in features.columns:
            continue
        low = lower.get(col)
        high = upper.get(col)
        if low is not None or high is not None:
            features[col] = features[col].clip(lower=low, upper=high)


def _scaled_uniform_noise(
    rng: np.random.Generator,
    config: ConstrainedAttackConfig,
    mutable: list[str],
    row_count: int,
    scale: np.ndarray,
) -> np.ndarray:
    normalized = rng.uniform(-config.epsilon, config.epsilon, size=(row_count, len(mutable)))
    return normalized * scale


def _project_budget_inplace(
    clean: pd.DataFrame,
    attacked: pd.DataFrame,
    mutable: list[str],
    target_index: pd.Index,
    config: ConstrainedAttackConfig,
    scale: np.ndarray,
) -> None:
    if not mutable or len(target_index) == 0 or config.epsilon <= 0.0:
        return
    delta = attacked.loc[target_index, mutable].to_numpy(dtype=float) - clean.loc[target_index, mutable].to_numpy(dtype=float)
    normalized = delta / scale
    if config.budget_norm in {"inf", "linf", "l_inf"}:
        norm = np.max(np.abs(normalized), axis=1)
    elif config.budget_norm in {"1", "l1"}:
        norm = np.sum(np.abs(normalized), axis=1)
    elif config.budget_norm in {"2", "l2"}:
        norm = np.linalg.norm(normalized, ord=2, axis=1)
    else:
        raise ValueError(f"Unsupported budget_norm: {config.budget_norm}")
    over_budget = norm > config.epsilon
    if not over_budget.any():
        return
    factors = np.ones_like(norm, dtype=float)
    factors[over_budget] = config.epsilon / np.maximum(norm[over_budget], 1e-12)
    projected = clean.loc[target_index, mutable].to_numpy(dtype=float) + normalized * factors[:, None] * scale
    attacked.loc[target_index, mutable] = projected


def _project_nonnegative(features: pd.DataFrame, config: ConstrainedAttackConfig) -> pd.DataFrame:
    projected = features.copy()
    _project_nonnegative_inplace(projected, config)
    return projected


def _budget_mask(
    clean: pd.DataFrame,
    attacked: pd.DataFrame,
    mutable: list[str],
    target_index: pd.Index,
    config: ConstrainedAttackConfig,
    scale: np.ndarray,
) -> np.ndarray:
    passed = pd.Series(True, index=clean.index)
    if not mutable or len(target_index) == 0 or config.epsilon <= 0.0:
        return passed.to_numpy(dtype=bool)
    delta = attacked.loc[target_index, mutable] - clean.loc[target_index, mutable]
    scaled = delta.to_numpy(dtype=float) / scale
    if config.budget_norm in {"inf", "linf", "l_inf"}:
        norm = np.max(np.abs(scaled), axis=1)
    elif config.budget_norm in {"1", "l1"}:
        norm = np.sum(np.abs(scaled), axis=1)
    elif config.budget_norm in {"2", "l2"}:
        norm = np.linalg.norm(scaled, ord=2, axis=1)
    else:
        raise ValueError(f"Unsupported budget_norm: {config.budget_norm}")
    passed.loc[target_index] = norm <= config.epsilon + 1e-9
    return passed.to_numpy(dtype=bool)


def _bounds_mask(
    attacked: pd.DataFrame,
    config: ConstrainedAttackConfig,
    lower_bounds: Mapping[str, float],
    upper_bounds: Mapping[str, float],
) -> np.ndarray:
    passed = np.ones(len(attacked), dtype=bool)
    checked = False
    for col in attacked.columns:
        low = lower_bounds.get(col)
        high = upper_bounds.get(col)
        if low is not None:
            checked = True
            passed &= (attacked[col] >= float(low) - 1e-9).to_numpy()
        if high is not None:
            checked = True
            passed &= (attacked[col] <= float(high) + 1e-9).to_numpy()
    for col in config.nonnegative_features:
        if col in attacked.columns and col not in lower_bounds:
            checked = True
            passed &= (attacked[col] >= 0.0).to_numpy()
    return passed if checked else np.ones(len(attacked), dtype=bool)


def _immutable_mask(clean: pd.DataFrame, attacked: pd.DataFrame, mutable: list[str]) -> np.ndarray:
    mutable_set = set(mutable)
    immutable = [col for col in clean.columns if col not in mutable_set and col in attacked.columns]
    passed = np.ones(len(clean), dtype=bool)
    for col in immutable:
        if pd.api.types.is_numeric_dtype(clean[col]):
            passed &= np.isclose(clean[col].to_numpy(dtype=float), attacked[col].to_numpy(dtype=float), atol=1e-9)
        else:
            passed &= (clean[col].astype(str) == attacked[col].astype(str)).to_numpy()
    return passed


def _relation_mask(
    attacked: pd.DataFrame,
    relationship_check: Callable[[pd.DataFrame], np.ndarray] | None,
) -> np.ndarray:
    if relationship_check is None:
        return np.ones(len(attacked), dtype=bool)
    return np.asarray(relationship_check(attacked), dtype=bool)


def validity_rate(features: pd.DataFrame, config: ConstrainedAttackConfig) -> float:
    """Check non-negativity constraints for generated adversarial samples."""
    valid = np.ones(len(features), dtype=bool)
    checked = False
    for col in config.nonnegative_features:
        if col in features.columns:
            checked = True
            valid &= (features[col] >= 0.0).to_numpy()
    if not checked:
        return 1.0
    return float(valid.mean())
