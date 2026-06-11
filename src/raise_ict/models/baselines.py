"""Small baseline model factory."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_model(
    model_id: str,
    seed: int = 0,
) -> DummyClassifier | LogisticRegression | RandomForestClassifier | ExtraTreesClassifier:
    """Build one of the supported baseline estimators with stable defaults."""
    if model_id == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if model_id == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(n_estimators=40, class_weight="balanced", random_state=seed, n_jobs=1)
    if model_id == "extra_trees":
        return ExtraTreesClassifier(n_estimators=40, class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported model_id: {model_id}")
