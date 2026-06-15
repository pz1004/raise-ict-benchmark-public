"""Small baseline model factory."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_model(
    model_id: str,
    seed: int = 0,
) -> DummyClassifier | LogisticRegression | RandomForestClassifier | ExtraTreesClassifier | HistGradientBoostingClassifier | MLPClassifier:
    """Build one of the supported baseline estimators with stable defaults."""
    if model_id == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if model_id == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(n_estimators=40, class_weight="balanced", random_state=seed, n_jobs=1)
    if model_id == "extra_trees":
        return ExtraTreesClassifier(n_estimators=40, class_weight="balanced", random_state=seed, n_jobs=1)
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=True,
            class_weight="balanced",
            random_state=seed,
        )
    if model_id == "mlp_sklearn":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            max_iter=60,
            batch_size=512,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            alpha=1e-4,
            learning_rate_init=1e-3,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model_id: {model_id}")
