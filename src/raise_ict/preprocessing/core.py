"""Leakage-safe preprocessing for tabular flow data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DROP_COLUMNS = [
    "id",
    "dataset_id",
    "split",
    "timestamp",
    "src_ip",
    "dst_ip",
    "attack_cat",
    "type",
    "attack_type",
    "group",
]


@dataclass
class FlowPreprocessor:
    """Fit train-only tabular feature preprocessing for flow records."""

    label_column: str = "label"
    drop_columns: list[str] = field(default_factory=lambda: DEFAULT_DROP_COLUMNS.copy())
    categorical_columns: list[str] | None = field(default_factory=list)
    log_columns: list[str] = field(default_factory=list)

    numeric_columns_: list[str] = field(default_factory=list, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    means_: dict[str, float] = field(default_factory=dict, init=False)
    stds_: dict[str, float] = field(default_factory=dict, init=False)
    categories_: dict[str, list[str]] = field(default_factory=dict, init=False)
    feature_lower_bounds_: dict[str, float] = field(default_factory=dict, init=False)
    feature_upper_bounds_: dict[str, float] = field(default_factory=dict, init=False)
    feature_scales_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, frame: pd.DataFrame) -> "FlowPreprocessor":
        """Learn numeric statistics and categorical levels from training data."""
        features = self._raw_features(frame)
        if self.categorical_columns is None:
            candidate_categories = features.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            candidate_categories = self.categorical_columns
        self.categorical_columns = [col for col in candidate_categories if col in features.columns]
        self.numeric_columns_ = [col for col in features.columns if col not in self.categorical_columns]
        self.categories_ = {
            col: sorted(features[col].dropna().astype(str).unique().tolist())
            for col in self.categorical_columns
        }

        numeric = self._prepare_numeric(features[self.numeric_columns_].copy(), fit=True)
        self.medians_ = numeric.median(numeric_only=True).fillna(0.0).to_dict()
        numeric = numeric.fillna(self.medians_)
        self.means_ = numeric.mean(numeric_only=True).fillna(0.0).to_dict()
        self.stds_ = numeric.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0).to_dict()

        transformed = self.transform(frame)
        self.feature_columns_ = transformed.columns.tolist()
        self.feature_lower_bounds_ = transformed.min(numeric_only=True).fillna(0.0).to_dict()
        self.feature_upper_bounds_ = transformed.max(numeric_only=True).fillna(0.0).to_dict()
        self.feature_scales_ = (
            transformed.std(numeric_only=True, ddof=0)
            .replace(0.0, 1.0)
            .fillna(1.0)
            .to_dict()
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature schema to a new frame."""
        features = self._raw_features(frame)
        numeric = self._prepare_numeric(features.reindex(columns=self.numeric_columns_).copy(), fit=False)
        numeric = numeric.fillna(self.medians_).fillna(0.0)
        for col in self.numeric_columns_:
            numeric[col] = (numeric[col] - self.means_[col]) / self.stds_[col]

        categorical_parts = []
        for col, values in self.categories_.items():
            series = features[col].astype(str) if col in features else pd.Series("", index=frame.index, dtype="object")
            part = pd.DataFrame(
                {f"{col}={value}": (series == value).astype(float) for value in values},
                index=frame.index,
            )
            categorical_parts.append(part)

        out = pd.concat([numeric.astype(float), *categorical_parts], axis=1)
        if self.feature_columns_:
            out = out.reindex(columns=self.feature_columns_, fill_value=0.0)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing on a frame and return its transformed features."""
        return self.fit(frame).transform(frame)

    def labels(self, frame: pd.DataFrame) -> pd.Series:
        """Return integer labels from the configured label column."""
        return frame[self.label_column].astype(int)

    def _raw_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        drop = [self.label_column, *self.drop_columns]
        return frame.drop(columns=[col for col in drop if col in frame.columns], errors="ignore").copy()

    def _prepare_numeric(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        for col in self.log_columns:
            if col in frame.columns:
                frame[col] = np.log1p(frame[col].clip(lower=0.0))
        return frame

    def state_dict(self) -> dict[str, object]:
        """Return deterministic fitted-state metadata for reproducibility manifests."""
        return {
            "label_column": self.label_column,
            "drop_columns": self.drop_columns,
            "categorical_columns": self.categorical_columns,
            "log_columns": self.log_columns,
            "numeric_columns": self.numeric_columns_,
            "feature_columns": self.feature_columns_,
            "medians": self.medians_,
            "means": self.means_,
            "stds": self.stds_,
            "categories": self.categories_,
            "feature_lower_bounds": self.feature_lower_bounds_,
            "feature_upper_bounds": self.feature_upper_bounds_,
            "feature_scales": self.feature_scales_,
        }

    def state_sha256(self) -> str:
        """Hash the fitted preprocessing state for result and split manifests."""
        payload = json.dumps(self.state_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_bounds(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Compute min/max bounds for selected feature columns."""
    return {col: (float(frame[col].min()), float(frame[col].max())) for col in columns}
