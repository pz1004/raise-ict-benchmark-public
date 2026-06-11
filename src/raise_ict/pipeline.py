"""End-to-end smoke and real-data pipeline for the RAISE-ICT harness scaffold."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .attacks import (
    AttackValidityReport,
    ConstrainedAttackConfig,
    evaluate_constrained_perturbations,
)
from .datasets import (
    SyntheticDatasetSpec,
    load_cicids2017,
    load_cse_cic_ids2018,
    load_synthetic_frame,
    load_ton_iot_network,
    load_unsw_nb15,
)
from .metrics import classification_summary, raise_score, service_cost
from .models import build_model
from .preprocessing import FlowPreprocessor
from .profiling import profile_predict
from .schema import BenchmarkResult


@dataclass
class TrainedRunContext:
    """Reusable trained model state for evaluating multiple threat rows."""

    dataset: str
    split_id: str
    seed: int
    model_id: str
    hardware_id: str
    model: Any
    x_test: pd.DataFrame
    y_test: pd.Series
    clean: dict[str, float]
    clean_pred: np.ndarray
    profile: dict[str, float | int | str]
    service_cost_value: float
    preprocessor: FlowPreprocessor
    preprocessing_state_sha256: str
    train_rows: int
    test_rows: int


def load_dataset_from_config(dataset_config: Mapping[str, Any]) -> pd.DataFrame:
    """Load a supported real or synthetic dataset from a config mapping."""
    dataset_id = dataset_config.get("dataset_id", "synthetic_raise_ict")
    if dataset_id == "CICIDS2017":
        return load_cicids2017(dataset_config)
    if dataset_id == "CSE-CIC-IDS2018":
        return load_cse_cic_ids2018(dataset_config)
    if dataset_id == "UNSW-NB15":
        return load_unsw_nb15(dataset_config)
    if dataset_id == "TON_IoT":
        return load_ton_iot_network(dataset_config)
    if dataset_id != "synthetic_raise_ict":
        raise ValueError(f"Unsupported dataset_id: {dataset_id}")
    spec = SyntheticDatasetSpec(
        dataset_id=dataset_id,
        n_samples=int(dataset_config.get("n_samples", 240)),
        seed=int(dataset_config.get("seed", 7)),
    )
    return load_synthetic_frame(spec)


def make_split(frame: pd.DataFrame, seed: int, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test frames, honoring explicit split labels when present."""
    if "split" in frame.columns:
        split = frame["split"].astype(str).str.lower()
        if {"train", "test"}.issubset(set(split.unique())):
            train = frame[split == "train"]
            test = frame[split == "test"]
            return train.reset_index(drop=True), test.reset_index(drop=True)
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_and_evaluate(config: Mapping[str, Any]) -> BenchmarkResult:
    """Run dataset loading, training, attack evaluation, and result packaging."""
    seed = int(config.get("seed", 0))
    frame = load_dataset_from_config(config.get("dataset", {}))
    return train_and_evaluate_frame(config, frame)


def train_and_evaluate_frame(config: Mapping[str, Any], frame: pd.DataFrame) -> BenchmarkResult:
    """Run training and one configured threat against an already loaded frame."""
    context = train_context(config, frame)
    return evaluate_threat(config, context)


def train_context(config: Mapping[str, Any], frame: pd.DataFrame) -> TrainedRunContext:
    """Train a model once and return reusable clean/profile evaluation state."""
    seed = int(config.get("seed", 0))
    train, test = make_split(frame, seed=seed, test_size=float(config.get("test_size", 0.3)))

    prep_cfg = config.get("preprocessing", {})
    prep_kwargs = {
        "categorical_columns": prep_cfg.get("categorical_columns", ["protocol", "service"]),
        "log_columns": prep_cfg.get("log_columns", ["fwd_bytes", "bwd_bytes"]),
    }
    if "drop_columns" in prep_cfg:
        prep_kwargs["drop_columns"] = prep_cfg["drop_columns"]
    preprocessor = FlowPreprocessor(**prep_kwargs)
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    y_train = preprocessor.labels(train)
    y_test = preprocessor.labels(test)

    model_id = config.get("model", {}).get("model_id", "logistic_regression")
    model = build_model(model_id, seed=seed)
    model.fit(x_train, y_train)

    clean_pred = model.predict(x_test)
    clean = classification_summary(y_test, clean_pred)
    svc_cost = service_cost(y_test, clean_pred)
    profile = profile_predict(
        model,
        x_test,
        repeats=int(config.get("profile_repeats", 2)),
        hardware=config.get("hardware", {}),
    )
    return TrainedRunContext(
        dataset=config.get("dataset", {}).get("dataset_id", "synthetic_raise_ict"),
        split_id=config.get("split_id", "synthetic_seeded_split"),
        seed=seed,
        model_id=model_id,
        hardware_id=config.get("hardware", {}).get("hardware_id", "cpu_proxy"),
        model=model,
        x_test=x_test,
        y_test=y_test,
        clean=clean,
        clean_pred=clean_pred,
        profile=profile,
        service_cost_value=svc_cost,
        preprocessor=preprocessor,
        preprocessing_state_sha256=preprocessor.state_sha256(),
        train_rows=len(train),
        test_rows=len(test),
    )


def _benign_score_fn(model: Any, benign_label: int = 0) -> Callable[[pd.DataFrame], np.ndarray]:
    def score(features: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if benign_label in classes:
                return model.predict_proba(features)[:, classes.index(benign_label)]
        return (model.predict(features) == benign_label).astype(float)

    return score


def _threat_type(attack_cfg: Mapping[str, Any]) -> str:
    explicit = attack_cfg.get("threat_type")
    if explicit:
        return str(explicit)
    threat_id = str(attack_cfg.get("threat_id", ""))
    if threat_id.startswith("a0"):
        return "clean"
    if threat_id.startswith("a4"):
        return "shift"
    return "evasion"


def _all_valid_report(size: int) -> AttackValidityReport:
    valid = np.ones(size, dtype=bool)
    return AttackValidityReport(
        valid_mask=valid,
        valid_count=size,
        invalid_count=0,
        budget_pass_rate=1.0,
        bounds_pass_rate=1.0,
        immutable_pass_rate=1.0,
        relation_pass_rate=1.0,
        validity_rate=1.0,
    )


def _classification_summary_for_valid(
    y_true: pd.Series,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    if not valid_mask.any():
        return {"clean_macro_f1": 0.0, "clean_bal_acc": 0.0, "utility": 0.0}
    return classification_summary(y_true.loc[valid_mask], y_pred[valid_mask])


def _asr_for_valid_malicious(y_true: pd.Series, y_pred: np.ndarray, valid_mask: np.ndarray) -> float:
    malicious_valid = valid_mask & (y_true.to_numpy() != 0)
    denominator = int(malicious_valid.sum())
    if denominator == 0:
        return 0.0
    return float(((y_pred == 0) & malicious_valid).sum() / denominator)


def _score_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    allowed = {
        "latency_budget_ms",
        "latency_cap_ms",
        "memory_budget_mb",
        "memory_cap_mb",
        "energy_budget_j",
        "energy_cap_j",
        "alpha",
        "beta",
        "latency_weight",
        "energy_weight",
        "memory_weight",
        "service_weight",
    }
    return {
        key: float(value)
        for key, value in config.get("scoring", {}).items()
        if key in allowed
    }


def evaluate_threat(config: Mapping[str, Any], context: TrainedRunContext) -> BenchmarkResult:
    """Evaluate one configured threat using a previously trained context."""
    seed = int(config.get("seed", context.seed))
    attack_cfg = config.get("attack", {})
    attack = ConstrainedAttackConfig(
        threat_id=attack_cfg.get("threat_id", "t1_constrained_feature"),
        epsilon=float(attack_cfg.get("epsilon", 0.15)),
        mutable_features=attack_cfg.get("mutable_features", []),
        nonnegative_features=attack_cfg.get("nonnegative_features", []),
        seed=seed,
        strategy=attack_cfg.get("strategy", "random"),
        n_candidates=int(attack_cfg.get("n_candidates", 1)),
        budget_norm=str(attack_cfg.get("budget_norm", "inf")),
    )
    threat_type = _threat_type(attack_cfg)
    if threat_type in {"clean", "shift"}:
        x_adv = context.x_test
        report = _all_valid_report(len(context.x_test))
    else:
        score_fn = _benign_score_fn(context.model) if attack.strategy == "score_search" else None
        evaluation = evaluate_constrained_perturbations(
            context.x_test,
            attack,
            labels=context.y_test,
            score_fn=score_fn,
            lower_bounds=context.preprocessor.feature_lower_bounds_,
            upper_bounds=context.preprocessor.feature_upper_bounds_,
            scales=context.preprocessor.feature_scales_,
        )
        x_adv = evaluation.x_adv
        report = evaluation.report
    adv_pred = context.model.predict(x_adv)
    robust = _classification_summary_for_valid(context.y_test, adv_pred, report.valid_mask)
    is_evasion_attack = bool(attack.mutable_features) and attack.epsilon > 0.0
    asr = _asr_for_valid_malicious(context.y_test, adv_pred, report.valid_mask) if is_evasion_attack else 0.0
    shift_drop = context.clean["utility"] - robust["utility"] if threat_type == "shift" else 0.0
    score = raise_score(
        clean_utility=context.clean["utility"],
        robust_utility=robust["utility"],
        p95_latency_ms=float(context.profile["p95_latency_ms"]),
        peak_mem_mb=float(context.profile["peak_mem_mb"]),
        service_cost_value=context.service_cost_value,
        energy_per_flow_j=float(context.profile["energy_per_flow_j"]),
        **_score_kwargs(config),
    )
    return BenchmarkResult(
        dataset=context.dataset,
        split_id=context.split_id,
        seed=seed,
        model_id=context.model_id,
        threat_id=attack.threat_id,
        hardware_id=context.hardware_id,
        clean_macro_f1=context.clean["clean_macro_f1"],
        clean_bal_acc=context.clean["clean_bal_acc"],
        robust_utility=robust["utility"],
        asr=asr,
        validity_rate=report.validity_rate,
        p95_latency_ms=float(context.profile["p95_latency_ms"]),
        throughput_fps=float(context.profile["throughput_fps"]),
        peak_mem_mb=float(context.profile["peak_mem_mb"]),
        energy_per_flow_j=float(context.profile["energy_per_flow_j"]),
        service_cost=context.service_cost_value,
        raise_score=score,
        valid_count=report.valid_count,
        invalid_count=report.invalid_count,
        budget_pass_rate=report.budget_pass_rate,
        bounds_pass_rate=report.bounds_pass_rate,
        immutable_pass_rate=report.immutable_pass_rate,
        relation_pass_rate=report.relation_pass_rate,
        thread_count=int(context.profile["thread_count"]),
        batch_size=int(context.profile["batch_size"]),
        runtime=str(context.profile["runtime"]),
        measurement_mode=str(context.profile["measurement_mode"]),
        energy_source=str(context.profile["energy_source"]),
        config_path=str(config.get("config_path", "")),
        preprocessing_state_sha256=context.preprocessing_state_sha256,
        shift_group_field=str(
            attack_cfg.get("shift_group_field", config.get("dataset", {}).get("split_strategy", ""))
        )
        if threat_type == "shift"
        else "",
        source_split=str(attack_cfg.get("source_split", "train")) if threat_type == "shift" else "",
        target_split=str(attack_cfg.get("target_split", "test")) if threat_type == "shift" else "",
        shift_utility_drop=shift_drop,
    )


def write_result(result: BenchmarkResult, out_dir: str | Path) -> Path:
    """Write a single standard-schema result CSV and return its path."""
    safe = "_".join(
        str(part).replace("/", "_").replace(" ", "_")
        for part in [result.dataset, result.split_id, result.model_id, result.threat_id, result.seed]
    )
    out_path = Path(out_dir) / f"{safe}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.to_row()]).to_csv(out_path, index=False)
    return out_path
