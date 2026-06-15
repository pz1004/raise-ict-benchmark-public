#!/usr/bin/env python
"""Run a configured RAISE-ICT experiment grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from raise_ict import __version__  # noqa: E402
from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.datasets.real import (  # noqa: E402
    DEFAULT_CICIDS2017_FILES,
    DEFAULT_CSE_CIC_IDS2018_FILES,
)
from raise_ict.pipeline import (  # noqa: E402
    evaluate_threat,
    load_dataset_from_config,
    make_split,
    train_context,
    write_result,
)
from raise_ict.preprocessing import FlowPreprocessor  # noqa: E402


TIMING_FIELDS = [
    "event_id",
    "stage",
    "dataset",
    "split_id",
    "seed",
    "model_id",
    "threat_id",
    "start_iso",
    "end_iso",
    "elapsed_s",
    "rows",
    "output_path",
    "detail",
]


def _schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _json_hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _record_timing(
    events: list[dict[str, object]],
    stage: str,
    start_wall: float,
    end_wall: float,
    **metadata: object,
) -> None:
    events.append(
        {
            "event_id": len(events) + 1,
            "stage": stage,
            "dataset": metadata.get("dataset", ""),
            "split_id": metadata.get("split_id", ""),
            "seed": metadata.get("seed", ""),
            "model_id": metadata.get("model_id", ""),
            "threat_id": metadata.get("threat_id", ""),
            "start_iso": _iso_from_timestamp(start_wall),
            "end_iso": _iso_from_timestamp(end_wall),
            "elapsed_s": max(0.0, end_wall - start_wall),
            "rows": metadata.get("rows", ""),
            "output_path": metadata.get("output_path", ""),
            "detail": metadata.get("detail", ""),
        }
    )


def _record_elapsed_timing(
    events: list[dict[str, object]],
    stage: str,
    elapsed_s: float,
    **metadata: object,
) -> None:
    end_wall = time.time()
    start_wall = end_wall - max(0.0, elapsed_s)
    _record_timing(events, stage, start_wall, end_wall, **metadata)


def _write_timing_artifacts(events: list[dict[str, object]], timing_events_path: str) -> None:
    if not timing_events_path:
        return
    events_path = Path(timing_events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    summary_path = events_path.with_name(events_path.stem.replace("_events", "") + "_summary.csv")
    frame = pd.DataFrame(events)
    if frame.empty:
        summary = pd.DataFrame(columns=["stage", "event_count", "elapsed_s_total", "elapsed_s_mean", "elapsed_s_max"])
    else:
        grouped = frame.groupby("stage", dropna=False)["elapsed_s"]
        summary = grouped.agg(
            event_count="count",
            elapsed_s_total="sum",
            elapsed_s_mean="mean",
            elapsed_s_max="max",
        ).reset_index()
    summary.to_csv(summary_path, index=False)


def _manifest_preprocessor(prep_cfg: Mapping[str, Any]) -> FlowPreprocessor:
    return FlowPreprocessor(
        categorical_columns=prep_cfg.get("categorical_columns", []),
        drop_columns=prep_cfg.get("drop_columns", None) or FlowPreprocessor().drop_columns,
        log_columns=prep_cfg.get("log_columns", []),
    )


def _split_group_field(dataset_config: Mapping[str, Any], frame: pd.DataFrame) -> str:
    split_strategy = dataset_config.get("split_strategy")
    group_field = "split" if "split" in frame.columns else "stratified_random"
    if split_strategy == "attack_type_holdout":
        return "attack_type"
    if split_strategy == "date_holdout":
        return "date"
    if split_strategy == "day_holdout":
        return "day"
    if split_strategy == "scenario_holdout":
        return "scenario"
    return group_field


def _raw_files(dataset_config: Mapping[str, Any]) -> list[str]:
    dataset_id = dataset_config.get("dataset_id", "")
    if dataset_id == "UNSW-NB15":
        root = Path(dataset_config.get("data_root", "data/raw/unsw_nb15/temporal"))
        return [
            str(root / dataset_config.get("train_file", "train-00000-of-00001.parquet")),
            str(root / dataset_config.get("test_file", "test-00000-of-00001.parquet")),
        ]
    if dataset_id in {"CICIDS2017", "CSE-CIC-IDS2018"}:
        root = Path(dataset_config.get("data_root", ""))
        default_files = DEFAULT_CICIDS2017_FILES if dataset_id == "CICIDS2017" else DEFAULT_CSE_CIC_IDS2018_FILES
        return [str(root / filename) for filename in dataset_config.get("files", default_files)]
    if "path" in dataset_config:
        return [str(dataset_config["path"])]
    return []


def _manifest_and_schema_records(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    seed = int(config.get("seed", 0))
    train, test = make_split(frame, seed=seed, test_size=float(config.get("test_size", 0.3)))
    prep_cfg = config.get("preprocessing", {})
    preprocessor = _manifest_preprocessor(prep_cfg)
    feature_columns = preprocessor.fit_transform(train).columns.tolist()
    preprocessing_state_sha256 = preprocessor.state_sha256()
    dataset_config = config.get("dataset", {})
    dataset_id = dataset_config.get("dataset_id", "unknown")
    split_strategy = dataset_config.get("split_strategy")
    group_field = _split_group_field(dataset_config, frame)
    raw_files = _raw_files(dataset_config)
    manifest_row = {
        "dataset": dataset_id,
        "split_id": config.get("split_id", "unknown_split"),
        "seed": seed,
        "raw_files": ";".join(raw_files),
        "raw_rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive": int((train["label"] != 0).sum()),
        "test_positive": int((test["label"] != 0).sum()),
        "group_field": group_field,
        "split_strategy": split_strategy or group_field,
        "feature_count": len(feature_columns),
        "feature_schema_sha256": _schema_hash(feature_columns),
        "dataset_config_sha256": _json_hash(dataset_config),
        "preprocessing_state_sha256": preprocessing_state_sha256,
        "config_path": config.get("config_path", ""),
        "software_version": __version__,
    }
    schema_record = {
        "dataset": config.get("dataset", {}).get("dataset_id", "unknown"),
        "split_id": config.get("split_id", "unknown_split"),
        "seed": seed,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "preprocessing_state_sha256": preprocessing_state_sha256,
        "config_path": config.get("config_path", ""),
        "software_version": __version__,
    }
    return manifest_row, schema_record


def _configured_datasets(grid: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "datasets" in grid:
        return list(grid.get("datasets", []))
    return [grid["dataset"]] if "dataset" in grid else []


def _configured_models(grid: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "models" in grid:
        return list(grid.get("models", []))
    return [grid["model"]] if "model" in grid else []


def _configured_threats(grid: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "threats" in grid:
        return list(grid.get("threats", []))
    return [grid["attack"]] if "attack" in grid else []


def _configured_seeds(grid: Mapping[str, Any]) -> list[int]:
    if "seeds" in grid:
        return [int(seed) for seed in grid.get("seeds", [])]
    return [int(grid.get("seed", 0))]


def _write_profile_manifest(
    path: str,
    grid: Mapping[str, Any],
    out_dir: Path,
    result_rows: list[dict[str, object]],
) -> None:
    if not path:
        return
    frame = pd.DataFrame(result_rows)
    profile = {
        "hardware_id": grid.get("hardware", {}).get("hardware_id", "unknown"),
        "thread_count": int(grid.get("hardware", {}).get("thread_count", grid.get("hardware", {}).get("threads", 1))),
        "batch_size": int(grid.get("hardware", {}).get("batch_size", 1)),
        "runtime": grid.get("hardware", {}).get("runtime", "python_sklearn_cpu"),
        "measurement_mode": grid.get("hardware", {}).get("measurement_mode", "proxy"),
        "energy_source": grid.get("hardware", {}).get("energy_source", "proxy"),
        "p95_latency_ms_min": float(frame["p95_latency_ms"].min()) if not frame.empty else 0.0,
        "p95_latency_ms_median": float(frame["p95_latency_ms"].median()) if not frame.empty else 0.0,
        "p95_latency_ms_max": float(frame["p95_latency_ms"].max()) if not frame.empty else 0.0,
        "energy_per_flow_j": float(frame["energy_per_flow_j"].median()) if not frame.empty else 0.0,
        "energy_per_flow_j_min": float(frame["energy_per_flow_j"].min()) if not frame.empty else 0.0,
        "energy_per_flow_j_max": float(frame["energy_per_flow_j"].max()) if not frame.empty else 0.0,
    }
    manifest = {
        "schema_version": 1,
        "result_dir": str(out_dir),
        "result_count": len(result_rows),
        "hardware": grid.get("hardware", {}),
        "profile": profile,
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="results/raw/tier_p")
    parser.add_argument("--split-manifest", default="manifests/splits/tier_p_split_manifest.csv")
    parser.add_argument("--feature-schema", default="manifests/feature_schemas/tier_p_feature_schema.json")
    parser.add_argument("--hardware-config", default="")
    parser.add_argument("--profile-manifest", default="")
    parser.add_argument("--timing-events", default="")
    args = parser.parse_args()

    grid = load_yaml(args.config)
    grid["config_path"] = args.config
    if args.hardware_config:
        grid["hardware"] = load_yaml(args.hardware_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = []
    schema_rows = []
    result_paths = []
    result_rows = []
    timing_events: list[dict[str, object]] = []

    for dataset_cfg in _configured_datasets(grid):
        for seed in _configured_seeds(grid):
            dataset_with_seed = dict(dataset_cfg)
            dataset_with_seed["seed"] = seed
            dataset_id = str(dataset_cfg.get("dataset_id", "unknown"))
            split_id = str(dataset_cfg.get("split_id", grid.get("split_id", "realdata_split")))
            start_wall = time.time()
            frame = load_dataset_from_config(dataset_with_seed)
            end_wall = time.time()
            _record_timing(
                timing_events,
                "dataset_load",
                start_wall,
                end_wall,
                dataset=dataset_id,
                split_id=split_id,
                seed=seed,
                rows=len(frame),
            )
            base_config = {
                "seed": seed,
                "profile_repeats": grid.get("profile_repeats", 2),
                "test_size": grid.get("test_size", 0.3),
                "split_id": split_id,
                "dataset": dataset_with_seed,
                "hardware": grid.get("hardware", {"hardware_id": "cpu_proxy"}),
                "preprocessing": grid.get("preprocessing", {}),
                "config_path": grid.get("config_path", args.config),
                "scoring": grid.get("scoring", {}),
            }
            start_wall = time.time()
            split_row, schema_row = _manifest_and_schema_records(base_config, frame)
            end_wall = time.time()
            _record_timing(
                timing_events,
                "preprocess_manifest",
                start_wall,
                end_wall,
                dataset=dataset_id,
                split_id=split_id,
                seed=seed,
                rows=len(frame),
            )
            split_rows.append(split_row)
            schema_rows.append(schema_row)
            for model_cfg in _configured_models(grid):
                model_config = {**base_config, "model": dict(model_cfg)}
                model_id = str(model_cfg.get("model_id", "unknown_model"))

                def timing_callback(stage: str, elapsed_s: float, model_id: str = model_id) -> None:
                    _record_elapsed_timing(
                        timing_events,
                        stage,
                        elapsed_s,
                        dataset=dataset_id,
                        split_id=split_id,
                        seed=seed,
                        model_id=model_id,
                        rows=len(frame),
                    )

                context = train_context(model_config, frame, timing_callback=timing_callback)
                for threat_cfg in _configured_threats(grid):
                    attack_config = dict(threat_cfg)
                    attack_config["seed"] = seed
                    config = {**model_config, "attack": attack_config}
                    threat_id = str(attack_config.get("threat_id", "unknown_threat"))
                    start_wall = time.time()
                    result = evaluate_threat(config, context)
                    end_wall = time.time()
                    _record_timing(
                        timing_events,
                        "threat_evaluation",
                        start_wall,
                        end_wall,
                        dataset=dataset_id,
                        split_id=split_id,
                        seed=seed,
                        model_id=model_id,
                        threat_id=threat_id,
                        rows=context.test_rows,
                    )
                    result_rows.append(result.to_row())
                    start_wall = time.time()
                    result_path = write_result(result, out_dir)
                    end_wall = time.time()
                    result_paths.append(str(result_path))
                    _record_timing(
                        timing_events,
                        "result_writing",
                        start_wall,
                        end_wall,
                        dataset=dataset_id,
                        split_id=split_id,
                        seed=seed,
                        model_id=model_id,
                        threat_id=threat_id,
                        rows=1,
                        output_path=str(result_path),
                    )

    split_manifest = Path(args.split_manifest)
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).drop_duplicates().to_csv(split_manifest, index=False)

    feature_schema = Path(args.feature_schema)
    feature_schema.parent.mkdir(parents=True, exist_ok=True)
    feature_schema.write_text(json.dumps(schema_rows, indent=2), encoding="utf-8")
    _write_profile_manifest(args.profile_manifest, grid, out_dir, result_rows)
    _write_timing_artifacts(timing_events, args.timing_events)

    print(
        json.dumps(
            {
                "results": result_paths,
                "split_manifest": str(split_manifest),
                "feature_schema": str(feature_schema),
                "timing_events": args.timing_events,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
