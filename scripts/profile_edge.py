#!/usr/bin/env python
"""Run CPU-proxy profiling for the configured smoke benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.pipeline import train_and_evaluate, write_result  # noqa: E402


def load_profile_experiment(config_path: str, experiment_config_path: str) -> dict[str, object]:
    """Load a full experiment config or overlay a hardware-only config."""
    override = load_yaml(config_path)
    if {"dataset", "model", "split_id"} & set(override):
        override["config_path"] = config_path
        return override
    config = load_yaml(experiment_config_path)
    config["hardware"] = override
    config["config_path"] = f"{experiment_config_path};{config_path}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-config", default="configs/experiments/tier_s.yaml")
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--profile-manifest", default="")
    args = parser.parse_args()
    config = load_profile_experiment(args.config, args.experiment_config)
    result = train_and_evaluate(config)
    result_path = write_result(result, args.out_dir)
    print(result_path)
    if args.profile_manifest:
        manifest_path = Path(args.profile_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "result_path": str(result_path),
            "hardware": config.get("hardware", {}),
            "profile": {
                "hardware_id": result.hardware_id,
                "thread_count": result.thread_count,
                "batch_size": result.batch_size,
                "runtime": result.runtime,
                "measurement_mode": result.measurement_mode,
                "energy_source": result.energy_source,
                "p95_latency_ms": result.p95_latency_ms,
                "throughput_fps": result.throughput_fps,
                "peak_mem_mb": result.peak_mem_mb,
                "energy_per_flow_j": result.energy_per_flow_j,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest_path)


if __name__ == "__main__":
    main()
