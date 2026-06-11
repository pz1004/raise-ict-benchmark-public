#!/usr/bin/env python
"""Run the configured constrained-attack smoke evaluation."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.pipeline import train_and_evaluate, write_result  # noqa: E402


def load_attack_experiment(config_path: str, experiment_config_path: str) -> dict[str, object]:
    """Load a full experiment config or overlay an attack-only config."""
    override = load_yaml(config_path)
    if {"dataset", "model", "split_id"} & set(override):
        override["config_path"] = config_path
        return override
    config = load_yaml(experiment_config_path)
    config["attack"] = override
    config["config_path"] = f"{experiment_config_path};{config_path}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-config", default="configs/experiments/tier_s.yaml")
    parser.add_argument("--out-dir", default="results/raw")
    args = parser.parse_args()
    result = train_and_evaluate(load_attack_experiment(args.config, args.experiment_config))
    print(write_result(result, args.out_dir))


if __name__ == "__main__":
    main()
