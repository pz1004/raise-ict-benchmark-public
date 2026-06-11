#!/usr/bin/env python
"""Train and evaluate the configured smoke benchmark."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.pipeline import train_and_evaluate, write_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="results/raw")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config["config_path"] = args.config
    result = train_and_evaluate(config)
    print(write_result(result, args.out_dir))


if __name__ == "__main__":
    main()
