#!/usr/bin/env python
"""Create a split manifest for the configured smoke dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.pipeline import load_dataset_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="manifests/splits")
    args = parser.parse_args()
    config = load_yaml(args.config)
    frame = load_dataset_from_config(config.get("dataset", {}))
    indices = pd.Series(frame.index, name="row_id")
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(config.get("test_size", 0.3)),
        random_state=int(config.get("seed", 0)),
        stratify=frame["label"],
    )
    manifest = pd.DataFrame(
        {
            "row_id": list(train_idx) + list(test_idx),
            "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
            "split_id": config.get("split_id", "synthetic_seeded_split"),
        }
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split_manifest.csv"
    manifest.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()

