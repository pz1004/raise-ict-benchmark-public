#!/usr/bin/env python
"""Merge benchmark manifest artifacts without changing their schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def merge_csv(inputs: list[Path], out: Path) -> int:
    """Merge CSV inputs row-wise and return the emitted row count."""
    frames = [pd.read_csv(path) for path in inputs]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    return len(merged)


def merge_json_lists(inputs: list[Path], out: Path) -> int:
    """Merge top-level JSON lists and return the emitted record count."""
    records: list[object] = []
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a top-level JSON list")
        records.extend(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or JSON-list files to merge")
    parser.add_argument("--out", required=True, help="Merged output path")
    args = parser.parse_args()

    inputs = [Path(path) for path in args.inputs]
    out = Path(args.out)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        count = merge_csv(inputs, out)
    elif suffix == ".json":
        count = merge_json_lists(inputs, out)
    else:
        raise ValueError("--out must end in .csv or .json")
    print(f"{out} rows={count}")


if __name__ == "__main__":
    main()
