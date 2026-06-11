#!/usr/bin/env python
"""Aggregate raw benchmark result CSVs into paper artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from raise_ict.reporting import render_dataset_suite, render_main_results, render_placeholder_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=["results/raw"])
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--figures", default="results/figures")
    args = parser.parse_args()
    files: list[Path] = []
    for result_dir in args.results:
        files.extend(sorted(Path(result_dir).glob("*.csv")))
    frames = [pd.read_csv(path) for path in files]
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dataset_path = render_dataset_suite(args.out)
    main_path = render_main_results(results, args.out)
    figure_paths = render_placeholder_figures(results, args.figures)
    print(dataset_path)
    print(main_path)
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
