"""Generate paper-table and figure artifacts from benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from raise_ict.stats import normal_mean_ci


DATASET_SUITE_ROWS = [
    {
        "dataset": "CICIDS2017",
        "domain": "enterprise",
        "feature_source": "CICFlowMeter CSV",
        "task": "IDS",
        "split_policy": "day/scenario-aware",
        "role": "source-domain benchmark",
    },
    {
        "dataset": "CSE-CIC-IDS2018",
        "domain": "enterprise/cloud",
        "feature_source": "CICFlowMeter-V3 CSV",
        "task": "IDS",
        "split_policy": "day/scenario-aware",
        "role": "scale stress test",
    },
    {
        "dataset": "UNSW-NB15",
        "domain": "cyber range",
        "feature_source": "Argus/Bro/CSV",
        "task": "IDS",
        "split_policy": "official and group-aware",
        "role": "fixed-split comparison",
    },
    {
        "dataset": "TON_IoT",
        "domain": "IoT/IIoT",
        "feature_source": "network/telemetry/OS traces",
        "task": "IDS",
        "split_policy": "source/layer-aware",
        "role": "edge-fog-cloud benchmark",
    },
]


def render_dataset_suite(out_dir: str | Path) -> Path:
    """Write the static dataset-suite table used by the paper scaffold."""
    out_path = Path(out_dir) / "table_dataset_suite.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DATASET_SUITE_ROWS).to_csv(out_path, index=False)
    return out_path


METRIC_COLUMNS = [
    "clean_macro_f1",
    "clean_bal_acc",
    "robust_utility",
    "asr",
    "validity_rate",
    "valid_count",
    "invalid_count",
    "budget_pass_rate",
    "bounds_pass_rate",
    "immutable_pass_rate",
    "relation_pass_rate",
    "p95_latency_ms",
    "throughput_fps",
    "peak_mem_mb",
    "energy_per_flow_j",
    "service_cost",
    "raise_score",
    "thread_count",
    "batch_size",
    "shift_utility_drop",
]


def _summary_stats(values: pd.Series) -> dict[str, float]:
    return normal_mean_ci(values)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw benchmark rows by dataset, model, threat, and hardware."""
    if results.empty:
        return pd.DataFrame()
    groups = ["dataset", "split_id", "model_id", "threat_id", "hardware_id"]
    rows = []
    for keys, part in results.groupby(groups, dropna=False):
        row = dict(zip(groups, keys, strict=True))
        row["n_runs"] = int(part["seed"].nunique()) if "seed" in part.columns else int(len(part))
        for metric in METRIC_COLUMNS:
            if metric not in part.columns:
                continue
            stats = _summary_stats(part[metric])
            row[metric] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def render_main_results(results: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write raw and summarized result tables to the requested directory."""
    out_path = Path(out_dir) / "table_main_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summarize_results(results).to_csv(out_path, index=False)
    results.to_csv(Path(out_dir) / "table_raw_results.csv", index=False)
    return out_path


def render_placeholder_figures(results: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Render the pipeline and Pareto figures expected by the paper scaffold."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = out / "figure_pipeline.pdf"
    pareto = out / "figure_pareto.pdf"

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    stages = ["datasets", "splits", "models", "attacks", "profile", "tables"]
    ax.plot(range(len(stages)), [1] * len(stages), marker="o")
    ax.set_xticks(range(len(stages)), stages, rotation=20)
    ax.set_yticks([])
    ax.set_title("RAISE-ICT Benchmark Pipeline")
    fig.tight_layout()
    fig.savefig(pipeline)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    summary = summarize_results(results)
    if {"p95_latency_ms", "robust_utility", "model_id"}.issubset(summary.columns) and not summary.empty:
        ax.scatter(summary["p95_latency_ms"], summary["robust_utility"])
        for _, row in summary.iterrows():
            label = f"{row['dataset']}:{row['model_id']}:{row['threat_id']}"
            ax.annotate(label, (row["p95_latency_ms"], row["robust_utility"]), fontsize=5)
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("robust utility")
    ax.set_title("Robust-Efficiency Pareto View")
    fig.tight_layout()
    fig.savefig(pareto)
    plt.close(fig)
    return [pipeline, pareto]
