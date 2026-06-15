#!/usr/bin/env python
"""Create an analysis bundle for RAISE-ICT experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _bootstrap import bootstrap

bootstrap()

from raise_ict.stats import normal_mean_ci  # noqa: E402


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    """Return tuple form expected by the analysis table builder."""
    summary = normal_mean_ci(values)
    return summary["mean"], summary["std"], summary["ci_low"], summary["ci_high"]


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm correction while preserving input p-value order."""
    order = np.argsort(p_values)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_attack_drop(raw: pd.DataFrame, attack_threat: str) -> pd.DataFrame:
    """Summarize paired clean-vs-attack robust-utility drops."""
    clean = raw[raw["threat_id"] == "a0_clean"].set_index(["dataset", "model_id", "seed"])
    attack = raw[raw["threat_id"] == attack_threat].copy()
    for column, default in {"valid_count": 0, "invalid_count": 0, "validity_rate": 1.0}.items():
        if column not in attack.columns:
            attack[column] = default
    attack = attack.set_index(["dataset", "model_id", "seed"])
    joined = clean[["robust_utility"]].join(
        attack[["robust_utility", "asr", "validity_rate", "valid_count", "invalid_count"]],
        how="inner",
        lsuffix="_clean",
        rsuffix="_attack",
    ).reset_index()
    joined["robust_drop"] = joined["robust_utility_clean"] - joined["robust_utility_attack"]

    rows = []
    p_values = []
    for (dataset, model_id), part in joined.groupby(["dataset", "model_id"]):
        diff = part["robust_drop"]
        mean, std, low, high = mean_ci(diff)
        if len(diff) >= 2 and not np.allclose(diff, 0.0):
            test = stats.wilcoxon(diff)
            p_value = float(test.pvalue)
        else:
            p_value = 1.0
        p_values.append(p_value)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "n_pairs": int(len(diff)),
                "mean_robust_drop": mean,
                "std_robust_drop": std,
                "ci_low": low,
                "ci_high": high,
                "mean_asr": float(part["asr"].mean()),
                "mean_validity_rate": float(part["validity_rate"].mean()),
                "mean_valid_count": float(part["valid_count"].mean()),
                "mean_invalid_count": float(part["invalid_count"].mean()),
                "wilcoxon_p": p_value,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted, strict=True):
        row["holm_p"] = p_adj
    return pd.DataFrame(rows).sort_values(["dataset", "model_id"]).reset_index(drop=True)


def render_figures(summary: pd.DataFrame, drops: pd.DataFrame, out_dir: Path, attack_threat: str) -> list[Path]:
    """Render analysis figures for the constrained-attack bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    subset = summary[summary["threat_id"].isin(["a0_clean", attack_threat])].copy()
    subset["label"] = subset["dataset"] + "\n" + subset["model_id"] + "\n" + subset["threat_id"]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(range(len(subset)), subset["robust_utility"], yerr=subset["robust_utility_std"], capsize=2)
    ax.set_xticks(range(len(subset)), subset["label"], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("robust utility")
    ax.set_title("Clean vs. constrained-feature utility by dataset and model")
    fig.tight_layout()
    path = out_dir / "figure-01-clean-vs-constrained.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    attack = summary[summary["threat_id"] == attack_threat].copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for dataset, part in attack.groupby("dataset"):
        ax.scatter(part["p95_latency_ms"], part["robust_utility"], label=dataset)
        for _, row in part.iterrows():
            ax.annotate(row["model_id"], (row["p95_latency_ms"], row["robust_utility"]), fontsize=7)
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("robust utility")
    ax.set_title("Constrained-attack robust utility vs. p95 latency")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "figure-02-pareto-constrained.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels = drops["dataset"] + "\n" + drops["model_id"]
    ax.bar(range(len(drops)), drops["mean_robust_drop"], yerr=drops["std_robust_drop"], capsize=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(drops)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("a0 utility - a1 utility")
    ax.set_title("Paired robust-utility drop under constrained perturbations")
    fig.tight_layout()
    path = out_dir / "figure-03-robust-drop.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def write_bundle(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    raw_path: str,
    summary_path: str,
    split_manifest_path: str,
    dataset_manifest_path: str,
    attack_threat: str,
    label: str,
    scope_note: str,
    caveats: list[str],
) -> None:
    """Write the markdown, CSV, and PDF analysis bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drops = paired_attack_drop(raw, attack_threat)
    drops.to_csv(out_dir / "paired_attack_drop.csv", index=False)
    figure_paths = render_figures(summary, drops, out_dir / "figures", attack_threat)

    best_attack = summary[summary["threat_id"] == attack_threat].sort_values(
        ["dataset", "robust_utility"], ascending=[True, False]
    )
    best_lines = []
    for dataset, part in best_attack.groupby("dataset"):
        row = part.iloc[0]
        best_lines.append(
            f"- {dataset}: {row['model_id']} has the highest constrained robust utility "
            f"({row['robust_utility']:.3f} mean over {int(row['n_runs'])} seeds)."
        )
    seed_counts = raw.groupby(["dataset", "model_id", "threat_id"]).seed.nunique()

    report = [
        f"# RAISE-ICT {label} Analysis Report",
        "",
        "## Scope",
        "",
        f"- Raw rows: {len(raw)}.",
        f"- Datasets: {', '.join(sorted(raw['dataset'].unique()))}.",
        f"- Models: {', '.join(sorted(raw['model_id'].unique()))}.",
        f"- Threat rows: {', '.join(sorted(raw['threat_id'].unique()))}.",
        f"- Primary constrained threat for paired analysis: {attack_threat}.",
        f"- Seeds per dataset/model/threat: {seed_counts.min()} to {seed_counts.max()}.",
        "",
        "## Main Findings",
        "",
        *best_lines,
        "",
        "Constrained perturbations produce measurable robust-utility changes in this run. "
        f"These findings are from {scope_note}, not field-wide ranking claims.",
        "",
        "## Claim Candidates",
        "",
        f"- Claim: The harness executes the {label} experiment grid on public intrusion-detection datasets.",
        f"  - Source evidence: `{raw_path}`, `{dataset_manifest_path}`, and `{split_manifest_path}`.",
        f"  - Allowed wording: \"The {label} run validates the RAISE-ICT execution path for {scope_note}.\"",
        "  - Forbidden stronger wording: \"RAISE-ICT establishes a final field-wide ranking\" "
        "or \"model X is generally best.\"",
        "",
        f"- Claim: The constrained-attack rows in the {label} run report explicit validity counts "
        "and pass rates for the implemented budget, bounds, immutable-field, and relation checks.",
        "  - Source evidence: `valid_count`, `invalid_count`, `validity_rate`, and component pass-rate columns.",
        "  - Allowed wording: \"Generated constrained-feature examples are filtered by the implemented validity checks.\"",
        "  - Forbidden stronger wording: \"The attacks are packet-realizable.\"",
        "",
        "## Caveats",
        "",
        *[f"- {caveat}" for caveat in caveats],
    ]
    (out_dir / "analysis-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    min_seed_count = int(seed_counts.min())
    max_seed_count = int(seed_counts.max())
    if min_seed_count == max_seed_count:
        seed_phrase = f"{min_seed_count} seeds"
    else:
        seed_phrase = f"{min_seed_count} to {max_seed_count} seeds"

    stats_md = [
        f"# RAISE-ICT {label} Stats Appendix",
        "",
        "## Paired Constrained-Attack Drop",
        "",
        drops.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Wilcoxon signed-rank tests compare matched seed-level `a0_clean` and `{attack_threat}` "
        "robust utility within each dataset and model. Holm correction is applied across the "
        "reported contrasts.",
        "",
        "## Summary Table Source",
        "",
        f"`{summary_path}` contains mean, standard deviation, and normal-approximation 95% "
        f"confidence intervals over {seed_phrase}.",
    ]
    (out_dir / "stats-appendix.md").write_text("\n".join(stats_md) + "\n", encoding="utf-8")

    catalog = [
        "# RAISE-ICT Figure Catalog",
        "",
    ]
    for path in figure_paths:
        catalog.extend(
            [
                f"## {path.name}",
                f"- File: `{path}`",
                f"- Data source: `{summary_path}` and `{out_dir / 'paired_attack_drop.csv'}`.",
                "- Purpose: show the constrained-attack comparison that can be discussed in the "
                "manuscript without claiming a final field-wide ranking.",
                f"- Caveat: {scope_note}.",
                "",
            ]
        )
    (out_dir / "figure-catalog.md").write_text("\n".join(catalog), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/tables/table_raw_results.csv")
    parser.add_argument("--summary", default="results/tables/table_main_results.csv")
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--attack-threat", default="a1_constrained_feature")
    parser.add_argument("--label", default="Tier-P Pilot")
    parser.add_argument("--scope-note", default="sampled real-data runs")
    parser.add_argument("--caveat", action="append", default=None)
    parser.add_argument("--split-manifest", default="manifests/splits/tier_p_split_manifest.csv")
    parser.add_argument("--dataset-manifest", default="manifests/dataset_hashes/download_manifest.json")
    args = parser.parse_args()
    raw = pd.read_csv(args.raw)
    summary = pd.read_csv(args.summary)
    caveats = args.caveat or [
        "The run is an experiment-scope validation path, not a final field-wide ranking claim.",
        "Feature-space constrained attacks are not packet replay or simulator validated.",
        "Latency and energy claims are limited to the declared hardware tier and measurement metadata.",
        "Dataset-specific split limits are documented in the split manifest and should be "
        "carried into manuscript claims.",
    ]
    write_bundle(
        raw,
        summary,
        Path(args.out),
        args.raw,
        args.summary,
        args.split_manifest,
        args.dataset_manifest,
        args.attack_threat,
        args.label,
        args.scope_note,
        caveats,
    )
    print(args.out)


if __name__ == "__main__":
    main()
