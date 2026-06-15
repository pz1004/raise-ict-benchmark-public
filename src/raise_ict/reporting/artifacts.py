"""Generate paper-table and figure artifacts from benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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


def _draw_pipeline_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
    edgecolor: str,
    *,
    title_size: float = 5.9,
    detail_size: float = 4.75,
    linewidth: float = 0.75,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color="#374151",
        linespacing=1.15,
    )


def _draw_pipeline_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    linewidth: float = 0.7,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _draw_pipeline_item(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        linewidth=0.45,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)
    ax.add_patch(
        Rectangle(
            (x + 0.018, y + height / 2 - 0.008),
            0.016,
            0.016,
            linewidth=0,
            facecolor="#1F4E79",
        )
    )
    ax.text(
        x + 0.045,
        y + height / 2,
        text,
        ha="left",
        va="center",
        fontsize=5.1,
        color="#1F2937",
    )


def _render_pipeline_figure(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.5, 2.82))
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(
            0.04,
            0.94,
            "RAISE-ICT row-admission workflow",
            ha="left",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color="#111827",
        )
        ax.plot([0.04, 0.96], [0.895, 0.895], color="#CBD5E1", lw=0.55)

        claim = (0.04, 0.735, 0.27, 0.135)
        candidate = (0.04, 0.565, 0.27, 0.135)
        gate = (0.385, 0.63, 0.26, 0.18)
        admit = (0.73, 0.735, 0.23, 0.12)
        reject = (0.73, 0.565, 0.23, 0.12)

        _draw_pipeline_box(
            ax,
            *claim,
            "Claim context $c$",
            "A0/A1/A4\nor CPU profile",
            "#FFFFFF",
            "#475569",
            title_size=5.7,
            detail_size=4.45,
        )
        _draw_pipeline_box(
            ax,
            *candidate,
            "Candidate row $r$",
            "score +\nartifacts",
            "#FFFFFF",
            "#475569",
            title_size=5.7,
            detail_size=4.45,
        )
        _draw_pipeline_box(
            ax,
            *gate,
            "Admission gate",
            "$A_c(r)=1$ iff\nresolve + match $c$",
            "#F8FAFC",
            "#1F4E79",
            title_size=5.5,
            detail_size=4.4,
            linewidth=0.95,
        )
        _draw_pipeline_box(
            ax,
            *admit,
            "Admit",
            "compare on\nchosen axis",
            "#FFFFFF",
            "#475569",
            title_size=5.7,
            detail_size=4.45,
        )
        _draw_pipeline_box(
            ax,
            *reject,
            "Not admitted",
            "not this\ncomparison",
            "#FFFFFF",
            "#475569",
            title_size=5.7,
            detail_size=4.45,
        )

        _draw_pipeline_arrow(ax, (0.31, 0.805), (0.377, 0.792))
        _draw_pipeline_arrow(ax, (0.31, 0.632), (0.377, 0.648))
        _draw_pipeline_arrow(ax, (0.653, 0.792), (0.722, 0.795))
        _draw_pipeline_arrow(ax, (0.653, 0.648), (0.722, 0.625), color="#64748B", linewidth=0.65)

        ax.text(
            0.04,
            0.505,
            "Required evidence fields",
            ha="left",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color="#111827",
        )

        fields = [
            "Dataset + sample scope",
            "Split + preprocessing",
            "Condition + control role",
            "Model + seed",
            "Threat + validity",
            "Profile + timing",
            "Service-cost rule",
            "Rows, citation, audit",
        ]
        x_positions = [0.06, 0.535]
        y_positions = [0.39, 0.305, 0.22, 0.135]
        field_index = 0
        for y in y_positions:
            for x in x_positions:
                _draw_pipeline_item(ax, x, y, 0.41, 0.064, fields[field_index])
                field_index += 1

        ax.text(
            0.5,
            0.055,
            "Substitution blocked: sample, split, threat, profile, cost.",
            ha="center",
            va="center",
            fontsize=5.4,
            fontstyle="italic",
            color="#475569",
        )

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out_path


MODEL_PARETO_STYLES = {
    "extra_trees": ("ET", "s", "#6B7280"),
    "hist_gradient_boosting": ("HGB", "D", "#2F6B3F"),
    "logistic_regression": ("LR", "o", "#1F4E79"),
    "mlp_sklearn": ("MLP", "^", "#8C2D04"),
    "random_forest": ("RF", "v", "#6B4E9B"),
}


def _score_search_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if "threat_id" not in summary.columns:
        return summary
    score_search = summary[summary["threat_id"].eq("a1_constrained_score_search")]
    return score_search if not score_search.empty else summary


def _pareto_front(part: pd.DataFrame) -> pd.DataFrame:
    ordered = part.sort_values("p95_latency_ms")
    front_indices = []
    best_utility = float("-inf")
    for idx, row in ordered.iterrows():
        utility = float(row["robust_utility"])
        if utility > best_utility + 1e-12:
            front_indices.append(idx)
            best_utility = utility
    return ordered.loc[front_indices]


def _short_dataset_name(name: str) -> str:
    return {
        "CSE-CIC-IDS2018": "CSE-CIC-IDS2018",
        "CICIDS2017": "CICIDS2017",
        "TON_IoT": "TON-IoT",
        "UNSW-NB15": "UNSW-NB15",
    }.get(name, name)


def _render_pareto_figure(results: pd.DataFrame, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _score_search_summary(summarize_results(results)).copy()
    if "p95_latency_ms" in summary.columns:
        summary["p95_latency_us"] = summary["p95_latency_ms"] * 1000.0

    required = {"dataset", "model_id", "p95_latency_us", "robust_utility"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0), sharex=True, sharey=True)
        axes_flat = axes.ravel()

        if not required.issubset(summary.columns) or summary.empty:
            for ax in axes_flat:
                ax.axis("off")
            axes_flat[0].text(
                0.5,
                0.5,
                "No admitted Pareto rows",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#475569",
            )
        else:
            dataset_order = ["CICIDS2017", "CSE-CIC-IDS2018", "TON_IoT", "UNSW-NB15"]
            datasets = [name for name in dataset_order if name in set(summary["dataset"])]
            datasets.extend(sorted(set(summary["dataset"]) - set(datasets)))

            x_min = max(float(summary["p95_latency_us"].min()) * 0.6, 1e-4)
            x_max = float(summary["p95_latency_us"].max()) * 1.45
            y_min = max(0.0, float(summary["robust_utility"].min()) - 0.05)
            y_max = min(1.0, float(summary["robust_utility"].max()) + 0.04)

            for ax, dataset in zip(axes_flat, datasets, strict=False):
                part = summary[summary["dataset"].eq(dataset)]
                front = _pareto_front(part)
                front_models = set(front["model_id"])

                if len(front) > 1:
                    ax.plot(
                        front["p95_latency_us"],
                        front["robust_utility"],
                        color="#111827",
                        linewidth=0.65,
                        zorder=2,
                    )

                for _, row in part.iterrows():
                    label, marker, color = MODEL_PARETO_STYLES.get(
                        row["model_id"],
                        (str(row["model_id"]), "o", "#374151"),
                    )
                    is_front = row["model_id"] in front_models
                    ax.scatter(
                        row["p95_latency_us"],
                        row["robust_utility"],
                        s=26 if is_front else 20,
                        marker=marker,
                        facecolors=color if is_front else "white",
                        edgecolors=color,
                        linewidths=0.85,
                        alpha=0.95 if is_front else 0.75,
                        zorder=3 if is_front else 2,
                    )

                ax.set_title(_short_dataset_name(str(dataset)), fontsize=6.8, pad=2.0)
                ax.set_xscale("log")
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, which="major", color="#E5E7EB", linewidth=0.45)
                ax.tick_params(axis="both", labelsize=5.7, pad=1.2)
                from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

                ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=20))
                ax.xaxis.set_minor_formatter(NullFormatter())

            for ax in axes_flat[len(datasets) :]:
                ax.axis("off")

            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=4.5,
                    label=label,
                )
                for _, (label, marker, color) in MODEL_PARETO_STYLES.items()
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=5,
                frameon=False,
                fontsize=5.7,
                handletextpad=0.25,
                columnspacing=0.6,
                bbox_to_anchor=(0.62, 0.165),
            )

        fig.supxlabel(r"p95 latency ($\mu$s, log scale)", fontsize=6.5, x=0.62, y=0.245)
        fig.supylabel("robust utility", fontsize=6.5, x=0.18)
        fig.tight_layout(rect=(0.09, 0.14, 1.0, 0.995), h_pad=0.5, w_pad=0.45)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out_path


def render_placeholder_figures(results: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Render the pipeline and Pareto figures expected by the paper scaffold."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = out / "figure_pipeline.pdf"
    pareto = out / "figure_pareto.pdf"

    _render_pipeline_figure(pipeline)

    _render_pareto_figure(results, pareto)
    return [pipeline, pareto]
