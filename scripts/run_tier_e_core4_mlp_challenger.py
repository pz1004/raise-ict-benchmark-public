#!/usr/bin/env python
"""Run or print the Tier-E Core4 MLP challenger orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402


CORE4_HARDWARE_AUDIT_PATH = "manifests/hardware/tier_e_hardware_audit.json"
CHALLENGER_HARDWARE_AUDIT_PATH = "manifests/hardware/tier_e_mlp_challenger_hardware_audit.json"
CORE4_RAW_PATH = "results/tables/tier_e_core4/table_raw_results.csv"
CORE4_SUMMARY_PATH = "results/tables/tier_e_core4/table_main_results.csv"
CORE4_SPLIT_PATH = "manifests/splits/tier_e_core4_split_manifest.csv"
CORE4_FEATURE_PATH = "manifests/feature_schemas/tier_e_core4_feature_schema.json"
CORE4_PROFILE_PATH = "manifests/hardware/tier_e_profile_manifest.json"
CORE4_PRECHECK_PATH = "/tmp/raise_ict_tier_e_core4_precheck.json"
CHALLENGER_AUDIT_PATH = "manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json"

COMBINED_PROFILE_PATH = "manifests/hardware/tier_e_core4_mlp_challenger_profile_manifest.json"
COMBINED_RAW_PATH = "results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv"
COMBINED_SUMMARY_PATH = "results/tables/tier_e_core4_mlp_challenger/table_main_results.csv"
COMBINED_SPLIT_PATH = "manifests/splits/tier_e_core4_mlp_challenger_split_manifest.csv"
COMBINED_FEATURE_PATH = "manifests/feature_schemas/tier_e_core4_mlp_challenger_feature_schema.json"
COMBINED_TABLE_DIR = "results/tables/tier_e_core4_mlp_challenger"
COMBINED_FIGURE_DIR = "results/figures/tier_e_core4_mlp_challenger"
COMBINED_ANALYSIS_DIR = "results/analysis/tier_e_core4_mlp_challenger"

RANDOM_CONTROL_RAW_PATH = "results/tables/tier_e_random_control_mlp_challenger/table_raw_results.csv"
RANDOM_CONTROL_SUMMARY_PATH = "results/tables/tier_e_random_control_mlp_challenger/table_main_results.csv"
RANDOM_CONTROL_SPLIT_PATH = "manifests/splits/tier_e_random_control_mlp_challenger_split_manifest.csv"
RANDOM_CONTROL_FEATURE_PATH = "manifests/feature_schemas/tier_e_random_control_mlp_challenger_feature_schema.json"
RANDOM_CONTROL_TABLE_DIR = "results/tables/tier_e_random_control_mlp_challenger"
RANDOM_CONTROL_FIGURE_DIR = "results/figures/tier_e_random_control_mlp_challenger"

EXPECTED_MODELS = ["extra_trees", "logistic_regression", "mlp_sklearn", "random_forest"]
PRIMARY_THREAT = "a1_constrained_score_search"

CLASSICAL_RAW_DIRS = [
    "results/raw/tier_e_expanded",
    "results/raw/tier_e_cicids2017",
    "results/raw/tier_e_cse_cic_ids2018",
]

MLP_EDGE_RUNS = [
    {
        "name": "expanded_mlp",
        "config": "configs/experiments/tier_p_expanded_mlp.yaml",
        "raw": "results/raw/tier_e_expanded_mlp",
        "split": "manifests/splits/tier_e_expanded_mlp_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_expanded_mlp_feature_schema.json",
        "profile": "manifests/hardware/tier_e_expanded_mlp_profile_manifest.json",
    },
    {
        "name": "cicids2017_mlp",
        "config": "configs/experiments/tier_p_cicids2017_mlp.yaml",
        "raw": "results/raw/tier_e_cicids2017_mlp",
        "split": "manifests/splits/tier_e_cicids2017_mlp_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cicids2017_mlp_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cicids2017_mlp_profile_manifest.json",
    },
    {
        "name": "cse_cic_ids2018_mlp",
        "config": "configs/experiments/tier_p_cse_cic_ids2018_mlp.yaml",
        "raw": "results/raw/tier_e_cse_cic_ids2018_mlp",
        "split": "manifests/splits/tier_e_cse_cic_ids2018_mlp_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cse_cic_ids2018_mlp_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cse_cic_ids2018_mlp_profile_manifest.json",
    },
]

RANDOM_CONTROL_RUNS = [
    {
        "name": "cicids2017_random_control_classical",
        "config": "configs/experiments/tier_p_cicids2017_random_control.yaml",
        "raw": "results/raw/tier_e_cicids2017_random_control",
        "split": "manifests/splits/tier_e_cicids2017_random_control_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cicids2017_random_control_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cicids2017_random_control_profile_manifest.json",
    },
    {
        "name": "cse_cic_ids2018_random_control_classical",
        "config": "configs/experiments/tier_p_cse_cic_ids2018_random_control.yaml",
        "raw": "results/raw/tier_e_cse_cic_ids2018_random_control",
        "split": "manifests/splits/tier_e_cse_cic_ids2018_random_control_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cse_cic_ids2018_random_control_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cse_cic_ids2018_random_control_profile_manifest.json",
    },
    {
        "name": "cicids2017_random_control_mlp",
        "config": "configs/experiments/tier_p_cicids2017_random_control_mlp.yaml",
        "raw": "results/raw/tier_e_cicids2017_random_control_mlp",
        "split": "manifests/splits/tier_e_cicids2017_random_control_mlp_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cicids2017_random_control_mlp_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cicids2017_random_control_mlp_profile_manifest.json",
    },
    {
        "name": "cse_cic_ids2018_random_control_mlp",
        "config": "configs/experiments/tier_p_cse_cic_ids2018_random_control_mlp.yaml",
        "raw": "results/raw/tier_e_cse_cic_ids2018_random_control_mlp",
        "split": "manifests/splits/tier_e_cse_cic_ids2018_random_control_mlp_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cse_cic_ids2018_random_control_mlp_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cse_cic_ids2018_random_control_mlp_profile_manifest.json",
    },
]


def _command(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _profile_manifest_command(hardware_config: str) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4_mlp_challenger.py",
        "--hardware-config",
        hardware_config,
        "--write-profile-manifest-only",
    )


def _acceptance_report_command(hardware_config: str) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4_mlp_challenger.py",
        "--hardware-config",
        hardware_config,
        "--write-acceptance-report-only",
    )


def _run_benchmark_command(run: dict[str, str], hardware_config: str) -> list[str]:
    return _command(
        "scripts/run_benchmark.py",
        "--config",
        run["config"],
        "--out-dir",
        run["raw"],
        "--split-manifest",
        run["split"],
        "--feature-schema",
        run["features"],
        "--hardware-config",
        hardware_config,
        "--profile-manifest",
        run["profile"],
    )


def _core4_precheck_command(
    strict: bool = True,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
) -> list[str]:
    return _command(
        "scripts/check_completion.py",
        "--out",
        CORE4_PRECHECK_PATH,
        "--raw-results",
        CORE4_RAW_PATH,
        "--summary-results",
        CORE4_SUMMARY_PATH,
        "--split-manifest",
        CORE4_SPLIT_PATH,
        "--dataset-manifest",
        "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
        "--feature-schema",
        CORE4_FEATURE_PATH,
        "--hardware-audit",
        CORE4_HARDWARE_AUDIT_PATH,
        "--profile-manifest",
        CORE4_PROFILE_PATH,
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
        "--require-tier-e",
        *(["--strict"] if strict else []),
    )


def _core4_rerun_command(
    hardware_config: str,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4.py",
        "--hardware-config",
        hardware_config,
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
    )


def _combined_audit_command(
    strict: bool = True,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
) -> list[str]:
    return _command(
        "scripts/check_completion.py",
        "--out",
        CHALLENGER_AUDIT_PATH,
        "--raw-results",
        COMBINED_RAW_PATH,
        "--summary-results",
        COMBINED_SUMMARY_PATH,
        "--split-manifest",
        COMBINED_SPLIT_PATH,
        "--dataset-manifest",
        "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
        "--feature-schema",
        COMBINED_FEATURE_PATH,
        "--hardware-audit",
        CHALLENGER_HARDWARE_AUDIT_PATH,
        "--profile-manifest",
        COMBINED_PROFILE_PATH,
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
        "--expected-raw-rows",
        "320",
        "--expected-summary-rows",
        "64",
        "--expected-models",
        *EXPECTED_MODELS,
        "--require-tier-e",
        *(["--strict"] if strict else []),
    )


def _analysis_command() -> list[str]:
    return _command(
        "scripts/analyze_results.py",
        "--raw",
        COMBINED_RAW_PATH,
        "--summary",
        COMBINED_SUMMARY_PATH,
        "--out",
        COMBINED_ANALYSIS_DIR,
        "--attack-threat",
        PRIMARY_THREAT,
        "--label",
        "Tier-E Core4 MLP Challenger",
        "--scope-note",
        "physical Jetson Core4 run with three classical baselines and one sklearn MLP challenger",
        "--split-manifest",
        COMBINED_SPLIT_PATH,
        "--dataset-manifest",
        "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
    )


def _aggregate_core4_command() -> list[str]:
    raw_dirs = [*CLASSICAL_RAW_DIRS, *(run["raw"] for run in MLP_EDGE_RUNS)]
    return _command(
        "scripts/aggregate_results.py",
        "--results",
        *raw_dirs,
        "--out",
        COMBINED_TABLE_DIR,
        "--figures",
        COMBINED_FIGURE_DIR,
    )


def _aggregate_random_control_command() -> list[str]:
    return _command(
        "scripts/aggregate_results.py",
        "--results",
        *(run["raw"] for run in RANDOM_CONTROL_RUNS),
        "--out",
        RANDOM_CONTROL_TABLE_DIR,
        "--figures",
        RANDOM_CONTROL_FIGURE_DIR,
    )


def _merge_split_command(runs: list[dict[str, str]], out: str) -> list[str]:
    return _command("scripts/merge_artifacts.py", *(run["split"] for run in runs), "--out", out)


def _merge_feature_command(runs: list[dict[str, str]], out: str) -> list[str]:
    return _command("scripts/merge_artifacts.py", *(run["features"] for run in runs), "--out", out)


def build_commands(
    hardware_config: str,
    strict: bool = True,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
) -> list[list[str]]:
    """Build the full single-command graph without executing it."""
    commands = [
        _command("scripts/validate_hardware_config.py", "--config", hardware_config),
        _command("scripts/audit_hardware.py", "--out", CHALLENGER_HARDWARE_AUDIT_PATH),
        _core4_precheck_command(strict=True, manuscript=manuscript, bibliography=bibliography),
        _core4_rerun_command(hardware_config, manuscript=manuscript, bibliography=bibliography),
        *[_run_benchmark_command(run, hardware_config) for run in MLP_EDGE_RUNS],
        _merge_split_command(MLP_EDGE_RUNS, COMBINED_SPLIT_PATH),
        _merge_feature_command(MLP_EDGE_RUNS, COMBINED_FEATURE_PATH),
        _aggregate_core4_command(),
        _analysis_command(),
        *[_run_benchmark_command(run, hardware_config) for run in RANDOM_CONTROL_RUNS],
        _merge_split_command(RANDOM_CONTROL_RUNS, RANDOM_CONTROL_SPLIT_PATH),
        _merge_feature_command(RANDOM_CONTROL_RUNS, RANDOM_CONTROL_FEATURE_PATH),
        _aggregate_random_control_command(),
        _profile_manifest_command(hardware_config),
        _acceptance_report_command(hardware_config),
        _combined_audit_command(strict=strict, manuscript=manuscript, bibliography=bibliography),
    ]
    return commands


def _ensure_tier_e_host(audit_path: str = CHALLENGER_HARDWARE_AUDIT_PATH) -> None:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    classification = audit.get("classification", {})
    if not classification.get("tier_e_eligible"):
        reasons = classification.get("reasons", [])
        raise SystemExit(
            "Refusing to run Tier-E MLP rows on a non-edge host. "
            f"Hardware audit reasons: {reasons}"
        )


def _core4_audit_passes(manuscript: str, bibliography: str) -> bool:
    completed = subprocess.run(
        _core4_precheck_command(strict=True, manuscript=manuscript, bibliography=bibliography),
        check=False,
    )
    return completed.returncode == 0


def _read_profile(path: str) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    return json.loads(profile_path.read_text(encoding="utf-8"))


def write_combined_profile_manifest(
    hardware_config: str,
    out: str = COMBINED_PROFILE_PATH,
) -> Path:
    """Merge existing Core4 and MLP profile manifests for the combined audit."""
    hardware = load_yaml(hardware_config)
    component_paths = [
        CORE4_PROFILE_PATH,
        *(run["profile"] for run in MLP_EDGE_RUNS),
    ]
    profiles = [_read_profile(path) for path in component_paths]
    energy_values = [
        float(profile.get("profile", {}).get("energy_per_flow_j", 0.0) or 0.0)
        for profile in profiles
    ]
    latency_values = [
        float(profile.get("profile", {}).get("p95_latency_ms_median", 0.0) or 0.0)
        for profile in profiles
    ]
    manifest = {
        "schema_version": 1,
        "hardware": hardware,
        "profile": {
            "hardware_id": hardware.get("hardware_id", "unknown"),
            "energy_per_flow_j": min(energy_values) if energy_values else 0.0,
            "energy_per_flow_j_min": min(energy_values) if energy_values else 0.0,
            "energy_per_flow_j_max": max(energy_values) if energy_values else 0.0,
            "p95_latency_ms_median": sorted(latency_values)[len(latency_values) // 2] if latency_values else 0.0,
        },
        "component_manifests": component_paths,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def _read_csv(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    return pd.read_csv(table_path)


def _metric_row(summary: pd.DataFrame, dataset: str, model_id: str, threat_id: str) -> pd.Series | None:
    part = summary[
        summary["dataset"].eq(dataset)
        & summary["model_id"].eq(model_id)
        & summary["threat_id"].eq(threat_id)
    ]
    if part.empty:
        return None
    return part.iloc[0]


def _best_classical(summary: pd.DataFrame, dataset: str, threat_id: str, metric: str, higher: bool) -> pd.Series | None:
    part = summary[
        summary["dataset"].eq(dataset)
        & summary["threat_id"].eq(threat_id)
        & summary["model_id"].isin([model for model in EXPECTED_MODELS if model != "mlp_sklearn"])
    ]
    if part.empty:
        return None
    return part.sort_values(metric, ascending=not higher).iloc[0]


def _format_value(row: pd.Series | None, metric: str) -> str:
    if row is None or metric not in row:
        return "n/a"
    return f"{float(row[metric]):.4g}"


def _pareto_front(summary: pd.DataFrame) -> dict[str, list[str]]:
    front: dict[str, list[str]] = {}
    attack = summary[summary["threat_id"].eq(PRIMARY_THREAT)].copy()
    for dataset, part in attack.groupby("dataset"):
        labels: list[str] = []
        for _, row in part.iterrows():
            dominated = False
            for _, other in part.iterrows():
                if row["model_id"] == other["model_id"]:
                    continue
                latency_no_worse = float(other["p95_latency_ms"]) <= float(row["p95_latency_ms"])
                utility_no_worse = float(other["robust_utility"]) >= float(row["robust_utility"])
                strictly_better = (
                    float(other["p95_latency_ms"]) < float(row["p95_latency_ms"])
                    or float(other["robust_utility"]) > float(row["robust_utility"])
                )
                if latency_no_worse and utility_no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                labels.append(str(row["model_id"]))
        front[str(dataset)] = sorted(labels)
    return front


def _count_line(path: str, expected: int) -> str:
    frame = _read_csv(path)
    status = "OK" if len(frame) == expected else f"EXPECTED {expected}"
    return f"- `{path}`: {len(frame)} rows ({status})."


def write_acceptance_evidence_report(
    out: str = f"{COMBINED_ANALYSIS_DIR}/acceptance-evidence-report.md",
) -> Path:
    """Write a reviewer-facing report for deciding whether to revise the manuscript."""
    summary = _read_csv(COMBINED_SUMMARY_PATH)
    random_summary = _read_csv(RANDOM_CONTROL_SUMMARY_PATH)
    front = _pareto_front(summary)
    lines = [
        "# Tier-E Core4 MLP Challenger Acceptance Evidence",
        "",
        "## Scope",
        "",
        _count_line(COMBINED_RAW_PATH, 320),
        _count_line(COMBINED_SUMMARY_PATH, 64),
        _count_line(RANDOM_CONTROL_RAW_PATH, 40),
        _count_line(RANDOM_CONTROL_SUMMARY_PATH, 8),
        f"- Primary constrained threat: `{PRIMARY_THREAT}`.",
        "- Energy remains shared INA3221 `VDD_IN` module-power context, not per-model isolated energy.",
        "",
        "## Pareto Frontier Check",
        "",
    ]
    for dataset, models in front.items():
        changed = "yes" if "mlp_sklearn" in models else "no"
        lines.append(f"- {dataset}: front={', '.join(models)}; MLP on front={changed}.")

    lines.extend(
        [
            "",
            "## MLP Versus Best Classical Reference",
            "",
            "| Dataset | Metric | MLP | Best classical | Classical model | Direction |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    comparisons = [
        ("clean_macro_f1", True, "higher"),
        ("robust_utility", True, "higher"),
        ("asr", False, "lower"),
        ("p95_latency_ms", False, "lower"),
        ("peak_mem_mb", False, "lower"),
        ("service_cost", False, "lower"),
    ]
    for dataset in sorted(summary["dataset"].dropna().unique()):
        mlp = _metric_row(summary, str(dataset), "mlp_sklearn", PRIMARY_THREAT)
        for metric, higher, direction in comparisons:
            classical = _best_classical(summary, str(dataset), PRIMARY_THREAT, metric, higher)
            classical_model = "n/a" if classical is None else str(classical["model_id"])
            lines.append(
                f"| {dataset} | {metric} | {_format_value(mlp, metric)} | "
                f"{_format_value(classical, metric)} | {classical_model} | {direction} is better |"
            )

    lines.extend(
        [
            "",
            "## Random-Split Versus Held-Out Contrast",
            "",
            "| Dataset | Model | Random-control clean F1 | Held-out score-search clean F1 |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset in ["CICIDS2017", "CSE-CIC-IDS2018"]:
        for model_id in EXPECTED_MODELS:
            random_row = _metric_row(random_summary, dataset, model_id, "a0_clean")
            heldout_row = _metric_row(summary, dataset, model_id, PRIMARY_THREAT)
            lines.append(
                f"| {dataset} | {model_id} | {_format_value(random_row, 'clean_macro_f1')} | "
                f"{_format_value(heldout_row, 'clean_macro_f1')} |"
            )

    lines.extend(
        [
            "",
            "## Claim Candidates",
            "",
            "- Allowed: The single-command Tier-E challenger run evaluates one CPU-compatible neural IDS challenger under the same dataset, split, threat, seed, hardware, validity, and profiling fields as the Core4 references.",
            "- Allowed: The MLP challenger can be discussed only after the 320-row strict audit passes at `manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json`.",
            "- Allowed: Random-control rows remain pipeline-sanity evidence and should be contrasted with held-out score-search rows.",
            "- Forbidden: Do not claim a general neural IDS leaderboard, SOTA result, packet-level attack realizability, per-model isolated energy, or hardware-independent efficiency.",
            "",
            "## Manuscript Use",
            "",
            "- Do not update `jkics/jkics.tex` from the current 240-row claim until the 320-row audit is complete.",
            "- If the 320-row audit passes, revise the manuscript from three classical baselines to three classical baselines plus one CPU-compatible MLP challenger.",
        ]
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    hardware_config: str,
    strict: bool = True,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
) -> None:
    _run(_command("scripts/validate_hardware_config.py", "--config", hardware_config))
    _run(_command("scripts/audit_hardware.py", "--out", CHALLENGER_HARDWARE_AUDIT_PATH))
    _ensure_tier_e_host()

    if _core4_audit_passes(manuscript=manuscript, bibliography=bibliography):
        print(f"Reusing existing strict Tier-E Core4 package; precheck wrote {CORE4_PRECHECK_PATH}", flush=True)
    else:
        print("Existing Tier-E Core4 package did not pass strict precheck; rerunning Core4.", flush=True)
        _run(_core4_rerun_command(hardware_config, manuscript=manuscript, bibliography=bibliography))
        _run(_core4_precheck_command(strict=True, manuscript=manuscript, bibliography=bibliography))

    for run in MLP_EDGE_RUNS:
        _run(_run_benchmark_command(run, hardware_config))
    _run(_merge_split_command(MLP_EDGE_RUNS, COMBINED_SPLIT_PATH))
    _run(_merge_feature_command(MLP_EDGE_RUNS, COMBINED_FEATURE_PATH))
    _run(_aggregate_core4_command())
    _run(_analysis_command())

    for run in RANDOM_CONTROL_RUNS:
        _run(_run_benchmark_command(run, hardware_config))
    _run(_merge_split_command(RANDOM_CONTROL_RUNS, RANDOM_CONTROL_SPLIT_PATH))
    _run(_merge_feature_command(RANDOM_CONTROL_RUNS, RANDOM_CONTROL_FEATURE_PATH))
    _run(_aggregate_random_control_command())

    print(write_combined_profile_manifest(hardware_config), flush=True)
    print(write_acceptance_evidence_report(), flush=True)
    _run(_combined_audit_command(strict=strict, manuscript=manuscript, bibliography=bibliography))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--write-profile-manifest-only", action="store_true")
    parser.add_argument("--write-acceptance-report-only", action="store_true")
    parser.add_argument("--manuscript", default="anonymous_manuscript.tex")
    parser.add_argument("--bibliography", default="anonymous_references.bib")
    args = parser.parse_args()

    if args.write_profile_manifest_only:
        print(write_combined_profile_manifest(args.hardware_config))
        return
    if args.write_acceptance_report_only:
        print(write_acceptance_evidence_report())
        return
    if args.dry_run:
        payload = {
            "commands": build_commands(
                args.hardware_config,
                strict=not args.no_strict,
                manuscript=args.manuscript,
                bibliography=args.bibliography,
            ),
            "notes": [
                "The run_tier_e_core4.py command is conditional: it runs only if the classical Core4 precheck fails.",
                f"The challenger audit writes {CHALLENGER_AUDIT_PATH}; it does not overwrite manuscript-side audit files.",
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    run_experiment(
        args.hardware_config,
        strict=not args.no_strict,
        manuscript=args.manuscript,
        bibliography=args.bibliography,
    )


if __name__ == "__main__":
    main()
