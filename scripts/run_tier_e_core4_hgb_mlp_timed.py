#!/usr/bin/env python
"""Run or print the timed Tier-E Core4 HGB+MLP evidence upgrade."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.validation import audit_completion  # noqa: E402


RUN_ID = "tier_e_core4_hgb_mlp_timed"
EXPECTED_MODELS = [
    "extra_trees",
    "hist_gradient_boosting",
    "logistic_regression",
    "mlp_sklearn",
    "random_forest",
]
DEFAULT_SEEDS = list(range(10))
PRIMARY_THREAT = "a1_constrained_score_search"

HARDWARE_AUDIT_PATH = "manifests/hardware/tier_e_core4_hgb_mlp_timed_hardware_audit.json"
AUDIT_PATH = "manifests/completion/benchmark_completion_audit_strict_tier_e_core4_hgb_mlp_timed.json"
PRE_TIMING_AUDIT_PATH = "/tmp/raise_ict_tier_e_core4_hgb_mlp_timed_pre_timing_audit.json"

CONFIG_DIR = Path("results/configs") / RUN_ID
TIMING_DIR = Path("results/timing") / RUN_ID
COMMAND_TIMELINE_PATH = TIMING_DIR / "command_timeline.json"
TIMING_EVENTS_PATH = TIMING_DIR / "timing_events.csv"
TIMING_SUMMARY_PATH = TIMING_DIR / "timing_summary.csv"

MAIN_TABLE_DIR = f"results/tables/{RUN_ID}"
MAIN_FIGURE_DIR = f"results/figures/{RUN_ID}"
MAIN_ANALYSIS_DIR = f"results/analysis/{RUN_ID}"
MAIN_RAW_PATH = f"{MAIN_TABLE_DIR}/table_raw_results.csv"
MAIN_SUMMARY_PATH = f"{MAIN_TABLE_DIR}/table_main_results.csv"
MAIN_SPLIT_PATH = f"manifests/splits/{RUN_ID}_split_manifest.csv"
MAIN_FEATURE_PATH = f"manifests/feature_schemas/{RUN_ID}_feature_schema.json"
MAIN_PROFILE_PATH = f"manifests/hardware/{RUN_ID}_profile_manifest.json"

RANDOM_RUN_ID = "tier_e_random_control_hgb_mlp_timed"
RANDOM_TABLE_DIR = f"results/tables/{RANDOM_RUN_ID}"
RANDOM_FIGURE_DIR = f"results/figures/{RANDOM_RUN_ID}"
RANDOM_RAW_PATH = f"{RANDOM_TABLE_DIR}/table_raw_results.csv"
RANDOM_SUMMARY_PATH = f"{RANDOM_TABLE_DIR}/table_main_results.csv"
RANDOM_SPLIT_PATH = f"manifests/splits/{RANDOM_RUN_ID}_split_manifest.csv"
RANDOM_FEATURE_PATH = f"manifests/feature_schemas/{RANDOM_RUN_ID}_feature_schema.json"

REJECTION_REPORT_PATH = f"{MAIN_ANALYSIS_DIR}/admissibility_rejection_report.md"
REJECTION_JSON_PATH = f"{MAIN_ANALYSIS_DIR}/admissibility_rejection_report.json"
ACCEPTANCE_REPORT_PATH = f"{MAIN_ANALYSIS_DIR}/timed_acceptance_evidence_report.md"

MAIN_RUNS = [
    {
        "name": "expanded",
        "base_config": "configs/experiments/tier_p_expanded.yaml",
        "config": str(CONFIG_DIR / "expanded.yaml"),
        "raw": f"results/raw/{RUN_ID}_expanded",
        "split": f"manifests/splits/{RUN_ID}_expanded_split_manifest.csv",
        "features": f"manifests/feature_schemas/{RUN_ID}_expanded_feature_schema.json",
        "profile": f"manifests/hardware/{RUN_ID}_expanded_profile_manifest.json",
        "timing": str(TIMING_DIR / "expanded_events.csv"),
    },
    {
        "name": "cicids2017",
        "base_config": "configs/experiments/tier_p_cicids2017.yaml",
        "config": str(CONFIG_DIR / "cicids2017.yaml"),
        "raw": f"results/raw/{RUN_ID}_cicids2017",
        "split": f"manifests/splits/{RUN_ID}_cicids2017_split_manifest.csv",
        "features": f"manifests/feature_schemas/{RUN_ID}_cicids2017_feature_schema.json",
        "profile": f"manifests/hardware/{RUN_ID}_cicids2017_profile_manifest.json",
        "timing": str(TIMING_DIR / "cicids2017_events.csv"),
    },
    {
        "name": "cse_cic_ids2018",
        "base_config": "configs/experiments/tier_p_cse_cic_ids2018.yaml",
        "config": str(CONFIG_DIR / "cse_cic_ids2018.yaml"),
        "raw": f"results/raw/{RUN_ID}_cse_cic_ids2018",
        "split": f"manifests/splits/{RUN_ID}_cse_cic_ids2018_split_manifest.csv",
        "features": f"manifests/feature_schemas/{RUN_ID}_cse_cic_ids2018_feature_schema.json",
        "profile": f"manifests/hardware/{RUN_ID}_cse_cic_ids2018_profile_manifest.json",
        "timing": str(TIMING_DIR / "cse_cic_ids2018_events.csv"),
    },
]

RANDOM_RUNS = [
    {
        "name": "cicids2017_random_control",
        "base_config": "configs/experiments/tier_p_cicids2017_random_control.yaml",
        "config": str(CONFIG_DIR / "cicids2017_random_control.yaml"),
        "raw": f"results/raw/{RANDOM_RUN_ID}_cicids2017",
        "split": f"manifests/splits/{RANDOM_RUN_ID}_cicids2017_split_manifest.csv",
        "features": f"manifests/feature_schemas/{RANDOM_RUN_ID}_cicids2017_feature_schema.json",
        "profile": f"manifests/hardware/{RANDOM_RUN_ID}_cicids2017_profile_manifest.json",
        "timing": str(TIMING_DIR / "cicids2017_random_control_events.csv"),
    },
    {
        "name": "cse_cic_ids2018_random_control",
        "base_config": "configs/experiments/tier_p_cse_cic_ids2018_random_control.yaml",
        "config": str(CONFIG_DIR / "cse_cic_ids2018_random_control.yaml"),
        "raw": f"results/raw/{RANDOM_RUN_ID}_cse_cic_ids2018",
        "split": f"manifests/splits/{RANDOM_RUN_ID}_cse_cic_ids2018_split_manifest.csv",
        "features": f"manifests/feature_schemas/{RANDOM_RUN_ID}_cse_cic_ids2018_feature_schema.json",
        "profile": f"manifests/hardware/{RANDOM_RUN_ID}_cse_cic_ids2018_profile_manifest.json",
        "timing": str(TIMING_DIR / "cse_cic_ids2018_random_control_events.csv"),
    },
]


def _command(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time(), tz=timezone.utc).isoformat()


def _models() -> list[dict[str, str]]:
    return [{"model_id": model_id} for model_id in EXPECTED_MODELS]


def _write_generated_configs(seeds: list[int]) -> list[str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for run in [*MAIN_RUNS, *RANDOM_RUNS]:
        config = load_yaml(run["base_config"])
        config["experiment_id"] = f"{RUN_ID}_{run['name']}"
        config["models"] = _models()
        config["seeds"] = seeds
        path = Path(run["config"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(str(path))
    return written


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
        "--timing-events",
        run["timing"],
    )


def _merge_split_command(runs: list[dict[str, str]], out: str) -> list[str]:
    return _command("scripts/merge_artifacts.py", *(run["split"] for run in runs), "--out", out)


def _merge_feature_command(runs: list[dict[str, str]], out: str) -> list[str]:
    return _command("scripts/merge_artifacts.py", *(run["features"] for run in runs), "--out", out)


def _aggregate_command(runs: list[dict[str, str]], out: str, figures: str) -> list[str]:
    return _command("scripts/aggregate_results.py", "--results", *(run["raw"] for run in runs), "--out", out, "--figures", figures)


def _analysis_command() -> list[str]:
    return _command(
        "scripts/analyze_results.py",
        "--raw",
        MAIN_RAW_PATH,
        "--summary",
        MAIN_SUMMARY_PATH,
        "--out",
        MAIN_ANALYSIS_DIR,
        "--attack-threat",
        PRIMARY_THREAT,
        "--label",
        "Tier-E Timed Core4 HGB MLP",
        "--scope-note",
        "physical Jetson run with three classical baselines, HGB, and sklearn MLP over ten seeds",
        "--split-manifest",
        MAIN_SPLIT_PATH,
        "--dataset-manifest",
        "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
    )


def _profile_manifest_command(hardware_config: str) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4_hgb_mlp_timed.py",
        "--hardware-config",
        hardware_config,
        "--write-profile-manifest-only",
    )


def _acceptance_report_command(hardware_config: str) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4_hgb_mlp_timed.py",
        "--hardware-config",
        hardware_config,
        "--write-acceptance-report-only",
    )


def _rejection_suite_command(
    hardware_config: str,
    seeds: list[int],
    manuscript: str,
    bibliography: str,
) -> list[str]:
    return _command(
        "scripts/run_tier_e_core4_hgb_mlp_timed.py",
        "--hardware-config",
        hardware_config,
        "--run-rejection-suite-only",
        "--seeds",
        *(str(seed) for seed in seeds),
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
    )


def _audit_command(
    out: str,
    seeds: list[int],
    require_timing: bool,
    manuscript: str,
    bibliography: str,
) -> list[str]:
    command = _command(
        "scripts/check_completion.py",
        "--out",
        out,
        "--raw-results",
        MAIN_RAW_PATH,
        "--summary-results",
        MAIN_SUMMARY_PATH,
        "--split-manifest",
        MAIN_SPLIT_PATH,
        "--dataset-manifest",
        "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
        "--feature-schema",
        MAIN_FEATURE_PATH,
        "--hardware-audit",
        HARDWARE_AUDIT_PATH,
        "--profile-manifest",
        MAIN_PROFILE_PATH,
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
        "--expected-raw-rows",
        "800",
        "--expected-summary-rows",
        "80",
        "--expected-split-rows",
        "40",
        "--expected-feature-schema-records",
        "40",
        "--expected-seeds",
        *(str(seed) for seed in seeds),
        "--expected-models",
        *EXPECTED_MODELS,
        "--require-tier-e",
    )
    if require_timing:
        command.extend(
            [
                "--require-timing",
                "--timing-events",
                str(TIMING_EVENTS_PATH),
                "--timing-summary",
                str(TIMING_SUMMARY_PATH),
                "--command-timeline",
                str(COMMAND_TIMELINE_PATH),
            ]
        )
    command.append("--strict")
    return command


def build_commands(
    hardware_config: str,
    seeds: list[int],
    manuscript: str = "jkics/jkics.tex",
    bibliography: str = "jkics/reference.bib",
) -> list[list[str]]:
    return [
        _command("scripts/validate_hardware_config.py", "--config", hardware_config),
        _command("scripts/audit_hardware.py", "--out", HARDWARE_AUDIT_PATH),
        *[_run_benchmark_command(run, hardware_config) for run in MAIN_RUNS],
        _merge_split_command(MAIN_RUNS, MAIN_SPLIT_PATH),
        _merge_feature_command(MAIN_RUNS, MAIN_FEATURE_PATH),
        _aggregate_command(MAIN_RUNS, MAIN_TABLE_DIR, MAIN_FIGURE_DIR),
        _analysis_command(),
        *[_run_benchmark_command(run, hardware_config) for run in RANDOM_RUNS],
        _merge_split_command(RANDOM_RUNS, RANDOM_SPLIT_PATH),
        _merge_feature_command(RANDOM_RUNS, RANDOM_FEATURE_PATH),
        _aggregate_command(RANDOM_RUNS, RANDOM_TABLE_DIR, RANDOM_FIGURE_DIR),
        _profile_manifest_command(hardware_config),
        _acceptance_report_command(hardware_config),
        _rejection_suite_command(hardware_config, seeds, manuscript, bibliography),
        _audit_command(PRE_TIMING_AUDIT_PATH, seeds, require_timing=False, manuscript=manuscript, bibliography=bibliography),
        _audit_command(AUDIT_PATH, seeds, require_timing=True, manuscript=manuscript, bibliography=bibliography),
    ]


def _ensure_tier_e_host(audit_path: str = HARDWARE_AUDIT_PATH) -> None:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    classification = audit.get("classification", {})
    if not classification.get("tier_e_eligible"):
        raise SystemExit(f"Refusing Tier-E timed run on non-edge host: {classification.get('reasons', [])}")


def _read_profile(path: str) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    return json.loads(profile_path.read_text(encoding="utf-8"))


def write_combined_profile_manifest(hardware_config: str, out: str = MAIN_PROFILE_PATH) -> Path:
    hardware = load_yaml(hardware_config)
    component_paths = [run["profile"] for run in MAIN_RUNS]
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
        "timing": {
            "command_timeline": str(COMMAND_TIMELINE_PATH),
            "timing_events": str(TIMING_EVENTS_PATH),
            "timing_summary": str(TIMING_SUMMARY_PATH),
            "meaning": "wall-clock command and benchmark-stage timing, not energy measurement",
        },
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


def write_acceptance_evidence_report(out: str = ACCEPTANCE_REPORT_PATH) -> Path:
    summary = _read_csv(MAIN_SUMMARY_PATH)
    random_summary = _read_csv(RANDOM_SUMMARY_PATH)
    timing = _read_csv(str(TIMING_SUMMARY_PATH)) if TIMING_SUMMARY_PATH.exists() else pd.DataFrame()
    lines = [
        "# Timed Tier-E Core4 HGB+MLP Acceptance Evidence",
        "",
        "## Scope",
        "",
        f"- `{MAIN_RAW_PATH}`: {len(_read_csv(MAIN_RAW_PATH))} rows; expected 800.",
        f"- `{MAIN_SUMMARY_PATH}`: {len(summary)} rows; expected 80.",
        f"- `{RANDOM_RAW_PATH}`: {len(_read_csv(RANDOM_RAW_PATH))} rows; expected 100.",
        f"- `{RANDOM_SUMMARY_PATH}`: {len(random_summary)} rows; expected 10.",
        f"- Models: {', '.join(sorted(summary['model_id'].dropna().unique()))}.",
        "- Energy remains shared INA3221 `VDD_IN` module-power context, not model-isolated energy.",
        "- Timing sidecars report wall-clock execution burden and are not energy measurements.",
        "",
        "## Timing Summary",
        "",
    ]
    if timing.empty:
        lines.append("- Timing summary not yet available.")
    else:
        for _, row in timing.sort_values("stage").iterrows():
            lines.append(
                f"- {row['stage']}: events={int(row['event_count'])}, "
                f"total_s={float(row['elapsed_s_total']):.3f}, max_s={float(row['elapsed_s_max']):.3f}."
            )
    lines.extend(
        [
            "",
            "## Manuscript Boundary",
            "",
            "- Allowed: constructed admissibility negative controls were rejected by the checker.",
            "- Forbidden: do not describe the rejection suite as a test on independently submitted manuscripts.",
            "- Forbidden: do not use timing evidence as calibrated energy or model-isolated power evidence.",
        ]
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _base_audit_kwargs(seeds: list[int], manuscript: str, bibliography: str) -> dict[str, Any]:
    return {
        "raw_results_path": MAIN_RAW_PATH,
        "summary_results_path": MAIN_SUMMARY_PATH,
        "split_manifest_path": MAIN_SPLIT_PATH,
        "dataset_manifest_path": "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
        "feature_schema_path": MAIN_FEATURE_PATH,
        "hardware_audit_path": HARDWARE_AUDIT_PATH,
        "profile_manifest_path": MAIN_PROFILE_PATH,
        "manuscript_path": manuscript,
        "bibliography_path": bibliography,
        "require_tier_e": True,
        "expected_raw_rows": 800,
        "expected_summary_rows": 80,
        "expected_models": EXPECTED_MODELS,
        "expected_seeds": seeds,
        "expected_split_rows": 40,
        "expected_feature_schema_records": 40,
    }


def _copy_inputs(tmpdir: Path) -> dict[str, Path]:
    paths = {
        "raw_results_path": Path(MAIN_RAW_PATH),
        "summary_results_path": Path(MAIN_SUMMARY_PATH),
        "split_manifest_path": Path(MAIN_SPLIT_PATH),
        "feature_schema_path": Path(MAIN_FEATURE_PATH),
        "hardware_audit_path": Path(HARDWARE_AUDIT_PATH),
        "profile_manifest_path": Path(MAIN_PROFILE_PATH),
    }
    copied: dict[str, Path] = {}
    for key, source in paths.items():
        destination = tmpdir / source.name
        shutil.copy2(source, destination)
        copied[key] = destination
    return copied


def run_rejection_suite(
    seeds: list[int],
    manuscript: str = "jkics/jkics.tex",
    bibliography: str = "jkics/reference.bib",
    out: str = REJECTION_REPORT_PATH,
) -> Path:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="raise_ict_rejection_") as tmp:
        tmpdir = Path(tmp)
        copied = _copy_inputs(tmpdir)
        base = _base_audit_kwargs(seeds, manuscript, bibliography)

        cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

        def missing_split(kwargs: dict[str, Any]) -> None:
            kwargs["split_manifest_path"] = tmpdir / "missing_split.csv"

        def wrong_row_count(kwargs: dict[str, Any]) -> None:
            kwargs["expected_raw_rows"] = 801

        def missing_model_rows(kwargs: dict[str, Any]) -> None:
            raw = pd.read_csv(copied["raw_results_path"])
            raw = raw[~raw["model_id"].eq("hist_gradient_boosting")]
            path = tmpdir / "missing_model_raw.csv"
            raw.to_csv(path, index=False)
            kwargs["raw_results_path"] = path

        def non_jetson_hardware(kwargs: dict[str, Any]) -> None:
            raw = pd.read_csv(copied["raw_results_path"])
            raw["hardware_id"] = "cpu_proxy"
            path = tmpdir / "cpu_proxy_raw.csv"
            raw.to_csv(path, index=False)
            kwargs["raw_results_path"] = path

        def low_validity(kwargs: dict[str, Any]) -> None:
            raw = pd.read_csv(copied["raw_results_path"])
            mask = raw["threat_id"].eq(PRIMARY_THREAT)
            raw.loc[mask, "validity_rate"] = 0.1
            path = tmpdir / "low_validity_raw.csv"
            raw.to_csv(path, index=False)
            kwargs["raw_results_path"] = path

        def missing_profile_metadata(kwargs: dict[str, Any]) -> None:
            raw = pd.read_csv(copied["raw_results_path"])
            raw.loc[raw.index[0], "runtime"] = ""
            path = tmpdir / "missing_runtime_raw.csv"
            raw.to_csv(path, index=False)
            kwargs["raw_results_path"] = path

        cases.extend(
            [
                ("missing_split_manifest", missing_split),
                ("wrong_row_count", wrong_row_count),
                ("missing_model_rows", missing_model_rows),
                ("non_jetson_hardware", non_jetson_hardware),
                ("a1_validity_below_threshold", low_validity),
                ("missing_runtime_profile_metadata", missing_profile_metadata),
            ]
        )

        for case_id, mutate in cases:
            kwargs = {**base, **copied}
            mutate(kwargs)
            report = audit_completion(**kwargs)
            rejected = not bool(report["complete"])
            results.append(
                {
                    "case_id": case_id,
                    "rejected": rejected,
                    "incomplete_checks": [check["id"] for check in report["blocking_requirements"]],
                }
            )

    all_rejected = all(item["rejected"] for item in results)
    payload = {"schema_version": 1, "all_rejected": all_rejected, "cases": results}
    json_path = Path(REJECTION_JSON_PATH)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Constructed Admissibility-Rejection Suite",
        "",
        f"- All constructed invalid bundles rejected: `{str(all_rejected).lower()}`.",
        "- These are constructed negative controls, not tests on independently submitted manuscripts.",
        "",
        "| Case | Rejected | Blocking checks |",
        "|---|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | {str(item['rejected']).lower()} | "
            f"{', '.join(item['incomplete_checks'])} |"
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _event_row(
    event_id: int,
    stage: str,
    start_wall: float,
    end_wall: float,
    detail: str,
    output_path: str = "",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "stage": stage,
        "dataset": "",
        "split_id": "",
        "seed": "",
        "model_id": "",
        "threat_id": "",
        "start_iso": _iso(start_wall),
        "end_iso": _iso(end_wall),
        "elapsed_s": max(0.0, end_wall - start_wall),
        "rows": "",
        "output_path": output_path,
        "detail": detail,
    }


def _write_timing_sidecars(command_events: list[dict[str, object]]) -> None:
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    for run in [*MAIN_RUNS, *RANDOM_RUNS]:
        path = Path(run["timing"])
        if path.exists():
            timing_rows.extend(pd.read_csv(path).to_dict("records"))
    next_id = len(timing_rows) + 1
    for event in command_events:
        timing_rows.append(
            _event_row(
                next_id,
                str(event["stage"]),
                float(event["start_wall"]),
                float(event["end_wall"]),
                str(event.get("command", "")),
            )
        )
        next_id += 1

    fieldnames = [
        "event_id",
        "stage",
        "dataset",
        "split_id",
        "seed",
        "model_id",
        "threat_id",
        "start_iso",
        "end_iso",
        "elapsed_s",
        "rows",
        "output_path",
        "detail",
    ]
    with TIMING_EVENTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)

    frame = pd.DataFrame(timing_rows)
    summary = (
        frame.groupby("stage", dropna=False)["elapsed_s"]
        .agg(event_count="count", elapsed_s_total="sum", elapsed_s_mean="mean", elapsed_s_max="max")
        .reset_index()
        if not frame.empty
        else pd.DataFrame(columns=["stage", "event_count", "elapsed_s_total", "elapsed_s_mean", "elapsed_s_max"])
    )
    summary.to_csv(TIMING_SUMMARY_PATH, index=False)

    timeline = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "events": [
            {
                "index": index + 1,
                "stage": event["stage"],
                "command": event.get("command", ""),
                "start_iso": _iso(float(event["start_wall"])),
                "end_iso": _iso(float(event["end_wall"])),
                "elapsed_s": max(0.0, float(event["end_wall"]) - float(event["start_wall"])),
                "returncode": event.get("returncode", 0),
            }
            for index, event in enumerate(command_events)
        ],
    }
    COMMAND_TIMELINE_PATH.write_text(json.dumps(timeline, indent=2), encoding="utf-8")


def _timed_subprocess(command: list[str], stage: str, command_events: list[dict[str, object]]) -> None:
    print("+ " + " ".join(command), flush=True)
    start_wall = time.time()
    completed = subprocess.run(command, check=False)
    end_wall = time.time()
    command_events.append(
        {
            "stage": stage,
            "command": " ".join(command),
            "start_wall": start_wall,
            "end_wall": end_wall,
            "returncode": completed.returncode,
        }
    )
    _write_timing_sidecars(command_events)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _timed_call(stage: str, detail: str, command_events: list[dict[str, object]], func: Callable[[], Any]) -> Any:
    print(f"+ {detail}", flush=True)
    start_wall = time.time()
    try:
        return func()
    finally:
        end_wall = time.time()
        command_events.append(
            {
                "stage": stage,
                "command": detail,
                "start_wall": start_wall,
                "end_wall": end_wall,
                "returncode": 0,
            }
        )
        _write_timing_sidecars(command_events)


def run_experiment(
    hardware_config: str,
    seeds: list[int],
    manuscript: str = "jkics/jkics.tex",
    bibliography: str = "jkics/reference.bib",
) -> None:
    command_events: list[dict[str, object]] = []
    _write_generated_configs(seeds)
    _timed_subprocess(_command("scripts/validate_hardware_config.py", "--config", hardware_config), "hardware_validation", command_events)
    _timed_subprocess(_command("scripts/audit_hardware.py", "--out", HARDWARE_AUDIT_PATH), "hardware_audit", command_events)
    _ensure_tier_e_host()

    for run in MAIN_RUNS:
        _timed_subprocess(_run_benchmark_command(run, hardware_config), "run_benchmark", command_events)
    _timed_subprocess(_merge_split_command(MAIN_RUNS, MAIN_SPLIT_PATH), "artifact_merge", command_events)
    _timed_subprocess(_merge_feature_command(MAIN_RUNS, MAIN_FEATURE_PATH), "artifact_merge", command_events)
    _timed_subprocess(_aggregate_command(MAIN_RUNS, MAIN_TABLE_DIR, MAIN_FIGURE_DIR), "aggregate_results", command_events)
    _timed_subprocess(_analysis_command(), "analysis", command_events)

    for run in RANDOM_RUNS:
        _timed_subprocess(_run_benchmark_command(run, hardware_config), "run_benchmark", command_events)
    _timed_subprocess(_merge_split_command(RANDOM_RUNS, RANDOM_SPLIT_PATH), "artifact_merge", command_events)
    _timed_subprocess(_merge_feature_command(RANDOM_RUNS, RANDOM_FEATURE_PATH), "artifact_merge", command_events)
    _timed_subprocess(_aggregate_command(RANDOM_RUNS, RANDOM_TABLE_DIR, RANDOM_FIGURE_DIR), "aggregate_results", command_events)
    _timed_call("profile_manifest", "write combined profile manifest", command_events, lambda: write_combined_profile_manifest(hardware_config))
    _timed_call("acceptance_report", "write timed acceptance report", command_events, write_acceptance_evidence_report)
    _timed_call(
        "admissibility_rejection",
        "run constructed admissibility rejection suite",
        command_events,
        lambda: run_rejection_suite(seeds, manuscript=manuscript, bibliography=bibliography),
    )
    _timed_subprocess(
        _audit_command(
            PRE_TIMING_AUDIT_PATH,
            seeds,
            require_timing=False,
            manuscript=manuscript,
            bibliography=bibliography,
        ),
        "strict_audit",
        command_events,
    )
    _write_timing_sidecars(command_events)
    _timed_subprocess(
        _audit_command(
            AUDIT_PATH,
            seeds,
            require_timing=True,
            manuscript=manuscript,
            bibliography=bibliography,
        ),
        "timed_strict_audit",
        command_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--manuscript", default="jkics/jkics.tex")
    parser.add_argument("--bibliography", default="jkics/reference.bib")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-profile-manifest-only", action="store_true")
    parser.add_argument("--write-acceptance-report-only", action="store_true")
    parser.add_argument("--run-rejection-suite-only", action="store_true")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds]
    if args.write_profile_manifest_only:
        print(write_combined_profile_manifest(args.hardware_config))
        return
    if args.write_acceptance_report_only:
        print(write_acceptance_evidence_report())
        return
    if args.run_rejection_suite_only:
        print(run_rejection_suite(seeds, manuscript=args.manuscript, bibliography=args.bibliography))
        return
    if args.dry_run:
        payload = {
            "generated_configs": [str(CONFIG_DIR / name) for name in [
                "expanded.yaml",
                "cicids2017.yaml",
                "cse_cic_ids2018.yaml",
                "cicids2017_random_control.yaml",
                "cse_cic_ids2018_random_control.yaml",
            ]],
            "commands": build_commands(
                args.hardware_config,
                seeds,
                manuscript=args.manuscript,
                bibliography=args.bibliography,
            ),
            "notes": [
                "Generated configs set five models and the requested seed list.",
                f"Timing sidecars write under {TIMING_DIR}.",
                f"The strict timed audit writes {AUDIT_PATH}; it does not overwrite older evidence packages.",
            ],
        }
        print(json.dumps(payload, indent=2))
        return
    run_experiment(args.hardware_config, seeds, manuscript=args.manuscript, bibliography=args.bibliography)


if __name__ == "__main__":
    main()
