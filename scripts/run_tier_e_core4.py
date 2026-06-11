#!/usr/bin/env python
"""Run or print the Tier-E Core4 benchmark orchestration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402


Command = list[str]


class EdgeRun(TypedDict):
    """Tier-E component run paths used to build the Core4 command graph."""

    name: str
    config: str
    raw: str
    split: str
    features: str
    profile: str


CORE4_SPLIT_MANIFEST = "manifests/splits/tier_e_core4_split_manifest.csv"
CORE4_FEATURE_SCHEMA = "manifests/feature_schemas/tier_e_core4_feature_schema.json"
CORE4_RESULT_TABLE_DIR = "results/tables/tier_e_core4"
CORE4_FIGURE_DIR = "results/figures/tier_e_core4"
CORE4_HARDWARE_AUDIT = "manifests/hardware/tier_e_hardware_audit.json"
CORE4_PROFILE_MANIFEST = "manifests/hardware/tier_e_profile_manifest.json"
CORE4_COMPLETION_AUDIT = "manifests/completion/benchmark_completion_audit_strict_tier_e.json"
DATASET_MANIFEST = "manifests/dataset_hashes/tier_p_core4_download_manifest.json"


EDGE_RUNS: list[EdgeRun] = [
    {
        "name": "expanded",
        "config": "configs/experiments/tier_p_expanded.yaml",
        "raw": "results/raw/tier_e_expanded",
        "split": "manifests/splits/tier_e_expanded_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_expanded_feature_schema.json",
        "profile": "manifests/hardware/tier_e_expanded_profile_manifest.json",
    },
    {
        "name": "cicids2017",
        "config": "configs/experiments/tier_p_cicids2017.yaml",
        "raw": "results/raw/tier_e_cicids2017",
        "split": "manifests/splits/tier_e_cicids2017_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cicids2017_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cicids2017_profile_manifest.json",
    },
    {
        "name": "cse_cic_ids2018",
        "config": "configs/experiments/tier_p_cse_cic_ids2018.yaml",
        "raw": "results/raw/tier_e_cse_cic_ids2018",
        "split": "manifests/splits/tier_e_cse_cic_ids2018_split_manifest.csv",
        "features": "manifests/feature_schemas/tier_e_cse_cic_ids2018_feature_schema.json",
        "profile": "manifests/hardware/tier_e_cse_cic_ids2018_profile_manifest.json",
    },
]


def _command(*parts: str) -> Command:
    return [sys.executable, *parts]


def _profile_manifest_command(hardware_config: str) -> Command:
    return _command(
        "scripts/run_tier_e_core4.py",
        "--hardware-config",
        hardware_config,
        "--write-profile-manifest-only",
    )


def _completion_audit_command(manuscript: str, bibliography: str, strict: bool) -> Command:
    return _command(
        "scripts/check_completion.py",
        "--require-tier-e",
        "--raw-results",
        f"{CORE4_RESULT_TABLE_DIR}/table_raw_results.csv",
        "--summary-results",
        f"{CORE4_RESULT_TABLE_DIR}/table_main_results.csv",
        "--split-manifest",
        CORE4_SPLIT_MANIFEST,
        "--dataset-manifest",
        DATASET_MANIFEST,
        "--feature-schema",
        CORE4_FEATURE_SCHEMA,
        "--hardware-audit",
        CORE4_HARDWARE_AUDIT,
        "--profile-manifest",
        CORE4_PROFILE_MANIFEST,
        "--manuscript",
        manuscript,
        "--bibliography",
        bibliography,
        "--out",
        CORE4_COMPLETION_AUDIT,
        *(["--strict"] if strict else []),
    )


def build_commands(
    hardware_config: str,
    strict: bool = True,
    manuscript: str = "anonymous_manuscript.tex",
    bibliography: str = "anonymous_references.bib",
    include_completion_audit: bool = True,
) -> list[Command]:
    """Build the Tier-E Core4 command graph without executing it."""
    raw_dirs = [run["raw"] for run in EDGE_RUNS]
    split_files = [run["split"] for run in EDGE_RUNS]
    feature_files = [run["features"] for run in EDGE_RUNS]
    commands: list[Command] = [
        _command("scripts/validate_hardware_config.py", "--config", hardware_config),
        _command("scripts/audit_hardware.py", "--out", CORE4_HARDWARE_AUDIT),
    ]
    for run in EDGE_RUNS:
        commands.append(
            _command(
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
        )
    commands.extend(
        [
            _command(
                "scripts/merge_artifacts.py",
                *split_files,
                "--out",
                CORE4_SPLIT_MANIFEST,
            ),
            _command(
                "scripts/merge_artifacts.py",
                *feature_files,
                "--out",
                CORE4_FEATURE_SCHEMA,
            ),
            _command(
                "scripts/aggregate_results.py",
                "--results",
                *raw_dirs,
                "--out",
                CORE4_RESULT_TABLE_DIR,
                "--figures",
                CORE4_FIGURE_DIR,
            ),
        ]
    )
    if include_completion_audit:
        commands.append(_completion_audit_command(manuscript, bibliography, strict))
    return commands


def write_combined_profile_manifest(
    hardware_config: str,
    out: str = CORE4_PROFILE_MANIFEST,
) -> Path:
    """Merge per-run Tier-E profile manifests into the strict audit manifest."""
    hardware = load_yaml(hardware_config)
    profiles: list[dict[str, Any]] = []
    for run in EDGE_RUNS:
        path = Path(run["profile"])
        if not path.exists():
            raise FileNotFoundError(path)
        profiles.append(json.loads(path.read_text(encoding="utf-8")))
    energy_values = [float(profile.get("profile", {}).get("energy_per_flow_j", 0.0) or 0.0) for profile in profiles]
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
        "component_manifests": [run["profile"] for run in EDGE_RUNS],
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def run_commands(commands: list[Command]) -> None:
    """Run generated commands sequentially with subprocess failure propagation."""
    for command in commands:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--skip-completion-audit", action="store_true")
    parser.add_argument("--manuscript", default="anonymous_manuscript.tex")
    parser.add_argument("--bibliography", default="anonymous_references.bib")
    parser.add_argument("--write-profile-manifest-only", action="store_true")
    args = parser.parse_args()
    commands = build_commands(
        args.hardware_config,
        strict=not args.no_strict,
        manuscript=args.manuscript,
        bibliography=args.bibliography,
        include_completion_audit=not args.skip_completion_audit,
    )
    if args.write_profile_manifest_only:
        print(write_combined_profile_manifest(args.hardware_config))
        return
    profile_manifest_command = _profile_manifest_command(args.hardware_config)
    if args.dry_run:
        if args.skip_completion_audit:
            dry_run_commands = commands + [profile_manifest_command]
        else:
            dry_run_commands = commands[:-1] + [profile_manifest_command, commands[-1]]
        print(json.dumps({"commands": dry_run_commands}, indent=2))
        return
    if args.skip_completion_audit:
        run_commands(commands)
        print(write_combined_profile_manifest(args.hardware_config))
        return
    run_commands(commands[:-1])
    print(write_combined_profile_manifest(args.hardware_config))
    run_commands(commands[-1:])


if __name__ == "__main__":
    main()
