"""Completion audit for RAISE-ICT benchmark evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from raise_ict.schema import RESULT_FIELDS


EXPECTED_DATASETS = ["CICIDS2017", "CSE-CIC-IDS2018", "TON_IoT", "UNSW-NB15"]
EXPECTED_MODELS = ["extra_trees", "logistic_regression", "random_forest"]
EXPECTED_THREATS = ["a0_clean", "a1_constrained_feature", "a1_constrained_score_search", "a4_split_shift"]
EXPECTED_SEEDS = [0, 1, 2, 3, 4]


def _passed(check_id: str, evidence: str) -> dict[str, str]:
    return {"id": check_id, "status": "passed", "evidence": evidence}


def _incomplete(check_id: str, evidence: str) -> dict[str, str]:
    return {"id": check_id, "status": "incomplete", "evidence": evidence}


def _not_required(check_id: str, evidence: str) -> dict[str, str]:
    return {"id": check_id, "status": "not_required", "evidence": evidence}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check_equal(check_id: str, actual: Any, expected: Any, source: str) -> dict[str, str]:
    if actual == expected:
        return _passed(check_id, f"{source}: {actual}")
    return _incomplete(check_id, f"{source}: expected {expected}, found {actual}")


def _check_set(check_id: str, actual: list[Any], expected: list[Any], source: str) -> dict[str, str]:
    actual_sorted = sorted(actual)
    expected_sorted = sorted(expected)
    if actual_sorted == expected_sorted:
        return _passed(check_id, f"{source}: {actual_sorted}")
    return _incomplete(check_id, f"{source}: expected {expected_sorted}, found {actual_sorted}")


def _check_positive_count(check_id: str, actual: int, source: str) -> dict[str, str]:
    """Pass when a count is present without freezing a moving evidence total."""
    if actual > 0:
        return _passed(check_id, f"{source}: {actual}")
    return _incomplete(check_id, f"{source}: expected at least 1, found {actual}")


def _unique_values(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame.columns:
        return []
    return frame[column].dropna().unique().tolist()


def _unique_int_values(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return []
    return frame[column].dropna().astype(int).unique().tolist()


def _missing_columns(frame: pd.DataFrame, columns: set[str]) -> list[str]:
    """Return sorted missing columns without assuming the frame is complete."""
    return sorted(columns - set(frame.columns))


def _citation_checks(manuscript: str, bibliography: str) -> list[dict[str, str]]:
    if not manuscript or not bibliography:
        return [_incomplete("citations.present", "manuscript or bibliography file is missing")]
    cite_groups = re.findall(r"\\(?:supercite|cite)\{([^}]+)\}", manuscript)
    cite_keys = sorted({key.strip() for group in cite_groups for key in group.split(",") if key.strip()})
    bib_keys = sorted(set(re.findall(r"@\w+\{([^,]+),", bibliography)))
    missing = sorted(set(cite_keys) - set(bib_keys))
    unused = sorted(set(bib_keys) - set(cite_keys))
    placeholders = re.findall(r"citation_needed|PLACEHOLDER|TODO: Verify", manuscript + "\n" + bibliography, flags=re.I)
    checks = [
        _check_positive_count("citations.key_count", len(cite_keys), "manuscript cite keys"),
        _check_positive_count("citations.bib_count", len(bib_keys), "BibTeX entries"),
    ]
    checks.append(
        _passed("citations.no_missing_or_unused", "no missing or unused BibTeX keys")
        if not missing and not unused
        else _incomplete("citations.no_missing_or_unused", f"missing={missing}, unused={unused}")
    )
    checks.append(
        _passed("citations.no_placeholders", "no placeholder citation markers")
        if not placeholders
        else _incomplete("citations.no_placeholders", f"placeholder markers={len(placeholders)}")
    )
    return checks


def _profile_manifest_checks(
    profile_manifest: Any,
    profile_manifest_path: Path,
    hardware_ids: list[str],
    require_tier_e: bool,
) -> list[dict[str, str]]:
    if not require_tier_e:
        if profile_manifest is None:
            return [_not_required("tier_e.profile_manifest_present", str(profile_manifest_path))]
        return [_passed("tier_e.profile_manifest_present", str(profile_manifest_path))]
    if profile_manifest is None:
        return [_incomplete("tier_e.profile_manifest_present", str(profile_manifest_path))]
    checks = [_passed("tier_e.profile_manifest_present", str(profile_manifest_path))]
    hardware = profile_manifest.get("hardware", {}) if isinstance(profile_manifest, dict) else {}
    profile = profile_manifest.get("profile", {}) if isinstance(profile_manifest, dict) else {}
    measurement_mode = str(hardware.get("measurement_mode", "")).lower()
    energy_source = str(hardware.get("energy_source", "")).lower()
    profile_hardware_id = str(profile.get("hardware_id", hardware.get("hardware_id", "")))
    energy_per_flow = float(profile.get("energy_per_flow_j", 0.0) or 0.0)
    measured_mode = any(token in measurement_mode for token in ["measured", "meter", "sensor"])
    measured_source = bool(energy_source) and energy_source not in {"proxy", "none", "unknown"}
    checks.append(
        _passed(
            "tier_e.profile_manifest_measured_mode",
            f"measurement_mode={measurement_mode}, energy_source={energy_source}",
        )
        if measured_mode and measured_source
        else _incomplete(
            "tier_e.profile_manifest_measured_mode",
            f"measurement_mode={measurement_mode}, energy_source={energy_source}",
        )
    )
    checks.append(
        _passed("tier_e.profile_manifest_hardware_match", f"profile_hardware_id={profile_hardware_id}")
        if profile_hardware_id in hardware_ids and profile_hardware_id != "cpu_proxy"
        else _incomplete(
            "tier_e.profile_manifest_hardware_match",
            f"profile_hardware_id={profile_hardware_id}, result_hardware_ids={hardware_ids}",
        )
    )
    checks.append(
        _passed("tier_e.profile_manifest_energy", f"energy_per_flow_j={energy_per_flow}")
        if energy_per_flow > 0.0
        else _incomplete("tier_e.profile_manifest_energy", f"energy_per_flow_j={energy_per_flow}")
    )
    return checks


def _claim_boundary_terms(require_tier_e: bool) -> tuple[list[str], str]:
    """Return manuscript boundary terms for the current evidence tier."""
    if require_tier_e:
        return (
            [
                "module-power",
                "not calibrated wall-power",
                "feature-space validity does not prove packet-level",
            ],
            "Tier-E module-power and packet-level limits are explicit",
        )
    return (
        [
            "CPU-proxy",
            "not treated as Tier-E evidence",
            "feature-space validity does not prove packet-level",
        ],
        "CPU-proxy, Tier-E, and packet-level limits are explicit",
    )


def audit_completion(
    raw_results_path: str | Path = "results/tables/tier_p_core4/table_raw_results.csv",
    summary_results_path: str | Path = "results/tables/tier_p_core4/table_main_results.csv",
    split_manifest_path: str | Path = "manifests/splits/tier_p_core4_split_manifest.csv",
    dataset_manifest_path: str | Path = "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
    feature_schema_path: str | Path = "manifests/feature_schemas/tier_p_core4_feature_schema.json",
    hardware_audit_path: str | Path = "manifests/hardware/tier_e_hardware_audit.json",
    profile_manifest_path: str | Path = "manifests/hardware/tier_e_profile_manifest.json",
    manuscript_path: str | Path = "raise_ict_manuscript_scaffold.tex",
    bibliography_path: str | Path = "references.bib",
    require_tier_e: bool = False,
    require_full_scale_cse: bool = False,
    expected_raw_rows: int = 240,
    expected_summary_rows: int = 48,
    expected_models: list[str] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable audit of benchmark completion evidence."""
    expected_model_ids = expected_models or EXPECTED_MODELS
    raw_path = Path(raw_results_path)
    summary_path = Path(summary_results_path)
    split_path = Path(split_manifest_path)
    dataset_path = Path(dataset_manifest_path)
    feature_path = Path(feature_schema_path)
    hardware_path = Path(hardware_audit_path)
    profile_path = Path(profile_manifest_path)
    manuscript_file = Path(manuscript_path)
    bibliography_file = Path(bibliography_path)

    raw = _read_csv(raw_path)
    summary = _read_csv(summary_path)
    splits = _read_csv(split_path)
    dataset_manifest = _read_json(dataset_path)
    feature_schema = _read_json(feature_path)
    hardware_audit = _read_json(hardware_path)
    profile_manifest = _read_json(profile_path)
    manuscript = manuscript_file.read_text(encoding="utf-8") if manuscript_file.exists() else ""
    bibliography = bibliography_file.read_text(encoding="utf-8") if bibliography_file.exists() else ""

    checks: list[dict[str, str]] = []
    checks.append(
        _passed("artifacts.raw_present", str(raw_path))
        if raw is not None
        else _incomplete("artifacts.raw_present", str(raw_path))
    )
    checks.append(
        _passed("artifacts.summary_present", str(summary_path))
        if summary is not None
        else _incomplete("artifacts.summary_present", str(summary_path))
    )
    checks.append(
        _passed("artifacts.split_manifest_present", str(split_path))
        if splits is not None
        else _incomplete("artifacts.split_manifest_present", str(split_path))
    )
    checks.append(
        _passed("artifacts.dataset_manifest_present", str(dataset_path))
        if dataset_manifest is not None
        else _incomplete("artifacts.dataset_manifest_present", str(dataset_path))
    )
    checks.append(
        _passed("artifacts.feature_schema_present", str(feature_path))
        if feature_schema is not None
        else _incomplete("artifacts.feature_schema_present", str(feature_path))
    )

    hardware_ids: list[str] = []
    if raw is not None:
        checks.append(_check_equal("core4.raw_rows", len(raw), expected_raw_rows, str(raw_path)))
        checks.append(_check_set("core4.datasets", _unique_values(raw, "dataset"), EXPECTED_DATASETS, str(raw_path)))
        checks.append(_check_set("core4.models", _unique_values(raw, "model_id"), expected_model_ids, str(raw_path)))
        checks.append(_check_set("core4.threats", _unique_values(raw, "threat_id"), EXPECTED_THREATS, str(raw_path)))
        checks.append(_check_set("core4.seeds", _unique_int_values(raw, "seed"), EXPECTED_SEEDS, str(raw_path)))
        missing_fields = [field for field in RESULT_FIELDS if field not in raw.columns]
        checks.append(
            _passed("schema.result_fields", "all standard result fields are present")
            if not missing_fields
            else _incomplete("schema.result_fields", f"missing fields={missing_fields}")
        )
        validity_columns = {
            "valid_count",
            "invalid_count",
            "budget_pass_rate",
            "bounds_pass_rate",
            "immutable_pass_rate",
            "relation_pass_rate",
        }
        missing_validity_columns = _missing_columns(raw, validity_columns)
        if missing_validity_columns:
            checks.append(
                _incomplete("attacks.validity_counts_present", f"missing fields={missing_validity_columns}")
            )
        else:
            total_counts = raw["valid_count"].fillna(0) + raw["invalid_count"].fillna(0)
            nonnegative_counts = bool((raw[["valid_count", "invalid_count"]].fillna(-1) >= 0).all().all())
            checks.append(
                _passed(
                    "attacks.validity_counts_present",
                    f"minimum valid+invalid count={int(total_counts.min())}",
                )
                if nonnegative_counts and int(total_counts.min()) > 0
                else _incomplete(
                    "attacks.validity_counts_present",
                    f"minimum valid+invalid count={int(total_counts.min()) if len(total_counts) else 0}",
                )
            )
        min_validity = float(raw["validity_rate"].min()) if "validity_rate" in raw else -1.0
        checks.append(
            _passed("attacks.validity_threshold", f"minimum validity_rate={min_validity:.3f}")
            if min_validity >= 0.95
            else _incomplete("attacks.validity_threshold", f"minimum validity_rate={min_validity:.3f}")
        )
        hardware_ids = sorted(_unique_values(raw, "hardware_id"))
        profile_columns = {"thread_count", "batch_size", "runtime", "measurement_mode", "energy_source"}
        missing_profile_columns = _missing_columns(raw, profile_columns)
        if missing_profile_columns:
            checks.append(
                _incomplete("profiling.profile_metadata_present", f"missing fields={missing_profile_columns}")
            )
        else:
            populated = (
                raw["thread_count"].fillna(0).astype(int).ge(1).all()
                and raw["batch_size"].fillna(0).astype(int).ge(1).all()
                and raw["runtime"].fillna("").astype(str).str.len().gt(0).all()
                and raw["measurement_mode"].fillna("").astype(str).str.len().gt(0).all()
                and raw["energy_source"].fillna("").astype(str).str.len().gt(0).all()
            )
            checks.append(
                _passed("profiling.profile_metadata_present", "thread_count, batch_size, runtime, mode, source present")
                if populated
                else _incomplete("profiling.profile_metadata_present", "one or more profile metadata fields are empty")
            )
        if require_tier_e:
            energy_min = float(raw["energy_per_flow_j"].min()) if "energy_per_flow_j" in raw else 0.0
            physical_ids = [hardware_id for hardware_id in hardware_ids if hardware_id != "cpu_proxy"]
            checks.append(
                _passed("profiling.physical_edge_results_required", f"hardware_id values={hardware_ids}")
                if physical_ids and len(physical_ids) == len(hardware_ids)
                else _incomplete("profiling.physical_edge_results_required", f"hardware_id values={hardware_ids}")
            )
            checks.append(
                _passed("profiling.measured_energy_required", f"minimum energy_per_flow_j={energy_min}")
                if energy_min > 0.0
                else _incomplete("profiling.measured_energy_required", f"minimum energy_per_flow_j={energy_min}")
            )
            if missing_profile_columns:
                checks.append(
                    _incomplete("profiling.measured_energy_metadata_required", f"missing fields={missing_profile_columns}")
                )
            else:
                modes = raw["measurement_mode"].fillna("").astype(str).str.lower()
                sources = raw["energy_source"].fillna("").astype(str).str.lower()
                measured_modes = modes.str.contains("measured|meter|sensor", regex=True).all()
                measured_sources = (~sources.isin({"", "proxy", "none", "unknown"})).all()
                checks.append(
                    _passed(
                        "profiling.measured_energy_metadata_required",
                        f"measurement_mode={sorted(modes.unique())}, energy_source={sorted(sources.unique())}",
                    )
                    if measured_modes and measured_sources
                    else _incomplete(
                        "profiling.measured_energy_metadata_required",
                        f"measurement_mode={sorted(modes.unique())}, energy_source={sorted(sources.unique())}",
                    )
                )
        else:
            checks.append(
                _passed("profiling.cpu_proxy_declared", f"hardware_id values={hardware_ids}")
                if hardware_ids == ["cpu_proxy"]
                else _incomplete("profiling.cpu_proxy_declared", f"hardware_id values={hardware_ids}")
            )
    if summary is not None:
        checks.append(_check_equal("core4.summary_rows", len(summary), expected_summary_rows, str(summary_path)))
    if splits is not None:
        checks.append(_check_equal("core4.split_rows", len(splits), 20, str(split_path)))
        split_columns = {"dataset", "test_rows"}
        missing_split_columns = sorted(split_columns - set(splits.columns))
        repro_split_columns = {
            "raw_files",
            "raw_rows",
            "dataset_config_sha256",
            "preprocessing_state_sha256",
            "config_path",
            "software_version",
        }
        missing_repro_split_columns = _missing_columns(splits, repro_split_columns)
        if missing_repro_split_columns:
            checks.append(
                _incomplete(
                    "splits.reproducibility_columns",
                    f"missing fields={missing_repro_split_columns}",
                )
            )
        else:
            reproducible = (
                splits["raw_rows"].fillna(0).astype(int).gt(0).all()
                and splits["preprocessing_state_sha256"].fillna("").astype(str).str.len().gt(0).all()
                and splits["config_path"].fillna("").astype(str).str.len().gt(0).all()
                and splits["software_version"].fillna("").astype(str).str.len().gt(0).all()
            )
            checks.append(
                _passed("splits.reproducibility_columns", "raw files, row counts, config, preprocessing hash, version present")
                if reproducible
                else _incomplete("splits.reproducibility_columns", "one or more reproducibility fields are empty")
            )
        cse = splits[splits["dataset"].eq("CSE-CIC-IDS2018")] if not missing_split_columns else pd.DataFrame()
        sampled_cse = not cse.empty and int(cse["test_rows"].max()) == 5000
        if require_full_scale_cse:
            checks.append(
                _incomplete("scale.full_cse_required", f"missing split columns={missing_split_columns}")
                if missing_split_columns
                else (
                    _incomplete(
                        "scale.full_cse_required",
                        "CSE-CIC-IDS2018 split has 5,000 test rows per seed, so it is bounded sampled evidence",
                    )
                    if sampled_cse
                    else _passed("scale.full_cse_required", "CSE-CIC-IDS2018 split is not the bounded sampled split")
                )
            )
        else:
            checks.append(
                _not_required(
                    "scale.full_cse_required",
                    "full-source CSE-CIC-IDS2018 is not required for the current Tier-P manuscript claim",
                )
            )
    if dataset_manifest is not None:
        checks.append(_check_equal("core4.dataset_manifest_records", len(dataset_manifest), 15, str(dataset_path)))
    if feature_schema is not None:
        checks.append(_check_equal("core4.feature_schema_records", len(feature_schema), 20, str(feature_path)))
        if isinstance(feature_schema, list) and feature_schema:
            feature_columns = set().union(*(record.keys() for record in feature_schema if isinstance(record, dict)))
            missing_feature_schema_columns = sorted(
                {"preprocessing_state_sha256", "config_path", "software_version"} - feature_columns
            )
            checks.append(
                _passed("features.reproducibility_fields", "preprocessing hash, config path, software version present")
                if not missing_feature_schema_columns
                else _incomplete(
                    "features.reproducibility_fields",
                    f"missing fields={missing_feature_schema_columns}",
                )
            )
        else:
            checks.append(_incomplete("features.reproducibility_fields", "feature schema is empty or not a list"))

    if hardware_audit is None:
        checks.append(_incomplete("tier_e.audit_present", str(hardware_path)))
        tier_e_eligible = False
    else:
        checks.append(_passed("tier_e.audit_present", str(hardware_path)))
        tier_e_eligible = bool(hardware_audit.get("classification", {}).get("tier_e_eligible"))
        evidence = json.dumps(hardware_audit.get("classification", {}), sort_keys=True)
        if require_tier_e:
            checks.append(
                _passed("tier_e.physical_edge_required", evidence)
                if tier_e_eligible
                else _incomplete("tier_e.physical_edge_required", evidence)
            )
        else:
            checks.append(
                _passed("tier_e.physical_edge_available", evidence)
                if tier_e_eligible
                else _not_required("tier_e.physical_edge_available", evidence)
            )
    checks.extend(_profile_manifest_checks(profile_manifest, profile_path, hardware_ids, require_tier_e))

    if manuscript:
        boundary_terms, boundary_evidence = _claim_boundary_terms(require_tier_e)
        missing_terms = [term for term in boundary_terms if term not in manuscript]
        checks.append(
            _passed("manuscript.claim_boundaries", boundary_evidence)
            if not missing_terms
            else _incomplete("manuscript.claim_boundaries", f"missing terms={missing_terms}")
        )
    else:
        checks.append(_incomplete("manuscript.present", str(manuscript_file)))
    checks.extend(_citation_checks(manuscript, bibliography))

    required_statuses = {"passed"}
    required_checks = [check for check in checks if check["status"] != "not_required"]
    complete = all(check["status"] in required_statuses for check in required_checks)
    incomplete = [check for check in checks if check["status"] == "incomplete"]
    return {
        "schema_version": 1,
        "require_tier_e": require_tier_e,
        "require_full_scale_cse": require_full_scale_cse,
        "expected_raw_rows": expected_raw_rows,
        "expected_summary_rows": expected_summary_rows,
        "expected_models": expected_model_ids,
        "complete": complete,
        "summary": {
            "passed": sum(check["status"] == "passed" for check in checks),
            "not_required": sum(check["status"] == "not_required" for check in checks),
            "incomplete": len(incomplete),
        },
        "checks": checks,
        "blocking_requirements": incomplete,
    }
