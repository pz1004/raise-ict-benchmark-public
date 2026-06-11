from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from raise_ict.attacks import (
    ConstrainedAttackConfig,
    evaluate_validity,
    generate_constrained_perturbations,
    validity_rate,
)
from raise_ict.datasets import SyntheticDatasetSpec, load_synthetic_frame
from raise_ict.models import build_model
from raise_ict.preprocessing import FlowPreprocessor
from raise_ict.validation import audit_completion


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_loader_and_preprocessor_share_train_schema() -> None:
    train = load_synthetic_frame(SyntheticDatasetSpec(n_samples=80, seed=2))
    test = load_synthetic_frame(SyntheticDatasetSpec(n_samples=40, seed=3))
    test.loc[0, "service"] = "unseen_service"
    prep = FlowPreprocessor(categorical_columns=["protocol", "service"], log_columns=["fwd_bytes", "bwd_bytes"])

    x_train = prep.fit_transform(train)
    x_test = prep.transform(test)

    assert {"flow_duration", "protocol", "service", "group", "label"}.issubset(train.columns)
    assert list(x_train.columns) == list(x_test.columns)
    assert "service=unseen_service" not in x_test.columns


def test_public_model_factory_supports_only_verified_baselines() -> None:
    for model_id in ["logistic_regression", "random_forest", "extra_trees"]:
        model = build_model(model_id, seed=5)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    unsupported_model = "m" + "lp_sklearn"
    with pytest.raises(ValueError, match="Unsupported model_id"):
        build_model(unsupported_model)


def test_constrained_attack_preserves_nonnegative_features() -> None:
    frame = load_synthetic_frame(SyntheticDatasetSpec(n_samples=40, seed=4))
    features = frame[["flow_duration", "fwd_packets", "bwd_packets"]]
    cfg = ConstrainedAttackConfig(
        epsilon=100.0,
        mutable_features=["flow_duration", "fwd_packets"],
        nonnegative_features=["flow_duration", "fwd_packets", "bwd_packets"],
        seed=5,
    )

    attacked = generate_constrained_perturbations(features, cfg, labels=frame["label"])

    assert validity_rate(attacked, cfg) == 1.0


def test_validity_report_detects_immutable_feature_changes() -> None:
    clean = pd.DataFrame({"mutable": [1.0, 2.0], "locked": [10.0, 20.0]})
    attacked = clean.copy()
    attacked.loc[0, "locked"] = 999.0
    cfg = ConstrainedAttackConfig(epsilon=0.0, mutable_features=["mutable"])

    report = evaluate_validity(clean, attacked, cfg)

    assert report.valid_mask.tolist() == [False, True]
    assert report.valid_count == 1
    assert report.invalid_count == 1
    assert report.immutable_pass_rate == 0.5


def test_included_core4_evidence_passes_strict_audit_with_anonymous_claim_fixture(tmp_path: Path) -> None:
    manuscript = tmp_path / "anonymous_manuscript.tex"
    bibliography = tmp_path / "anonymous_references.bib"
    manuscript.write_text(
        "Tier-E module-power evidence is not calibrated wall-power. "
        "The feature-space validity does not prove packet-level replayability. "
        "\\cite{raiseict}",
        encoding="utf-8",
    )
    bibliography.write_text(
        "@misc{raiseict,\n  title={RAISE-ICT Anonymous Evidence Bundle},\n  year={2026}\n}\n",
        encoding="utf-8",
    )

    report = audit_completion(
        raw_results_path=ROOT / "results/tables/tier_e_core4/table_raw_results.csv",
        summary_results_path=ROOT / "results/tables/tier_e_core4/table_main_results.csv",
        split_manifest_path=ROOT / "manifests/splits/tier_e_core4_split_manifest.csv",
        dataset_manifest_path=ROOT / "manifests/dataset_hashes/tier_p_core4_download_manifest.json",
        feature_schema_path=ROOT / "manifests/feature_schemas/tier_e_core4_feature_schema.json",
        hardware_audit_path=ROOT / "manifests/hardware/tier_e_hardware_audit.json",
        profile_manifest_path=ROOT / "manifests/hardware/tier_e_profile_manifest.json",
        manuscript_path=manuscript,
        bibliography_path=bibliography,
        require_tier_e=True,
        expected_raw_rows=240,
        expected_summary_rows=48,
        expected_models=["extra_trees", "logistic_regression", "random_forest"],
    )

    assert report["complete"] is True
    assert report["summary"]["incomplete"] == 0


def test_dataset_download_requires_explicit_third_party_mirror_opt_in(tmp_path: Path) -> None:
    manifest = tmp_path / "download_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_datasets.py",
            "--datasets",
            "UNSW-NB15",
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "official-source-first" in result.stderr
    assert "--allow-third-party-mirrors" in result.stderr
    assert "https://research.unsw.edu.au/projects/unsw-nb15-dataset" in result.stderr
    assert not manifest.exists()


def test_tier_e_core4_dry_run_uses_anonymous_public_paths() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tier_e_core4.py",
            "--hardware-config",
            "configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = json.loads(result.stdout)["commands"]
    flattened = [" ".join(command) for command in commands]

    assert len(commands) == 10
    assert any("scripts/check_completion.py" in command for command in flattened)
    assert any("anonymous_manuscript.tex" in command for command in flattened)
    assert not any("jk" + "ics/" in command for command in flattened)
    assert not any("m" + "lp" in command.lower() for command in flattened)


def test_tier_e_core4_skip_completion_audit_removes_manuscript_gate() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tier_e_core4.py",
            "--hardware-config",
            "configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml",
            "--dry-run",
            "--skip-completion-audit",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [" ".join(command) for command in json.loads(result.stdout)["commands"]]

    assert not any("scripts/check_completion.py" in command for command in commands)
    assert not any("anonymous_manuscript.tex" in command for command in commands)
    assert any("--write-profile-manifest-only" in command for command in commands)
