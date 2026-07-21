from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _entry(status: str, value: object = None) -> dict[str, object]:
    return {
        "status": status,
        "value": value,
        "evidence_anchor": "constructed-fixture:p1",
        "note": "constructed software fixture",
    }


def _row(
    row_id: str,
    model: str,
    *,
    dataset: str = "dataset-a",
    preprocessing_status: str = "resolved",
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "source_id": "constructed-fixture",
        "source_url": "https://example.test/constructed-fixture",
        "source_locator": f"fixture:{row_id}",
        "fields": {
            "model_id": _entry("resolved", model),
            "reported_value": _entry("resolved", 0.8),
            "dataset_id": _entry("resolved", dataset),
            "preprocessing": _entry(
                preprocessing_status,
                "standardized" if preprocessing_status == "resolved" else None,
            ),
            "seed": _entry("not_applicable"),
            "metric_name": _entry("resolved", "f1_score"),
            "metric_unit": _entry("resolved", "fraction"),
            "metric_direction": _entry("resolved", "maximize"),
            "citation_location": _entry("resolved", f"fixture:{row_id}"),
        },
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    contexts = root / "contexts.yaml"
    rows = root / "rows.yaml"
    pairs = root / "pairs.yaml"
    contexts.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "protocol_id": "constructed-pairwise-v1",
                "ablation_groups": {
                    "data": ["dataset_id", "seed"],
                    "pipeline": ["preprocessing"],
                    "metric": ["metric_name", "metric_unit", "metric_direction"],
                    "provenance": ["citation_location"],
                },
                "contexts": [
                    {
                        "context_id": "clean_model_ordering",
                        "description": "constructed software fixture",
                        "comparison_axis": "model_id",
                        "outcome_field": "reported_value",
                        "allowed_metrics": ["f1_score"],
                        "metric_identity_fields": [
                            "metric_name",
                            "metric_unit",
                            "metric_direction",
                        ],
                        "required_invariants": [
                            "dataset_id",
                            "preprocessing",
                            "seed",
                            "metric_name",
                            "metric_unit",
                            "metric_direction",
                        ],
                        "required_resolved": ["citation_location"],
                        "optional_fields": [],
                        "allowed_not_applicable": ["seed"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "protocol_id": "constructed-pairwise-v1",
                "rows": [
                    _row("row-a", "model-a"),
                    _row("row-b", "model-b"),
                    _row(
                        "row-c",
                        "model-c",
                        dataset="dataset-b",
                        preprocessing_status="unresolved",
                    ),
                    _row("row-d", "model-d", preprocessing_status="unresolved"),
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pairs.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "protocol_id": "constructed-pairwise-v1",
                "pairs": [
                    {
                        "pair_id": "defined-pair",
                        "context_id": "clean_model_ordering",
                        "row_a": "row-a",
                        "row_b": "row-b",
                        "metric": "f1_score",
                        "claim": "constructed defined comparison",
                    },
                    {
                        "pair_id": "mismatch-pair",
                        "context_id": "clean_model_ordering",
                        "row_a": "row-a",
                        "row_b": "row-c",
                        "metric": "f1_score",
                        "claim": "constructed invariant conflict",
                    },
                    {
                        "pair_id": "insufficient-pair",
                        "context_id": "clean_model_ordering",
                        "row_a": "row-a",
                        "row_b": "row-d",
                        "metric": "f1_score",
                        "claim": "constructed unresolved evidence",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return contexts, rows, pairs


def _run(contexts: Path, rows: Path, pairs: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_pairwise_admission.py",
            "--contexts",
            str(contexts),
            "--rows",
            str(rows),
            "--pairs",
            str(pairs),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_constructed_pairs_cover_all_three_decisions(tmp_path: Path) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    completed = _run(contexts, rows, pairs, out_dir)
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "decision_counts": {
            "context_mismatch": 1,
            "defined": 1,
            "insufficient_evidence": 1,
        },
        "high_confidence_diversity_gate": True,
        "pair_count": 3,
    }
    decisions = json.loads((out_dir / "pair_decisions.json").read_text(encoding="utf-8"))
    by_id = {row["pair_id"]: row for row in decisions}
    assert by_id["defined-pair"]["field_states"]["seed"] == "not_applicable"
    assert by_id["defined-pair"]["metric"] == "f1_score"
    assert by_id["mismatch-pair"]["mismatched_fields"] == ["dataset_id"]
    assert by_id["mismatch-pair"]["unresolved_fields"] == ["preprocessing"]
    assert by_id["insufficient-pair"]["decision"] == "insufficient_evidence"


@pytest.mark.parametrize(
    ("case_id", "expected_error"),
    [
        ("empty_identity", "metric_identity_fields must not be empty"),
        ("missing_name", "metric_identity_fields must include 'metric_name'"),
        ("empty_metrics", "allowed_metrics must not be empty"),
        ("metric_not_applicable", "cannot allow not_applicable for metric identity fields"),
    ],
)
def test_malformed_metric_context_is_rejected(
    tmp_path: Path,
    case_id: str,
    expected_error: str,
) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    payload = yaml.safe_load(contexts.read_text(encoding="utf-8"))
    context = payload["contexts"][0]
    if case_id == "empty_identity":
        context["metric_identity_fields"] = []
    elif case_id == "missing_name":
        context["metric_identity_fields"] = ["metric_unit"]
    elif case_id == "empty_metrics":
        context["allowed_metrics"] = []
    else:
        context["allowed_not_applicable"].append("metric_name")
    contexts.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = _run(contexts, rows, pairs, tmp_path / "out")
    assert completed.returncode == 1
    assert expected_error in completed.stderr


def test_pair_without_requested_metric_is_rejected(tmp_path: Path) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    payload = yaml.safe_load(pairs.read_text(encoding="utf-8"))
    del payload["pairs"][0]["metric"]
    pairs.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = _run(contexts, rows, pairs, tmp_path / "out")
    assert completed.returncode == 1
    assert "missing required key 'metric'" in completed.stderr


def test_unsupported_requested_metric_is_rejected(tmp_path: Path) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    payload = yaml.safe_load(pairs.read_text(encoding="utf-8"))
    payload["pairs"][0]["metric"] = "p95_latency_ms"
    pairs.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = _run(contexts, rows, pairs, tmp_path / "out")
    assert completed.returncode == 1
    assert "requests unsupported metric 'p95_latency_ms'" in completed.stderr


def test_rows_must_match_the_requested_metric(tmp_path: Path) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    context_payload = yaml.safe_load(contexts.read_text(encoding="utf-8"))
    context_payload["contexts"][0]["allowed_metrics"].append("p95_latency_ms")
    contexts.write_text(yaml.safe_dump(context_payload, sort_keys=False), encoding="utf-8")
    pair_payload = yaml.safe_load(pairs.read_text(encoding="utf-8"))
    pair_payload["pairs"][0]["metric"] = "p95_latency_ms"
    pair_payload["pairs"] = [pair_payload["pairs"][0]]
    pairs.write_text(yaml.safe_dump(pair_payload, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "out"
    completed = _run(contexts, rows, pairs, out_dir)
    assert completed.returncode == 0, completed.stderr
    decision = json.loads((out_dir / "pair_decisions.json").read_text(encoding="utf-8"))[0]
    assert decision["metric"] == "p95_latency_ms"
    assert decision["decision"] == "context_mismatch"
    assert decision["mismatched_fields"] == ["metric_name"]


def test_withdrawn_single_row_claims_mode_is_unavailable(tmp_path: Path) -> None:
    contexts, rows, pairs = _write_fixture(tmp_path)
    report = tmp_path / "report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_pairwise_admission.py",
            "--contexts",
            str(contexts),
            "--rows",
            str(rows),
            "--pairs",
            str(pairs),
            "--out-dir",
            str(tmp_path / "pairwise-out"),
            "--claims",
            "manifests/external_admissibility/external_ids_claims.yaml",
            "--out",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not report.exists()
