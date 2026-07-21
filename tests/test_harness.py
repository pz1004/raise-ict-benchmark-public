from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from raise_ict.attacks import (
    ConstrainedAttackConfig,
    evaluate_constrained_perturbations,
    evaluate_validity,
    generate_constrained_perturbations,
    validity_rate,
)
from raise_ict.datasets import (
    SyntheticDatasetSpec,
    load_cicids2017,
    load_cse_cic_ids2018,
    load_synthetic_frame,
    load_ton_iot_network,
)
from raise_ict.metrics import classification_summary, edge_penalty, raise_score, service_cost
from raise_ict.models import build_model
from raise_ict.pipeline import _asr_for_valid_malicious, train_and_evaluate
from raise_ict.preprocessing import FlowPreprocessor
from raise_ict.profiling import profile_predict
from raise_ict.validation import audit_completion


ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SPEC = importlib.util.spec_from_file_location("audit_hardware", ROOT / "scripts" / "audit_hardware.py")
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
classify_hardware = _AUDIT_MODULE.classify_hardware
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
_TIER_E_SPEC = importlib.util.spec_from_file_location("run_tier_e_core4", ROOT / "scripts" / "run_tier_e_core4.py")
assert _TIER_E_SPEC is not None and _TIER_E_SPEC.loader is not None
_TIER_E_MODULE = importlib.util.module_from_spec(_TIER_E_SPEC)
_TIER_E_SPEC.loader.exec_module(_TIER_E_MODULE)
_MLP_TIER_E_SPEC = importlib.util.spec_from_file_location(
    "run_tier_e_core4_mlp_challenger",
    ROOT / "scripts" / "run_tier_e_core4_mlp_challenger.py",
)
assert _MLP_TIER_E_SPEC is not None and _MLP_TIER_E_SPEC.loader is not None
_MLP_TIER_E_MODULE = importlib.util.module_from_spec(_MLP_TIER_E_SPEC)
_MLP_TIER_E_SPEC.loader.exec_module(_MLP_TIER_E_MODULE)
_TIMED_TIER_E_SPEC = importlib.util.spec_from_file_location(
    "run_tier_e_core4_hgb_mlp_timed",
    ROOT / "scripts" / "run_tier_e_core4_hgb_mlp_timed.py",
)
assert _TIMED_TIER_E_SPEC is not None and _TIMED_TIER_E_SPEC.loader is not None
_TIMED_TIER_E_MODULE = importlib.util.module_from_spec(_TIMED_TIER_E_SPEC)
_TIMED_TIER_E_SPEC.loader.exec_module(_TIMED_TIER_E_MODULE)
_ANALYZE_SPEC = importlib.util.spec_from_file_location("analyze_results", ROOT / "scripts" / "analyze_results.py")
assert _ANALYZE_SPEC is not None and _ANALYZE_SPEC.loader is not None
_ANALYZE_MODULE = importlib.util.module_from_spec(_ANALYZE_SPEC)
_ANALYZE_SPEC.loader.exec_module(_ANALYZE_MODULE)


def test_synthetic_dataset_loader_has_expected_columns() -> None:
    frame = load_synthetic_frame(SyntheticDatasetSpec(n_samples=32, seed=1))
    assert {"flow_duration", "protocol", "service", "group", "label"}.issubset(frame.columns)
    assert set(frame["label"].unique()).issubset({0, 1})


def test_preprocessor_uses_train_only_schema() -> None:
    train = load_synthetic_frame(SyntheticDatasetSpec(n_samples=80, seed=2))
    test = load_synthetic_frame(SyntheticDatasetSpec(n_samples=40, seed=3))
    test.loc[0, "service"] = "unseen_service"
    prep = FlowPreprocessor(categorical_columns=["protocol", "service"], log_columns=["fwd_bytes", "bwd_bytes"])
    x_train = prep.fit_transform(train)
    x_test = prep.transform(test)
    assert list(x_train.columns) == list(x_test.columns)
    assert "service=unseen_service" not in x_test.columns


def test_preprocessor_explicit_empty_categorical_list_disables_auto_encoding() -> None:
    frame = pd.DataFrame(
        {
            "numeric_as_text": ["1", "2", "3"],
            "high_cardinality_text": ["alpha", "beta", "gamma"],
            "label": [0, 1, 0],
        }
    )
    prep = FlowPreprocessor(categorical_columns=[])
    transformed = prep.fit_transform(frame)
    assert "numeric_as_text" in transformed.columns
    assert not any(col.startswith("high_cardinality_text=") for col in transformed.columns)


def test_mlp_sklearn_model_factory_fits_binary_frame() -> None:
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(640, 6)), columns=[f"x{i}" for i in range(6)])
    y = pd.Series((x["x0"] + x["x1"] > 0).astype(int))
    model = build_model("mlp_sklearn", seed=11)

    model.fit(x, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1]
    assert len(model.predict(x.head(5))) == 5


def test_hist_gradient_boosting_model_factory_supports_fixed_challenger() -> None:
    model = build_model("hist_gradient_boosting", seed=3)
    assert model.__class__.__name__ == "HistGradientBoostingClassifier"
    assert model.max_iter == 100
    assert model.class_weight == "balanced"


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


def test_a1_validity_report_preserves_immutable_features() -> None:
    features = pd.DataFrame({"mutable_rate": [0.2, 0.3, 0.4], "locked": [7.0, 8.0, 9.0]})
    labels = pd.Series([0, 1, 1])
    cfg = ConstrainedAttackConfig(
        epsilon=0.05,
        mutable_features=["mutable_rate"],
        nonnegative_features=["mutable_rate", "locked"],
        seed=11,
    )

    evaluation = evaluate_constrained_perturbations(features, cfg, labels=labels)

    assert evaluation.x_adv["locked"].equals(features["locked"])
    assert evaluation.report.immutable_pass_rate == 1.0
    assert evaluation.report.invalid_count == 0


def test_a1_validity_report_rejects_budget_violations() -> None:
    clean = pd.DataFrame({"mutable_rate": [0.0, 0.0], "locked": [1.0, 1.0]})
    attacked = pd.DataFrame({"mutable_rate": [0.20, 0.01], "locked": [1.0, 1.0]})
    labels = pd.Series([1, 1])
    cfg = ConstrainedAttackConfig(epsilon=0.05, mutable_features=["mutable_rate"], budget_norm="inf")

    report = evaluate_validity(clean, attacked, cfg, labels=labels, scales={"mutable_rate": 1.0})

    assert report.valid_mask.tolist() == [False, True]
    assert report.valid_count == 1
    assert report.invalid_count == 1
    assert report.budget_pass_rate == 0.5


def test_score_search_attack_uses_model_score() -> None:
    features = pd.DataFrame({"mutable_rate": [0.0] * 16, "locked": [1.0] * 16})
    labels = pd.Series([1] * 16)
    cfg = ConstrainedAttackConfig(
        epsilon=1.0,
        mutable_features=["mutable_rate"],
        nonnegative_features=[],
        seed=9,
        strategy="score_search",
        n_candidates=24,
    )

    attacked = generate_constrained_perturbations(
        features,
        cfg,
        labels=labels,
        score_fn=lambda frame: frame["mutable_rate"].to_numpy(),
    )

    assert attacked["locked"].eq(1.0).all()
    assert attacked["mutable_rate"].min() > 0.0


def test_ton_iot_attack_type_holdout_split(tmp_path: Path) -> None:
    path = tmp_path / "ton.csv"
    pd.DataFrame(
        {
            "duration": [1.0] * 12,
            "src_bytes": list(range(12)),
            "dst_bytes": list(range(12)),
            "proto": ["tcp"] * 12,
            "service": ["-"] * 12,
            "conn_state": ["SF"] * 12,
            "label": [0] * 6 + [1] * 3 + [1] * 3,
            "type": ["normal"] * 6 + ["backdoor"] * 3 + ["ransomware"] * 3,
        }
    ).to_csv(path, index=False)

    frame = load_ton_iot_network(
        {
            "path": str(path),
            "split_strategy": "attack_type_holdout",
            "holdout_attack_types": ["ransomware"],
            "holdout_normal_test_size": 0.5,
            "seed": 3,
        }
    )

    heldout = frame["attack_type"].eq("ransomware")
    seen_attack = frame["attack_type"].eq("backdoor")
    assert frame.loc[heldout, "split"].eq("test").all()
    assert frame.loc[seen_attack, "split"].eq("train").all()
    assert {"train", "test"}.issubset(set(frame["split"]))


def test_cicids2017_day_holdout_split(tmp_path: Path) -> None:
    root = tmp_path / "cic"
    root.mkdir()
    base = pd.DataFrame(
        {
            "Flow Duration": [1, 2, 3, 4],
            "Total Fwd Packet": [1, 2, 3, 4],
            "Label": ["BENIGN", "BENIGN", "PortScan", "DDoS"],
        }
    )
    monday = "Monday-WorkingHours.pcap_ISCX.csv.parquet"
    friday = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
    base.to_parquet(root / monday, index=False)
    base.to_parquet(root / friday, index=False)

    frame = load_cicids2017(
        {
            "data_root": str(root),
            "files": [monday, friday],
            "split_strategy": "day_holdout",
            "holdout_days": ["friday"],
        }
    )

    assert frame.loc[frame["day"].eq("monday"), "split"].eq("train").all()
    assert frame.loc[frame["day"].eq("friday"), "split"].eq("test").all()
    assert set(frame["label"]) == {0, 1}
    assert {"flow_duration", "total_fwd_packet", "attack_type", "scenario"}.issubset(frame.columns)


def test_cse_cic_ids2018_date_holdout_split(tmp_path: Path) -> None:
    root = tmp_path / "cse"
    root.mkdir()
    base = pd.DataFrame(
        {
            "Dst Port": [80, 443, 22, 3389],
            "Protocol": [6, 6, 6, 6],
            "Timestamp": ["01/03/2018 09:00:00"] * 4,
            "Flow Duration": [1, 2, 3, 4],
            "Tot Fwd Pkts": [1, 2, 3, 4],
            "Label": ["Benign", "Benign", "Bot", "DDoS attacks-LOIC-HTTP"],
        }
    )
    train_file = "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    test_file = "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    base.to_csv(root / train_file, index=False)
    base.to_csv(root / test_file, index=False)

    frame = load_cse_cic_ids2018(
        {
            "data_root": str(root),
            "files": [train_file, test_file],
            "split_strategy": "date_holdout",
            "holdout_dates": ["02-03-2018"],
        }
    )

    assert frame.loc[frame["date"].eq("01-03-2018"), "split"].eq("train").all()
    assert frame.loc[frame["date"].eq("02-03-2018"), "split"].eq("test").all()
    assert set(frame["label"]) == {0, 1}
    assert {"flow_duration", "tot_fwd_pkts", "attack_type", "day", "date", "scenario"}.issubset(frame.columns)


def test_metric_aggregation_outputs_standard_values() -> None:
    summary = classification_summary([0, 0, 1, 1], [0, 1, 1, 1])
    cost = service_cost([0, 0, 1, 1], [0, 1, 1, 1])
    score = raise_score(summary["utility"], summary["utility"], 0.1, 1.0, cost)
    assert 0.0 <= summary["clean_macro_f1"] <= 1.0
    assert cost > 0.0
    assert score <= 1.0


def test_asr_denominator_uses_only_valid_malicious_samples() -> None:
    y_true = pd.Series([1, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    valid_mask = np.array([True, True, False, True])

    assert _asr_for_valid_malicious(y_true, y_pred, valid_mask) == 0.5


def test_raise_score_uses_log_normalized_edge_penalties() -> None:
    penalty = edge_penalty(value=10.0, threshold=10.0, cap=100.0)
    score = raise_score(
        clean_utility=1.0,
        robust_utility=0.8,
        p95_latency_ms=10.0,
        peak_mem_mb=0.0,
        service_cost_value=0.0,
        energy_per_flow_j=0.0,
        latency_budget_ms=10.0,
        latency_cap_ms=100.0,
    )

    assert np.isclose(penalty, np.log1p(1.0) / np.log1p(10.0))
    assert np.isclose(score, 0.35 + 0.65 * 0.8 - 0.05 * penalty)


def test_service_cost_vectorized_path_matches_expected_cost() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    cost = service_cost(y_true, y_pred)

    assert cost == 1.0 / 2.0 + 10.0 * 2.0 / 3.0


def test_validity_rate_combines_multiple_nonnegative_constraints() -> None:
    features = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [3.0, 4.0, -0.1]})
    cfg = ConstrainedAttackConfig(nonnegative_features=["a", "b"])

    assert validity_rate(features, cfg) == 1.0 / 3.0


def test_profile_predict_uses_measured_energy_metadata() -> None:
    class ConstantModel:
        def predict(self, features):
            return [0] * len(features)

    features = pd.DataFrame({"x": [1, 2, 3, 4]})
    profile = profile_predict(
        ConstantModel(),
        features,
        repeats=2,
        hardware={
            "hardware_id": "raspberry_pi_5",
            "measurement_mode": "measured_external_meter",
            "average_power_w": 2.0,
            "measurement_duration_s": 4.0,
            "measured_flows": 8,
        },
    )
    assert profile["energy_per_flow_j"] == 1.0
    assert profile["thread_count"] == 1
    assert profile["batch_size"] == 1
    assert profile["measurement_mode"] == "measured_external_meter"
    assert profile["energy_source"] == "declared"


def test_a4_split_shift_records_metadata_without_attack_generation() -> None:
    config = {
        "seed": 2,
        "split_id": "synthetic_shift",
        "dataset": {"dataset_id": "synthetic_raise_ict", "n_samples": 80, "seed": 2},
        "model": {"model_id": "logistic_regression"},
        "attack": {
            "threat_id": "a4_split_shift",
            "threat_type": "shift",
            "epsilon": 0.0,
            "mutable_features": ["flow_duration"],
            "shift_group_field": "group",
        },
        "hardware": {"hardware_id": "cpu_proxy"},
        "preprocessing": {
            "categorical_columns": ["protocol", "service"],
            "log_columns": ["fwd_bytes", "bwd_bytes"],
        },
        "config_path": "synthetic_shift.yaml",
    }

    result = train_and_evaluate(config)

    assert result.threat_id == "a4_split_shift"
    assert result.asr == 0.0
    assert result.valid_count + result.invalid_count > 0
    assert result.invalid_count == 0
    assert result.shift_group_field == "group"
    assert result.source_split == "train"
    assert result.target_split == "test"


def test_cli_smoke_path_writes_expected_artifacts(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    raw_dir = tmp_path / "raw"
    attack_dir = tmp_path / "attack_raw"
    profile_dir = tmp_path / "profile_raw"
    table_dir = tmp_path / "tables"
    fig_dir = tmp_path / "figures"
    attack_config = tmp_path / "attack.yaml"
    hardware_config = tmp_path / "hardware.yaml"
    attack_config.write_text(
        "\n".join(
            [
                "threat_id: custom_test_attack",
                "epsilon: 0.05",
                "mutable_features:",
                "  - flow_duration",
                "nonnegative_features:",
                "  - flow_duration",
            ]
        ),
        encoding="utf-8",
    )
    hardware_config.write_text("hardware_id: custom_test_cpu\nmeasurement_mode: proxy\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/make_splits.py",
            "--config",
            "configs/experiments/tier_s.yaml",
            "--out-dir",
            str(split_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/experiments/tier_s.yaml", "--out-dir", str(raw_dir)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/attack.py", "--config", str(attack_config), "--out-dir", str(attack_dir)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/profile_edge.py", "--config", str(hardware_config), "--out-dir", str(profile_dir)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_results.py",
            "--results",
            str(raw_dir),
            "--out",
            str(table_dir),
            "--figures",
            str(fig_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (split_dir / "split_manifest.csv").exists()
    raw_results = sorted(raw_dir.glob("*.csv"))
    attack_results_files = sorted(attack_dir.glob("*.csv"))
    profile_results_files = sorted(profile_dir.glob("*.csv"))
    assert len(raw_results) == 1
    assert len(attack_results_files) == 1
    assert len(profile_results_files) == 1
    assert (table_dir / "table_dataset_suite.csv").exists()
    assert (table_dir / "table_main_results.csv").exists()
    assert (fig_dir / "figure_pipeline.pdf").exists()
    assert (fig_dir / "figure_pareto.pdf").exists()
    attack_results = pd.read_csv(attack_results_files[0])
    profile_results = pd.read_csv(profile_results_files[0])
    assert attack_results.loc[0, "threat_id"] == "custom_test_attack"
    assert {"valid_count", "invalid_count", "budget_pass_rate", "preprocessing_state_sha256"}.issubset(
        attack_results.columns
    )
    assert profile_results.loc[0, "hardware_id"] == "custom_test_cpu"
    assert {"thread_count", "batch_size", "runtime", "measurement_mode", "energy_source"}.issubset(
        profile_results.columns
    )
    results = pd.read_csv(table_dir / "table_main_results.csv")
    assert "raise_score" in results.columns
    assert "valid_count" in results.columns


def test_merge_artifacts_cli_merges_csv_and_json_lists(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_out = tmp_path / "merged.csv"
    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    json_out = tmp_path / "merged.json"

    pd.DataFrame([{"dataset": "A", "rows": 1}]).to_csv(csv_a, index=False)
    pd.DataFrame([{"dataset": "B", "rows": 2}]).to_csv(csv_b, index=False)
    json_a.write_text('[{"dataset": "A"}]\n', encoding="utf-8")
    json_b.write_text('[{"dataset": "B"}, {"dataset": "C"}]\n', encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/merge_artifacts.py", str(csv_a), str(csv_b), "--out", str(csv_out)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/merge_artifacts.py", str(json_a), str(json_b), "--out", str(json_out)],
        cwd=ROOT,
        check=True,
    )

    assert len(pd.read_csv(csv_out)) == 2
    assert len(pd.read_json(json_out)) == 3


def test_analyze_results_cli_writes_bundle(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    summary = tmp_path / "summary.csv"
    out = tmp_path / "analysis"
    raw_rows = []
    for seed in [0, 1]:
        raw_rows.extend(
            [
                {
                    "dataset": "synthetic",
                    "model_id": "logistic_regression",
                    "seed": seed,
                    "threat_id": "a0_clean",
                    "robust_utility": 0.90 - 0.01 * seed,
                    "asr": 0.0,
                    "validity_rate": 1.0,
                    "valid_count": 10,
                    "invalid_count": 0,
                },
                {
                    "dataset": "synthetic",
                    "model_id": "logistic_regression",
                    "seed": seed,
                    "threat_id": "a1_constrained_feature",
                    "robust_utility": 0.70 - 0.01 * seed,
                    "asr": 0.25,
                    "validity_rate": 1.0,
                    "valid_count": 8,
                    "invalid_count": 2,
                },
            ]
        )
    pd.DataFrame(raw_rows).to_csv(raw, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "model_id": "logistic_regression",
                "threat_id": "a0_clean",
                "robust_utility": 0.895,
                "robust_utility_std": 0.01,
                "p95_latency_ms": 0.10,
                "n_runs": 2,
            },
            {
                "dataset": "synthetic",
                "model_id": "logistic_regression",
                "threat_id": "a1_constrained_feature",
                "robust_utility": 0.695,
                "robust_utility_std": 0.01,
                "p95_latency_ms": 0.11,
                "n_runs": 2,
            },
        ]
    ).to_csv(summary, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--raw",
            str(raw),
            "--summary",
            str(summary),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )

    assert (out / "analysis-report.md").exists()
    assert (out / "stats-appendix.md").exists()
    assert (out / "paired_attack_drop.csv").exists()
    assert (out / "figures" / "figure-01-clean-vs-constrained.pdf").exists()
    paired = pd.read_csv(out / "paired_attack_drop.csv")
    assert paired.loc[0, "mean_valid_count"] == 8.0
    assert paired.loc[0, "mean_invalid_count"] == 2.0


def test_run_benchmark_accepts_single_experiment_config(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_manifest = tmp_path / "splits.csv"
    feature_schema = tmp_path / "features.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--config",
            "configs/experiments/tier_s.yaml",
            "--out-dir",
            str(raw_dir),
            "--split-manifest",
            str(split_manifest),
            "--feature-schema",
            str(feature_schema),
        ],
        cwd=ROOT,
        check=True,
    )

    raw_files = sorted(raw_dir.glob("*.csv"))
    assert len(raw_files) == 1
    result = pd.read_csv(raw_files[0])
    splits = pd.read_csv(split_manifest)
    feature_records = json.loads(feature_schema.read_text(encoding="utf-8"))
    assert len(splits) == 1
    assert {"raw_rows", "dataset_config_sha256", "preprocessing_state_sha256", "config_path"}.issubset(
        splits.columns
    )
    assert feature_records[0]["dataset"] == "synthetic_raise_ict"
    assert feature_records[0]["preprocessing_state_sha256"]
    assert result.loc[0, "config_path"] == "configs/experiments/tier_s.yaml"
    assert result.loc[0, "valid_count"] + result.loc[0, "invalid_count"] > 0


def test_run_benchmark_accepts_hardware_override_and_profile_manifest(tmp_path: Path) -> None:
    config = tmp_path / "grid.yaml"
    hardware = tmp_path / "edge.yaml"
    raw_dir = tmp_path / "raw"
    split_manifest = tmp_path / "splits.csv"
    feature_schema = tmp_path / "features.json"
    profile_manifest = tmp_path / "profile.json"
    config.write_text(
        "\n".join(
            [
                "profile_repeats: 1",
                "test_size: 0.3",
                "seeds: [0]",
                "datasets:",
                "  - dataset_id: synthetic_raise_ict",
                "    n_samples: 48",
                "    seed: 4",
                "    split_id: synthetic_edge_smoke",
                "models:",
                "  - model_id: logistic_regression",
                "threats:",
                "  - threat_id: a0_clean",
                "    epsilon: 0.0",
                "    mutable_features: []",
                "    nonnegative_features: []",
                "preprocessing:",
                "  categorical_columns:",
                "    - protocol",
                "    - service",
                "  log_columns:",
                "    - fwd_bytes",
                "    - bwd_bytes",
            ]
        ),
        encoding="utf-8",
    )
    hardware.write_text(
        "\n".join(
            [
                "hardware_id: raspberry_pi_5_test",
                "measurement_mode: measured_external_meter",
                "energy_source: external_power_meter",
                "average_power_w: 2.0",
                "measurement_duration_s: 4.0",
                "measured_flows: 8",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--config",
            str(config),
            "--out-dir",
            str(raw_dir),
            "--split-manifest",
            str(split_manifest),
            "--feature-schema",
            str(feature_schema),
            "--hardware-config",
            str(hardware),
            "--profile-manifest",
            str(profile_manifest),
        ],
        cwd=ROOT,
        check=True,
    )
    results = pd.concat(pd.read_csv(path) for path in raw_dir.glob("*.csv"))
    profile = json.loads(profile_manifest.read_text(encoding="utf-8"))
    assert results["hardware_id"].unique().tolist() == ["raspberry_pi_5_test"]
    assert results["energy_per_flow_j"].min() > 0.0
    assert results["measurement_mode"].unique().tolist() == ["measured_external_meter"]
    assert results["energy_source"].unique().tolist() == ["external_power_meter"]
    assert profile["profile"]["hardware_id"] == "raspberry_pi_5_test"
    assert profile["profile"]["measurement_mode"] == "measured_external_meter"


def test_validate_hardware_config_rejects_unmeasured_template() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_hardware_config.py",
            "--config",
            "configs/hardware/jetson_orin_nx_super_template.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    failed_ids = {check["id"] for check in report["failed_checks"]}
    assert completed.returncode == 1
    assert "hardware_id.not_template" in failed_ids
    assert "hardware.power_mode" in failed_ids
    assert "energy.measurement_window" in failed_ids
    assert "energy.measurement_mode" in failed_ids
    assert "energy.source" in failed_ids
    assert "energy.positive_input" in failed_ids


def test_validate_hardware_config_rejects_placeholder_metadata_with_energy(tmp_path: Path) -> None:
    config = tmp_path / "edge.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_40w_labmeter",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: replace_with_nvpmodel_mode",
                "measurement_mode: measured_external_meter",
                "energy_source: external_power_meter",
                "measurement_window: replace_with_inference_only_window_description",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    failed_ids = {check["id"] for check in report["failed_checks"]}
    assert completed.returncode == 1
    assert "hardware.power_mode" in failed_ids
    assert "energy.measurement_window" in failed_ids


def test_validate_hardware_config_rejects_untrusted_measurement_words(tmp_path: Path) -> None:
    config = tmp_path / "edge.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_40w_labmeter",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: MAXN_SUPER",
                "measurement_mode: not_measured_external_meter",
                "energy_source: guessed_power_meter",
                "measurement_window: inference-only Core4 profiling window, board input channel",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    failed_ids = {check["id"] for check in report["failed_checks"]}
    assert completed.returncode == 1
    assert "energy.measurement_mode" in failed_ids
    assert "energy.source" in failed_ids


def test_validate_hardware_config_rejects_vague_energy_source(tmp_path: Path) -> None:
    config = tmp_path / "edge.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_40w_labmeter",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: MAXN_SUPER",
                "measurement_mode: measured_external_meter",
                "energy_source: external",
                "measurement_window: inference-only Core4 profiling window, board input channel",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    failed_ids = {check["id"] for check in report["failed_checks"]}
    assert completed.returncode == 1
    assert "energy.source" in failed_ids


def test_validate_hardware_config_accepts_measured_edge_config(tmp_path: Path) -> None:
    config = tmp_path / "jetson.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_40w_labmeter",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: MAXN_SUPER",
                "measurement_mode: measured_external_meter",
                "energy_source: external_power_meter",
                "measurement_window: inference-only Core4 profiling window, meter channel board input",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["effective_energy_per_flow_j"] == 0.002


def test_validate_hardware_config_accepts_jetson_ina3221_source(tmp_path: Path) -> None:
    config = tmp_path / "jetson_ina3221.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_maxn_ina3221",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: MAXN_SUPER",
                "measurement_mode: measured_onboard_sensor",
                "energy_source: jetson_ina3221_vdd_in",
                "measurement_window: inference-only predict loop with INA3221 VDD_IN sampling",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["effective_energy_per_flow_j"] == 0.002


def test_validate_hardware_config_accepts_reported_jetpack_patch_stack(tmp_path: Path) -> None:
    config = tmp_path / "jetson_36_4_7.yaml"
    config.write_text(
        "\n".join(
            [
                "hardware_id: jetson_orin_nx16_super_40w_labmeter",
                "device_class: physical_edge",
                "runtime: python_sklearn_cpu",
                "jetson_linux_release: R36.4.7",
                "jetpack_release: 6.2.1+b38",
                "l4t_core_package: 36.4.7-20250918154033",
                "cuda_compiler_release: 12.6.68",
                "device_tree_model: NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super",
                "thread_count: 8",
                "batch_size: 1",
                "power_mode: MAXN_SUPER",
                "measurement_mode: measured_external_meter",
                "energy_source: external_power_meter",
                "measurement_window: inference-only Core4 profiling window, board input channel",
                "average_power_w: 8.0",
                "measurement_duration_s: 10.0",
                "measured_flows: 40000",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "scripts/validate_hardware_config.py", "--config", str(config)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["valid"] is True


def test_measure_inference_window_outputs_measurement_fields(tmp_path: Path) -> None:
    out = tmp_path / "energy_window.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/measure_inference_window.py",
            "--config",
            "configs/experiments/tier_s.yaml",
            "--seconds",
            "0.01",
            "--warmup-iterations",
            "1",
            "--start-delay-s",
            "0",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["dataset"] == "synthetic_raise_ict"
    assert report["model_id"] == "logistic_regression"
    assert report["flows_per_iteration"] > 0
    assert report["measured_flows"] >= report["flows_per_iteration"]
    assert report["measurement_duration_s"] > 0.0
    assert report["recommended_hardware_fields"]["measured_flows"] == report["measured_flows"]


def test_measure_inference_window_samples_fake_jetson_power_rail(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    hwmon = sysfs / "bus" / "i2c" / "drivers" / "ina3221" / "1-0040" / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "in1_label").write_text("VDD_IN\n", encoding="utf-8")
    (hwmon / "in1_input").write_text("5000\n", encoding="utf-8")
    (hwmon / "curr1_input").write_text("2000\n", encoding="utf-8")

    listed = subprocess.run(
        [
            sys.executable,
            "scripts/measure_inference_window.py",
            "--list-power-rails",
            "--power-sysfs-root",
            str(sysfs),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rails = json.loads(listed.stdout)
    assert rails[0]["label"] == "VDD_IN"

    out = tmp_path / "energy_window.json"
    power_log = tmp_path / "power.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/measure_inference_window.py",
            "--config",
            "configs/experiments/tier_s.yaml",
            "--seconds",
            "0.01",
            "--warmup-iterations",
            "1",
            "--start-delay-s",
            "0",
            "--power-sysfs-root",
            str(sysfs),
            "--power-rail",
            "VDD_IN",
            "--power-sample-interval-s",
            "0.001",
            "--power-log-out",
            str(power_log),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    onboard = report["onboard_power"]
    assert onboard["measurement_mode"] == "measured_onboard_sensor"
    assert onboard["energy_source"] == "jetson_ina3221_vdd_in"
    assert abs(onboard["average_power_w"] - 10.0) < 1e-9
    assert onboard["sample_count"] >= 1
    assert report["recommended_hardware_fields"]["average_power_w"] == onboard["average_power_w"]
    assert power_log.read_text(encoding="utf-8").startswith("sample_index,monotonic_s")


def test_tier_e_core4_dry_run_lists_required_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_tier_e_core4.py",
            "--hardware-config",
            "configs/hardware/edge_device_template.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    flattened = [" ".join(command) for command in payload["commands"]]
    assert len(payload["commands"]) == 10
    assert any("scripts/validate_hardware_config.py" in command for command in flattened)
    assert any("configs/experiments/tier_p_expanded.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cicids2017.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cse_cic_ids2018.yaml" in command for command in flattened)
    assert any("--hardware-audit manifests/hardware/tier_e_hardware_audit.json" in command for command in flattened)
    assert any("--manuscript jkics/jkics.tex" in command for command in flattened)
    assert any("--bibliography jkics/reference.bib" in command for command in flattened)
    assert any("--require-tier-e" in command and "--strict" in command for command in flattened)


def test_tier_e_core4_mlp_challenger_dry_run_lists_required_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_tier_e_core4_mlp_challenger.py",
            "--hardware-config",
            "configs/hardware/edge_device_template.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    flattened = [" ".join(command) for command in payload["commands"]]
    assert len(payload["commands"]) == 21
    assert any("scripts/validate_hardware_config.py" in command for command in flattened)
    assert any(
        "scripts/audit_hardware.py --out manifests/hardware/tier_e_mlp_challenger_hardware_audit.json" in command
        for command in flattened
    )
    assert any("scripts/run_tier_e_core4.py" in command for command in flattened)
    assert any("configs/experiments/tier_p_expanded_mlp.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cicids2017_mlp.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cse_cic_ids2018_mlp.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cicids2017_random_control.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cse_cic_ids2018_random_control.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cicids2017_random_control_mlp.yaml" in command for command in flattened)
    assert any("configs/experiments/tier_p_cse_cic_ids2018_random_control_mlp.yaml" in command for command in flattened)
    assert any("results/tables/tier_e_core4_mlp_challenger" in command for command in flattened)
    assert any("results/tables/tier_e_random_control_mlp_challenger" in command for command in flattened)
    assert any("--write-acceptance-report-only" in command for command in flattened)
    assert any(
        "--out manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json" in command
        for command in flattened
    )
    assert any(
        "--hardware-audit manifests/hardware/tier_e_mlp_challenger_hardware_audit.json" in command
        for command in flattened
    )
    assert not any("--out jkics/audit_completion_current.json" in command for command in flattened)
    assert any("--expected-raw-rows 320" in command for command in flattened)
    assert any("--expected-summary-rows 64" in command for command in flattened)
    assert any("mlp_sklearn" in command and "--expected-models" in command for command in flattened)
    assert any("--require-tier-e" in command and "--strict" in command for command in flattened)
    assert any("conditional" in note for note in payload["notes"])
    assert any("does not overwrite jkics/audit_completion_current.json" in note for note in payload["notes"])


def test_tier_e_core4_mlp_challenger_rejects_non_edge_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"classification": {"tier_e_eligible": False, "reasons": ["WSL test host"]}}),
        encoding="utf-8",
    )

    try:
        _MLP_TIER_E_MODULE._ensure_tier_e_host(str(audit))
    except SystemExit as exc:
        assert "non-edge host" in str(exc)
    else:
        raise AssertionError("expected non-edge host to be rejected")


def test_tier_e_core4_hgb_mlp_timed_dry_run_lists_required_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_tier_e_core4_hgb_mlp_timed.py",
            "--hardware-config",
            "configs/hardware/edge_device_template.yaml",
            "--seeds",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    flattened = [" ".join(command) for command in payload["commands"]]
    assert len(payload["generated_configs"]) == 5
    assert any("scripts/validate_hardware_config.py" in command for command in flattened)
    assert any("scripts/audit_hardware.py --out manifests/hardware/tier_e_core4_hgb_mlp_timed_hardware_audit.json" in command for command in flattened)
    assert any("results/configs/tier_e_core4_hgb_mlp_timed/expanded.yaml" in command for command in flattened)
    assert any("results/timing/tier_e_core4_hgb_mlp_timed/expanded_events.csv" in command for command in flattened)
    assert any("results/tables/tier_e_core4_hgb_mlp_timed" in command for command in flattened)
    assert any("results/tables/tier_e_random_control_hgb_mlp_timed" in command for command in flattened)
    assert any("--run-rejection-suite-only" in command for command in flattened)
    assert any("--expected-raw-rows 800" in command for command in flattened)
    assert any("--expected-summary-rows 80" in command for command in flattened)
    assert any("--expected-split-rows 40" in command for command in flattened)
    assert any("--expected-feature-schema-records 40" in command for command in flattened)
    assert any("--require-timing" in command for command in flattened)
    assert any("hist_gradient_boosting" in command and "--expected-models" in command for command in flattened)
    assert any("does not overwrite older evidence packages" in note for note in payload["notes"])


def test_pairwise_checker_rejects_withdrawn_legacy_claims_mode(tmp_path: Path) -> None:
    report = tmp_path / "legacy-report.md"
    table = tmp_path / "legacy-table.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_pairwise_admission.py",
            "--contexts",
            "unused-contexts.yaml",
            "--rows",
            "unused-rows.yaml",
            "--pairs",
            "unused-pairs.yaml",
            "--out-dir",
            str(tmp_path / "pairwise-out"),
            "--claims",
            "manifests/external_admissibility/external_ids_claims.yaml",
            "--out",
            str(report),
            "--csv",
            str(table),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not report.exists()
    assert not table.exists()


def test_paired_attack_drop_matches_split_id_when_random_controls_are_aggregated() -> None:
    rows = []
    for seed in [0, 1, 2]:
        rows.append(
            {
                "dataset": "CSE-CIC-IDS2018",
                "split_id": "holdout",
                "model_id": "mlp_sklearn",
                "seed": seed,
                "threat_id": "a0_clean",
                "robust_utility": 0.80,
            }
        )
        rows.append(
            {
                "dataset": "CSE-CIC-IDS2018",
                "split_id": "holdout",
                "model_id": "mlp_sklearn",
                "seed": seed,
                "threat_id": "a1_constrained_score_search",
                "robust_utility": 0.60,
                "asr": 0.5,
                "validity_rate": 1.0,
                "valid_count": 100,
                "invalid_count": 0,
            }
        )
        rows.append(
            {
                "dataset": "CSE-CIC-IDS2018",
                "split_id": "random_control",
                "model_id": "mlp_sklearn",
                "seed": seed,
                "threat_id": "a0_clean",
                "robust_utility": 0.95,
            }
        )

    drops = _ANALYZE_MODULE.paired_attack_drop(pd.DataFrame(rows), "a1_constrained_score_search")

    assert len(drops) == 1
    assert drops.loc[0, "split_id"] == "holdout"
    assert drops.loc[0, "n_pairs"] == 3
    assert np.isclose(drops.loc[0, "mean_robust_drop"], 0.20)


def test_tier_e_core4_combines_profile_manifests(tmp_path: Path) -> None:
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        "hardware_id: edge_test\nmeasurement_mode: measured_external_meter\nenergy_source: external_power_meter\n",
        encoding="utf-8",
    )
    original_runs = list(_TIER_E_MODULE.EDGE_RUNS)
    try:
        runs = []
        for idx, energy in enumerate([0.003, 0.002, 0.004]):
            profile_path = tmp_path / f"profile_{idx}.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "hardware": {"hardware_id": "edge_test"},
                        "profile": {
                            "hardware_id": "edge_test",
                            "energy_per_flow_j": energy,
                            "p95_latency_ms_median": 1.0 + idx,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runs.append({"profile": str(profile_path)})
        _TIER_E_MODULE.EDGE_RUNS = runs
        out = tmp_path / "combined.json"
        _TIER_E_MODULE.write_combined_profile_manifest(str(hardware), str(out))
    finally:
        _TIER_E_MODULE.EDGE_RUNS = original_runs
    combined = json.loads(out.read_text(encoding="utf-8"))
    assert combined["profile"]["hardware_id"] == "edge_test"
    assert combined["profile"]["energy_per_flow_j"] == 0.002
    assert len(combined["component_manifests"]) == 3


def test_hardware_audit_rejects_wsl_desktop_as_tier_e() -> None:
    classification = classify_hardware(
        "Linux host 6.6.87.2-microsoft-standard-WSL2 x86_64",
        {
            "Architecture": "x86_64",
            "Model name": "12th Gen Intel(R) Core(TM) i7-12700",
            "Hypervisor vendor": "Microsoft",
        },
        "",
        "NVIDIA GeForce GTX 1660 SUPER, 21 W, 6144 MiB",
    )
    assert classification["tier_e_eligible"] is False
    assert classification["reasons"]


def test_hardware_audit_accepts_named_edge_model() -> None:
    classification = classify_hardware(
        "Linux edge 6.1 aarch64",
        {"Architecture": "aarch64", "Model name": "ARM Cortex-A76"},
        "Raspberry Pi 5 Model B Rev 1.0",
        "",
    )
    assert classification["tier_e_eligible"] is True


def _write_completion_fixture(
    tmp_path: Path,
    tier_e_eligible: bool = False,
    hardware_id: str = "cpu_proxy",
    energy_per_flow_j: float = 0.0,
    include_profile_manifest: bool = False,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Path]:
    datasets = ["CICIDS2017", "CSE-CIC-IDS2018", "TON_IoT", "UNSW-NB15"]
    model_ids = models or ["extra_trees", "logistic_regression", "random_forest"]
    threats = ["a0_clean", "a1_constrained_feature", "a1_constrained_score_search", "a4_split_shift"]
    seed_ids = seeds or [0, 1, 2, 3, 4]
    measurement_mode = "measured_external_meter" if energy_per_flow_j > 0.0 else "proxy"
    energy_source = "external_power_meter" if energy_per_flow_j > 0.0 else "proxy"
    rows = []
    for dataset in datasets:
        for model in model_ids:
            for threat in threats:
                for seed in seed_ids:
                    rows.append(
                        {
                            "dataset": dataset,
                            "split_id": f"{dataset}_split",
                            "seed": seed,
                            "model_id": model,
                            "threat_id": threat,
                            "hardware_id": hardware_id,
                            "clean_macro_f1": 0.8,
                            "clean_bal_acc": 0.8,
                            "robust_utility": 0.7,
                            "asr": 0.1,
                            "validity_rate": 1.0,
                            "p95_latency_ms": 0.1,
                            "throughput_fps": 10.0,
                            "peak_mem_mb": 1.0,
                            "energy_per_flow_j": energy_per_flow_j,
                            "service_cost": 0.1,
                            "raise_score": 0.6,
                            "valid_count": 1000,
                            "invalid_count": 0,
                            "budget_pass_rate": 1.0,
                            "bounds_pass_rate": 1.0,
                            "immutable_pass_rate": 1.0,
                            "relation_pass_rate": 1.0,
                            "thread_count": 4 if energy_per_flow_j > 0.0 else 1,
                            "batch_size": 1,
                            "runtime": "python_sklearn_cpu",
                            "measurement_mode": measurement_mode,
                            "energy_source": energy_source,
                            "config_path": "configs/experiments/tier_p_core4.yaml",
                            "preprocessing_state_sha256": "prep123",
                            "shift_group_field": "date" if threat == "a4_split_shift" else "",
                            "source_split": "train" if threat == "a4_split_shift" else "",
                            "target_split": "test" if threat == "a4_split_shift" else "",
                            "shift_utility_drop": 0.0,
                        }
                    )
    summary_rows = [
        {
            "dataset": dataset,
            "model_id": model,
            "threat_id": threat,
            "clean_macro_f1": 0.8,
            "clean_bal_acc": 0.8,
            "robust_utility": 0.7,
            "asr": 0.1,
            "validity_rate": 1.0,
            "p95_latency_ms": 0.1,
            "raise_score": 0.6,
        }
        for dataset in datasets
        for model in model_ids
        for threat in threats
    ]
    split_rows = [
        {
            "dataset": dataset,
            "split_id": f"{dataset}_split",
            "seed": seed,
            "train_rows": 15000,
            "test_rows": 5000 if dataset == "CSE-CIC-IDS2018" else 1000,
            "train_positive": 100,
            "test_positive": 50,
            "group_field": "date",
            "split_strategy": "date_holdout",
            "feature_count": 10,
            "feature_schema_sha256": "abc",
            "raw_files": f"data/raw/{dataset}.csv",
            "raw_rows": 20000,
            "dataset_config_sha256": "dataset123",
            "preprocessing_state_sha256": "prep123",
            "config_path": "configs/experiments/tier_p_core4.yaml",
            "software_version": "0.1.0",
        }
        for dataset in datasets
        for seed in seed_ids
    ]
    raw = tmp_path / "raw.csv"
    summary = tmp_path / "summary.csv"
    splits = tmp_path / "splits.csv"
    dataset_manifest = tmp_path / "datasets.json"
    feature_schema = tmp_path / "features.json"
    hardware = tmp_path / "hardware.json"
    profile_manifest = tmp_path / "profile_manifest.json"
    manuscript = tmp_path / "paper.tex"
    bibliography = tmp_path / "refs.bib"
    pd.DataFrame(rows).to_csv(raw, index=False)
    pd.DataFrame(summary_rows).to_csv(summary, index=False)
    pd.DataFrame(split_rows).to_csv(splits, index=False)
    dataset_manifest.write_text("[{}" + ",{}" * 14 + "]", encoding="utf-8")
    feature_schema.write_text(
        json.dumps(
            [
                {
                    "dataset": dataset,
                    "split_id": f"{dataset}_split",
                    "seed": seed,
                    "feature_count": 10,
                    "feature_columns": ["x0", "x1"],
                    "preprocessing_state_sha256": "prep123",
                    "config_path": "configs/experiments/tier_p_core4.yaml",
                    "software_version": "0.1.0",
                }
                for dataset in datasets
                for seed in seed_ids
            ]
        ),
        encoding="utf-8",
    )
    hardware.write_text(
        '{"classification": {"tier_e_eligible": '
        + ("true" if tier_e_eligible else "false")
        + ', "reasons": ["test"]}}',
        encoding="utf-8",
    )
    if include_profile_manifest:
        profile_manifest.write_text(
            (
                '{"hardware": {"hardware_id": "'
                + hardware_id
                + '", "measurement_mode": "measured_external_meter", "energy_source": "external_power_meter"}, '
                + '"profile": {"hardware_id": "'
                + hardware_id
                + '", "energy_per_flow_j": '
                + str(energy_per_flow_j)
                + "}}"
            ),
            encoding="utf-8",
        )
    manuscript.write_text(
        "\n".join(
            [
                "CPU-proxy profiling is not physical edge measurement.",
                "not treated as Tier-E evidence",
                "feature-space validity does not prove packet-level",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )
    bibliography.write_text("\n".join(f"@misc{{k{i}, title={{T{i}}}}}" for i in range(10)), encoding="utf-8")
    return {
        "raw_results_path": raw,
        "summary_results_path": summary,
        "split_manifest_path": splits,
        "dataset_manifest_path": dataset_manifest,
        "feature_schema_path": feature_schema,
        "hardware_audit_path": hardware,
        "profile_manifest_path": profile_manifest,
        "manuscript_path": manuscript,
        "bibliography_path": bibliography,
    }


def _write_timing_fixture(tmp_path: Path) -> dict[str, Path]:
    stages = [
        "dataset_load",
        "preprocess_manifest",
        "model_training",
        "threat_evaluation",
        "result_writing",
        "aggregate_results",
        "analysis",
        "admissibility_rejection",
        "strict_audit",
    ]
    events = tmp_path / "timing_events.csv"
    summary = tmp_path / "timing_summary.csv"
    timeline = tmp_path / "command_timeline.json"
    pd.DataFrame(
        [
            {
                "event_id": index + 1,
                "stage": stage,
                "dataset": "",
                "split_id": "",
                "seed": "",
                "model_id": "",
                "threat_id": "",
                "start_iso": "2026-06-12T00:00:00+00:00",
                "end_iso": "2026-06-12T00:00:01+00:00",
                "elapsed_s": 1.0,
                "rows": "",
                "output_path": "",
                "detail": "",
            }
            for index, stage in enumerate(stages)
        ]
    ).to_csv(events, index=False)
    pd.DataFrame(
        [
            {
                "stage": stage,
                "event_count": 1,
                "elapsed_s_total": 1.0,
                "elapsed_s_mean": 1.0,
                "elapsed_s_max": 1.0,
            }
            for stage in stages
        ]
    ).to_csv(summary, index=False)
    timeline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [
                    {
                        "stage": "strict_audit",
                        "command": "python scripts/check_completion.py",
                        "elapsed_s": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "timing_events_path": events,
        "timing_summary_path": summary,
        "command_timeline_path": timeline,
    }


def test_completion_audit_passes_core4_when_tier_e_is_not_required(tmp_path: Path) -> None:
    paths = _write_completion_fixture(tmp_path, tier_e_eligible=False)
    report = audit_completion(**paths, require_tier_e=False)
    assert report["complete"] is True
    assert any(
        check["id"] == "tier_e.physical_edge_available" and check["status"] == "not_required"
        for check in report["checks"]
    )


def test_completion_audit_accepts_configurable_mlp_challenger_shape(tmp_path: Path) -> None:
    model_ids = ["extra_trees", "logistic_regression", "mlp_sklearn", "random_forest"]
    paths = _write_completion_fixture(
        tmp_path,
        tier_e_eligible=True,
        hardware_id="raspberry_pi_5",
        energy_per_flow_j=0.002,
        include_profile_manifest=True,
        models=model_ids,
    )
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "The energy evidence is module-power telemetry from an onboard sensor.",
                "It is not calibrated wall-power or external board-input energy.",
                "feature-space validity does not prove packet-level realizability.",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(
        **paths,
        require_tier_e=True,
        expected_raw_rows=320,
        expected_summary_rows=64,
        expected_models=model_ids,
    )

    assert report["complete"] is True
    assert report["expected_models"] == model_ids
    assert any(
        check["id"] == "core4.raw_rows"
        and check["status"] == "passed"
        and "320" in check["evidence"]
        for check in report["checks"]
    )


def test_completion_audit_accepts_timed_hgb_mlp_shape(tmp_path: Path) -> None:
    model_ids = ["extra_trees", "hist_gradient_boosting", "logistic_regression", "mlp_sklearn", "random_forest"]
    seeds = list(range(10))
    paths = _write_completion_fixture(
        tmp_path,
        tier_e_eligible=True,
        hardware_id="jetson_orin_nx_super",
        energy_per_flow_j=0.002,
        include_profile_manifest=True,
        models=model_ids,
        seeds=seeds,
    )
    timing_paths = _write_timing_fixture(tmp_path)
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "The energy evidence is module-power telemetry from an onboard sensor.",
                "It is not calibrated wall-power or external board-input energy.",
                "feature-space validity does not prove packet-level realizability.",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(
        **paths,
        **timing_paths,
        require_tier_e=True,
        require_timing=True,
        expected_raw_rows=800,
        expected_summary_rows=80,
        expected_models=model_ids,
        expected_seeds=seeds,
        expected_split_rows=40,
        expected_feature_schema_records=40,
    )

    assert report["complete"] is True
    assert report["require_timing"] is True
    assert report["expected_seeds"] == seeds
    assert any(check["id"] == "timing.expected_stages" and check["status"] == "passed" for check in report["checks"])


def test_completion_audit_blocks_missing_timing_stage(tmp_path: Path) -> None:
    paths = _write_completion_fixture(
        tmp_path,
        tier_e_eligible=True,
        hardware_id="jetson_orin_nx_super",
        energy_per_flow_j=0.002,
        include_profile_manifest=True,
    )
    timing_paths = _write_timing_fixture(tmp_path)
    events = pd.read_csv(timing_paths["timing_events_path"])
    events = events[~events["stage"].eq("admissibility_rejection")]
    events.to_csv(timing_paths["timing_events_path"], index=False)
    summary = pd.read_csv(timing_paths["timing_summary_path"])
    summary = summary[~summary["stage"].eq("admissibility_rejection")]
    summary.to_csv(timing_paths["timing_summary_path"], index=False)
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "The energy evidence is module-power telemetry from an onboard sensor.",
                "It is not calibrated wall-power or external board-input energy.",
                "feature-space validity does not prove packet-level realizability.",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(**paths, **timing_paths, require_tier_e=True, require_timing=True)

    assert report["complete"] is False
    assert any(
        check["id"] == "timing.expected_stages" and "admissibility_rejection" in check["evidence"]
        for check in report["blocking_requirements"]
    )


def test_completion_audit_accepts_matched_eleven_citation_entries(tmp_path: Path) -> None:
    paths = _write_completion_fixture(tmp_path, tier_e_eligible=False)
    keys = [f"k{i}" for i in range(11)]
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "CPU-proxy profiling is not physical edge measurement.",
                "not treated as Tier-E evidence",
                "feature-space validity does not prove packet-level",
                r"\cite{" + ",".join(keys) + "}",
            ]
        ),
        encoding="utf-8",
    )
    paths["bibliography_path"].write_text(
        "\n".join(f"@misc{{{key}, title={{T{idx}}}}}" for idx, key in enumerate(keys)),
        encoding="utf-8",
    )

    report = audit_completion(**paths, require_tier_e=False)

    assert report["complete"] is True
    assert any(
        check["id"] == "citations.key_count"
        and check["status"] == "passed"
        and "manuscript cite keys: 11" in check["evidence"]
        for check in report["checks"]
    )
    assert any(
        check["id"] == "citations.bib_count"
        and check["status"] == "passed"
        and "BibTeX entries: 11" in check["evidence"]
        for check in report["checks"]
    )


def test_completion_audit_accepts_supercite_entries(tmp_path: Path) -> None:
    paths = _write_completion_fixture(tmp_path, tier_e_eligible=False)
    keys = [f"k{i}" for i in range(10)]
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "CPU-proxy profiling is not physical edge measurement.",
                "not treated as Tier-E evidence",
                "feature-space validity does not prove packet-level",
                r"\supercite{" + ",".join(keys) + "}",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(**paths, require_tier_e=False)

    assert report["complete"] is True
    assert any(
        check["id"] == "citations.key_count"
        and check["status"] == "passed"
        and "manuscript cite keys: 10" in check["evidence"]
        for check in report["checks"]
    )


def test_completion_audit_blocks_missing_bibliography_key(tmp_path: Path) -> None:
    paths = _write_completion_fixture(tmp_path, tier_e_eligible=False)
    cite_keys = [f"k{i}" for i in range(11)]
    bib_keys = cite_keys[:-1]
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "CPU-proxy profiling is not physical edge measurement.",
                "not treated as Tier-E evidence",
                "feature-space validity does not prove packet-level",
                r"\cite{" + ",".join(cite_keys) + "}",
            ]
        ),
        encoding="utf-8",
    )
    paths["bibliography_path"].write_text(
        "\n".join(f"@misc{{{key}, title={{T{idx}}}}}" for idx, key in enumerate(bib_keys)),
        encoding="utf-8",
    )

    report = audit_completion(**paths, require_tier_e=False)

    assert report["complete"] is False
    assert any(
        check["id"] == "citations.no_missing_or_unused"
        and check["status"] == "incomplete"
        and "k10" in check["evidence"]
        for check in report["blocking_requirements"]
    )


def test_completion_audit_reports_missing_raw_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"dataset": ["partial"]}).to_csv(raw, index=False)

    report = audit_completion(
        raw_results_path=raw,
        summary_results_path=tmp_path / "missing_summary.csv",
        split_manifest_path=tmp_path / "missing_splits.csv",
        dataset_manifest_path=tmp_path / "missing_dataset.json",
        feature_schema_path=tmp_path / "missing_features.json",
        hardware_audit_path=tmp_path / "missing_hardware.json",
        profile_manifest_path=tmp_path / "missing_profile.json",
        manuscript_path=tmp_path / "missing_paper.tex",
        bibliography_path=tmp_path / "missing_refs.bib",
    )

    assert report["complete"] is False
    assert any(
        check["id"] == "schema.result_fields" and check["status"] == "incomplete"
        for check in report["checks"]
    )


def test_completion_audit_blocks_when_tier_e_is_required_but_absent(tmp_path: Path) -> None:
    paths = _write_completion_fixture(tmp_path, tier_e_eligible=False)
    report = audit_completion(**paths, require_tier_e=True)
    assert report["complete"] is False
    assert any(check["id"] == "tier_e.physical_edge_required" for check in report["blocking_requirements"])
    assert any(check["id"] == "profiling.physical_edge_results_required" for check in report["blocking_requirements"])


def test_completion_audit_passes_when_strict_tier_e_evidence_exists(tmp_path: Path) -> None:
    paths = _write_completion_fixture(
        tmp_path,
        tier_e_eligible=True,
        hardware_id="raspberry_pi_5",
        energy_per_flow_j=0.002,
        include_profile_manifest=True,
    )
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "The energy evidence is module-power telemetry from an onboard sensor.",
                "It is not calibrated wall-power or external board-input energy.",
                "feature-space validity does not prove packet-level realizability.",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )
    report = audit_completion(**paths, require_tier_e=True)
    assert report["complete"] is True


def test_completion_audit_blocks_tier_e_without_module_power_boundary(tmp_path: Path) -> None:
    paths = _write_completion_fixture(
        tmp_path,
        tier_e_eligible=True,
        hardware_id="raspberry_pi_5",
        energy_per_flow_j=0.002,
        include_profile_manifest=True,
    )
    paths["manuscript_path"].write_text(
        "\n".join(
            [
                "This paper has physical edge evidence.",
                "feature-space validity does not prove packet-level realizability.",
                r"\cite{k0,k1,k2,k3,k4,k5,k6,k7,k8,k9}",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_completion(**paths, require_tier_e=True)

    assert report["complete"] is False
    assert any(
        check["id"] == "manuscript.claim_boundaries"
        and "module-power" in check["evidence"]
        and "not calibrated wall-power" in check["evidence"]
        for check in report["blocking_requirements"]
    )
