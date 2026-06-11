"""Shared result schema for RAISE-ICT outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


RESULT_FIELDS = [
    "dataset",
    "split_id",
    "seed",
    "model_id",
    "threat_id",
    "hardware_id",
    "clean_macro_f1",
    "clean_bal_acc",
    "robust_utility",
    "asr",
    "validity_rate",
    "p95_latency_ms",
    "throughput_fps",
    "peak_mem_mb",
    "energy_per_flow_j",
    "service_cost",
    "raise_score",
    "valid_count",
    "invalid_count",
    "budget_pass_rate",
    "bounds_pass_rate",
    "immutable_pass_rate",
    "relation_pass_rate",
    "thread_count",
    "batch_size",
    "runtime",
    "measurement_mode",
    "energy_source",
    "config_path",
    "preprocessing_state_sha256",
    "shift_group_field",
    "source_split",
    "target_split",
    "shift_utility_drop",
]


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark result row with the standard RAISE-ICT schema."""

    dataset: str
    split_id: str
    seed: int
    model_id: str
    threat_id: str
    hardware_id: str
    clean_macro_f1: float
    clean_bal_acc: float
    robust_utility: float
    asr: float
    validity_rate: float
    p95_latency_ms: float
    throughput_fps: float
    peak_mem_mb: float
    energy_per_flow_j: float
    service_cost: float
    raise_score: float
    valid_count: int = 0
    invalid_count: int = 0
    budget_pass_rate: float = 1.0
    bounds_pass_rate: float = 1.0
    immutable_pass_rate: float = 1.0
    relation_pass_rate: float = 1.0
    thread_count: int = 1
    batch_size: int = 1
    runtime: str = "python_sklearn_cpu"
    measurement_mode: str = "proxy"
    energy_source: str = "proxy"
    config_path: str = ""
    preprocessing_state_sha256: str = ""
    shift_group_field: str = ""
    source_split: str = ""
    target_split: str = ""
    shift_utility_drop: float = 0.0

    def to_row(self) -> dict[str, float | int | str]:
        """Return a CSV/JSON-serializable result row without changing field names."""
        return asdict(self)
