"""Lightweight inference profiling."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np


def _energy_per_flow(hardware: dict[str, Any], n_flows: int, repeats: int) -> float:
    """Compute joules per flow from declared measurement metadata."""
    if not hardware:
        return 0.0
    if "energy_per_flow_j" in hardware:
        return float(hardware["energy_per_flow_j"])
    power_w = hardware.get("average_power_w")
    duration_s = hardware.get("measurement_duration_s")
    if power_w is None or duration_s is None:
        return 0.0
    measured_flows = int(hardware.get("measured_flows", max(1, n_flows * max(1, repeats))))
    return float(power_w) * float(duration_s) / max(1, measured_flows)


def profile_predict(
    model: Any,
    features: Any,
    repeats: int = 3,
    hardware: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    """Measure prediction latency and memory with optional energy metadata."""
    hardware = hardware or {}
    latencies_ms: list[float] = []
    tracemalloc.start()
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        model.predict(features)
        elapsed = time.perf_counter() - start
        latencies_ms.append(1000.0 * elapsed / max(1, len(features)))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p95 = float(np.percentile(latencies_ms, 95))
    throughput = float(1000.0 / p95) if p95 > 0 else 0.0
    energy = _energy_per_flow(hardware, len(features), repeats)
    return {
        "p95_latency_ms": p95,
        "throughput_fps": throughput,
        "peak_mem_mb": float(peak / (1024 * 1024)),
        "energy_per_flow_j": energy,
        "thread_count": int(hardware.get("thread_count", hardware.get("threads", 1))),
        "batch_size": int(hardware.get("batch_size", 1)),
        "runtime": str(hardware.get("runtime", "python_sklearn_cpu")),
        "measurement_mode": str(hardware.get("measurement_mode", "proxy")),
        "energy_source": str(hardware.get("energy_source", "proxy" if energy == 0.0 else "declared")),
    }
