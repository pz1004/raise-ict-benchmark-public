#!/usr/bin/env python
"""Validate measured physical-edge hardware configs before Tier-E runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402


TEMPLATE_MARKERS = {
    "",
    "unknown",
    "none",
    "null",
    "replace",
    "replace_with",
    "template",
    "unmeasured",
    "cpu_proxy",
}
UNMEASURED_MODES = {"", "proxy", "none", "unknown", "unmeasured_template"}
UNMEASURED_SOURCES = {"", "proxy", "none", "unknown"}
UNTRUSTED_MEASUREMENT_MARKERS = (
    "unmeasured",
    "not_measured",
    "not-measured",
    "not measured",
    "proxy",
    "none",
    "unknown",
    "template",
    "replace",
    "dummy",
    "fake",
    "guess",
    "guessed",
)
MEASURED_MODE_TOKENS = ("measured", "meter", "sensor", "logger", "calibrated")
ENERGY_SOURCE_TOKENS = ("meter", "sensor", "logger", "power", "joule", "calibrated", "ina3221")
REQUIRED_TEXT_FIELDS = {
    "power_mode": "hardware.power_mode",
    "measurement_window": "energy.measurement_window",
}
OPTIONAL_TEXT_FIELDS = {
    "jetson_linux_release": "hardware.jetson_linux_release",
    "jetpack_release": "hardware.jetpack_release",
    "device_tree_model": "hardware.device_tree_model",
    "l4t_core_package": "hardware.l4t_core_package",
    "cuda_compiler_release": "hardware.cuda_compiler_release",
}


def _as_float(config: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = config.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_template_marker(value: object) -> bool:
    text = str(value or "").strip().lower()
    return any(marker in text for marker in TEMPLATE_MARKERS if marker)


def _has_untrusted_measurement_marker(value: object) -> bool:
    text = str(value or "").strip().lower()
    return any(marker in text for marker in UNTRUSTED_MEASUREMENT_MARKERS)


def _text_field_check(
    config: Mapping[str, Any],
    key: str,
    check_id: str,
    *,
    required: bool,
) -> dict[str, str]:
    value = str(config.get(key, "") or "").strip()
    condition = bool(value) and not _has_template_marker(value)
    if not required and key not in config:
        condition = True
        value = "<absent>"
    return _check(condition, check_id, f"{key}={value!r}")


def _check(condition: bool, check_id: str, evidence: str) -> dict[str, str]:
    return {
        "id": check_id,
        "status": "passed" if condition else "failed",
        "evidence": evidence,
    }


def validate_hardware_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return validation checks for a physical Tier-E hardware config."""
    hardware_id = str(config.get("hardware_id", "")).strip()
    device_class = str(config.get("device_class", "")).strip().lower()
    measurement_mode = str(config.get("measurement_mode", "")).strip().lower()
    energy_source = str(config.get("energy_source", "")).strip().lower()
    thread_count = int(_as_float(config, "thread_count", 0.0))
    batch_size = int(_as_float(config, "batch_size", 0.0))
    energy_per_flow = _as_float(config, "energy_per_flow_j", 0.0)
    average_power = _as_float(config, "average_power_w", 0.0)
    duration = _as_float(config, "measurement_duration_s", 0.0)
    measured_flows = _as_float(config, "measured_flows", 0.0)
    derived_energy = average_power * duration / measured_flows if measured_flows > 0.0 else 0.0
    effective_energy = energy_per_flow if energy_per_flow > 0.0 else derived_energy
    measured_mode = (
        measurement_mode not in UNMEASURED_MODES
        and not _has_untrusted_measurement_marker(measurement_mode)
        and any(token in measurement_mode for token in MEASURED_MODE_TOKENS)
    )
    measured_source = (
        energy_source not in UNMEASURED_SOURCES
        and not _has_untrusted_measurement_marker(energy_source)
        and any(token in energy_source for token in ENERGY_SOURCE_TOKENS)
    )

    checks = [
        _check(bool(hardware_id), "hardware_id.present", f"hardware_id={hardware_id!r}"),
        _check(
            not _has_template_marker(hardware_id),
            "hardware_id.not_template",
            f"hardware_id={hardware_id!r}",
        ),
        _check(
            hardware_id != "cpu_proxy",
            "hardware_id.not_cpu_proxy",
            f"hardware_id={hardware_id!r}",
        ),
        _check(
            device_class == "physical_edge",
            "device_class.physical_edge",
            f"device_class={device_class!r}",
        ),
        _check(thread_count >= 1, "profile.thread_count", f"thread_count={thread_count}"),
        _check(batch_size >= 1, "profile.batch_size", f"batch_size={batch_size}"),
        *[
            _text_field_check(config, key, check_id, required=True)
            for key, check_id in REQUIRED_TEXT_FIELDS.items()
        ],
        *[
            _text_field_check(config, key, check_id, required=False)
            for key, check_id in OPTIONAL_TEXT_FIELDS.items()
        ],
        _check(
            measured_mode,
            "energy.measurement_mode",
            f"measurement_mode={measurement_mode!r}",
        ),
        _check(
            measured_source,
            "energy.source",
            f"energy_source={energy_source!r}",
        ),
        _check(
            energy_per_flow > 0.0 or (average_power > 0.0 and duration > 0.0 and measured_flows > 0.0),
            "energy.positive_input",
            (
                f"energy_per_flow_j={energy_per_flow}, average_power_w={average_power}, "
                f"measurement_duration_s={duration}, measured_flows={measured_flows}"
            ),
        ),
        _check(
            effective_energy > 0.0,
            "energy.effective_per_flow",
            f"effective_energy_per_flow_j={effective_energy}",
        ),
    ]
    failed = [check for check in checks if check["status"] == "failed"]
    return {
        "schema_version": 1,
        "valid": not failed,
        "hardware_id": hardware_id,
        "effective_energy_per_flow_j": effective_energy,
        "checks": checks,
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Hardware YAML config to validate.")
    args = parser.parse_args()
    report = validate_hardware_config(load_yaml(args.config))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
