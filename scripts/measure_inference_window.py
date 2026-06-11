#!/usr/bin/env python
"""Run an inference-only window for software-observable energy telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from raise_ict.config import load_yaml  # noqa: E402
from raise_ict.pipeline import load_dataset_from_config, train_context  # noqa: E402


@dataclass(frozen=True)
class PowerRail:
    """Read-only sysfs power rail descriptor."""

    label: str
    channel: str
    source_dir: Path
    voltage_path: Path | None
    current_path: Path | None
    power_path: Path | None

    @property
    def source_name(self) -> str:
        return f"jetson_ina3221_{_normalize_label(self.label).lower()}"

    def read(self) -> dict[str, float | str | None]:
        """Read instantaneous rail power in watts plus raw voltage/current."""
        voltage_mv = _read_float(self.voltage_path) if self.voltage_path is not None else None
        current_ma = _read_float(self.current_path) if self.current_path is not None else None
        if self.power_path is not None:
            power_w = _read_float(self.power_path) / 1_000_000.0
        elif voltage_mv is not None and current_ma is not None:
            power_w = voltage_mv * current_ma / 1_000_000.0
        else:
            raise ValueError(f"Rail {self.label!r} has no readable power or voltage/current pair")
        return {
            "rail": self.label,
            "power_w": power_w,
            "voltage_mv": voltage_mv,
            "current_ma": current_ma,
            "source_path": str(self.source_dir),
        }


class PowerSampler:
    """Sample one power rail from a background thread during inference."""

    def __init__(self, rail: PowerRail, interval_s: float) -> None:
        self.rail = rail
        self.interval_s = max(0.001, interval_s)
        self.samples: list[dict[str, float | int | str | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._origin_s = 0.0

    def start(self, origin_s: float) -> None:
        self._origin_s = origin_s
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="jetson-power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        now = time.perf_counter()
        sample = self.rail.read()
        sample["sample_index"] = len(self.samples)
        sample["monotonic_s"] = now
        sample["elapsed_s"] = now - self._origin_s
        self.samples.append(sample)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "sample_index",
            "monotonic_s",
            "elapsed_s",
            "rail",
            "power_w",
            "voltage_mv",
            "current_ma",
            "source_path",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self, duration_s: float, measured_flows: int) -> dict[str, float | int | str]:
        powers = [float(sample["power_w"]) for sample in self.samples]
        average_power = sum(powers) / len(powers) if powers else 0.0
        return {
            "measurement_mode": "measured_onboard_sensor",
            "energy_source": self.rail.source_name,
            "rail": self.rail.label,
            "sample_count": len(powers),
            "average_power_w": average_power,
            "min_power_w": min(powers) if powers else 0.0,
            "max_power_w": max(powers) if powers else 0.0,
            "measurement_duration_s": duration_s,
            "measured_flows": measured_flows,
            "energy_per_flow_j": average_power * duration_s / max(1, measured_flows),
            "source_path": str(self.rail.source_dir),
        }


def _normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_float(path: Path) -> float:
    return float(_read_text(path))


def discover_power_rails(sysfs_root: str | Path = "/sys") -> list[PowerRail]:
    """Discover readable INA3221-style hwmon rails under a sysfs root."""
    root = Path(sysfs_root)
    candidates: list[Path] = []
    for pattern in (
        "bus/i2c/drivers/ina3221/*/hwmon/hwmon*",
        "class/hwmon/hwmon*",
    ):
        candidates.extend(root.glob(pattern))

    rails: list[PowerRail] = []
    seen: set[tuple[str, str]] = set()
    for hwmon in sorted({path.resolve() for path in candidates if path.is_dir()}):
        for label_path in sorted(hwmon.glob("in*_label")):
            match = re.fullmatch(r"in(\d+)_label", label_path.name)
            if match is None:
                continue
            channel = match.group(1)
            try:
                label = _read_text(label_path)
            except OSError:
                continue
            voltage_path = hwmon / f"in{channel}_input"
            current_path = hwmon / f"curr{channel}_input"
            power_path = hwmon / f"power{channel}_input"
            rail = PowerRail(
                label=label,
                channel=channel,
                source_dir=hwmon,
                voltage_path=voltage_path if voltage_path.exists() else None,
                current_path=current_path if current_path.exists() else None,
                power_path=power_path if power_path.exists() else None,
            )
            if rail.power_path is None and (rail.voltage_path is None or rail.current_path is None):
                continue
            key = (_normalize_label(rail.label), str(rail.source_dir))
            if key not in seen:
                seen.add(key)
                rails.append(rail)
    return rails


def _select_power_rail(sysfs_root: str | Path, selected: str) -> PowerRail:
    rails = discover_power_rails(sysfs_root)
    if not rails:
        raise ValueError(f"No readable INA3221-style power rails found under {sysfs_root}")
    target = _normalize_label(selected)
    if target == "AUTO":
        for rail in rails:
            if _normalize_label(rail.label) == "VDD_IN":
                return rail
        return rails[0]
    for rail in rails:
        if _normalize_label(rail.label) == target:
            return rail
    available = ", ".join(sorted(rail.label for rail in rails))
    raise ValueError(f"Power rail {selected!r} was not found; available rails: {available}")


def _power_rail_rows(rails: Sequence[PowerRail]) -> list[dict[str, str]]:
    return [
        {
            "label": rail.label,
            "channel": rail.channel,
            "source_dir": str(rail.source_dir),
            "voltage_path": str(rail.voltage_path or ""),
            "current_path": str(rail.current_path or ""),
            "power_path": str(rail.power_path or ""),
        }
        for rail in rails
    ]


def _configured_items(grid: Mapping[str, Any], plural_key: str, singular_key: str) -> list[Mapping[str, Any]]:
    if plural_key in grid:
        return list(grid.get(plural_key, []))
    return [grid[singular_key]] if singular_key in grid else []


def _configured_seeds(grid: Mapping[str, Any]) -> list[int]:
    if "seeds" in grid:
        return [int(seed) for seed in grid.get("seeds", [])]
    return [int(grid.get("seed", 0))]


def _select_by_id(items: Sequence[Mapping[str, Any]], key: str, selected: str | None) -> Mapping[str, Any]:
    if not items:
        raise ValueError(f"No configured {key} entries found")
    if selected is None:
        return items[0]
    for item in items:
        if str(item.get(key, "")) == selected:
            return item
    available = ", ".join(str(item.get(key, "")) for item in items)
    raise ValueError(f"{selected!r} is not configured for {key}; available: {available}")


def _measurement_config(
    grid: Mapping[str, Any],
    config_path: str,
    dataset_id: str | None,
    model_id: str | None,
    seed: int | None,
) -> dict[str, Any]:
    dataset_cfg = dict(_select_by_id(_configured_items(grid, "datasets", "dataset"), "dataset_id", dataset_id))
    model_cfg = dict(_select_by_id(_configured_items(grid, "models", "model"), "model_id", model_id))
    selected_seed = _configured_seeds(grid)[0] if seed is None else seed
    dataset_cfg["seed"] = selected_seed
    return {
        "seed": selected_seed,
        "profile_repeats": 1,
        "test_size": grid.get("test_size", 0.3),
        "split_id": dataset_cfg.get("split_id", grid.get("split_id", "energy_window_split")),
        "dataset": dataset_cfg,
        "model": model_cfg,
        "hardware": grid.get("hardware", {"hardware_id": "measurement_window"}),
        "preprocessing": grid.get("preprocessing", {}),
        "config_path": config_path,
        "scoring": grid.get("scoring", {}),
    }


def run_inference_window(
    config: Mapping[str, Any],
    *,
    seconds: float,
    warmup_iterations: int,
    start_delay_s: float,
    power_rail: PowerRail | None = None,
    power_sample_interval_s: float = 0.1,
    power_log_out: str | None = None,
) -> dict[str, Any]:
    """Train outside the measured window, then loop over model prediction."""
    print("Preparing dataset, preprocessing, and model outside the energy window.", file=sys.stderr, flush=True)
    frame = load_dataset_from_config(config["dataset"])
    context = train_context(config, frame)
    features = context.x_test
    flows_per_iteration = len(features)
    for _ in range(max(0, warmup_iterations)):
        context.model.predict(features)

    if start_delay_s > 0.0:
        print(
            f"Prepare the software telemetry marker now; energy window starts in {start_delay_s:.1f}s.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(start_delay_s)

    print(
        "ENERGY_WINDOW_START "
        f"dataset={context.dataset} model={context.model_id} seed={context.seed} "
        f"flows_per_iteration={flows_per_iteration} target_seconds={seconds}",
        file=sys.stderr,
        flush=True,
    )

    power_sampler = PowerSampler(power_rail, power_sample_interval_s) if power_rail is not None else None
    start = time.perf_counter()
    if power_sampler is not None:
        power_sampler.start(start)
    iterations = 0
    while iterations == 0 or time.perf_counter() - start < seconds:
        context.model.predict(features)
        iterations += 1
    elapsed = time.perf_counter() - start
    if power_sampler is not None:
        power_sampler.stop()
    measured_flows = flows_per_iteration * iterations
    print(
        f"ENERGY_WINDOW_END elapsed_s={elapsed:.6f} measured_flows={measured_flows}",
        file=sys.stderr,
        flush=True,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": context.dataset,
        "split_id": context.split_id,
        "seed": context.seed,
        "model_id": context.model_id,
        "flows_per_iteration": flows_per_iteration,
        "warmup_iterations": max(0, warmup_iterations),
        "measured_iterations": iterations,
        "measured_flows": measured_flows,
        "measurement_duration_s": elapsed,
        "throughput_fps": measured_flows / elapsed if elapsed > 0.0 else 0.0,
        "recommended_hardware_fields": {
            "measurement_window": (
                f"inference-only predict loop: {context.dataset}/{context.model_id}/"
                f"seed{context.seed}, {iterations} iterations after {max(0, warmup_iterations)} warmups"
            ),
            "measurement_duration_s": elapsed,
            "measured_flows": measured_flows,
        },
    }
    if power_sampler is not None:
        if power_log_out:
            power_sampler.write_csv(Path(power_log_out))
        power_summary = power_sampler.summary(elapsed, measured_flows)
        if power_log_out:
            power_summary["sample_log_path"] = str(Path(power_log_out))
        report["onboard_power"] = power_summary
        report["recommended_hardware_fields"].update(
            {
                "measurement_mode": power_summary["measurement_mode"],
                "energy_source": power_summary["energy_source"],
                "average_power_w": power_summary["average_power_w"],
                "measurement_duration_s": elapsed,
                "measured_flows": measured_flows,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--start-delay-s", type=float, default=5.0)
    parser.add_argument("--list-power-rails", action="store_true")
    parser.add_argument("--power-rail", default="")
    parser.add_argument("--power-sysfs-root", default="/sys")
    parser.add_argument("--power-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--power-log-out", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.list_power_rails:
        print(json.dumps(_power_rail_rows(discover_power_rails(args.power_sysfs_root)), indent=2))
        return
    if not args.config:
        parser.error("--config is required unless --list-power-rails is used")

    grid = load_yaml(args.config)
    config = _measurement_config(grid, args.config, args.dataset_id, args.model_id, args.seed)
    power_rail = _select_power_rail(args.power_sysfs_root, args.power_rail) if args.power_rail else None
    report = run_inference_window(
        config,
        seconds=max(0.0, args.seconds),
        warmup_iterations=args.warmup_iterations,
        start_delay_s=max(0.0, args.start_delay_s),
        power_rail=power_rail,
        power_sample_interval_s=max(0.001, args.power_sample_interval_s),
        power_log_out=args.power_log_out or None,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
