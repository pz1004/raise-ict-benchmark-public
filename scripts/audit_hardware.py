#!/usr/bin/env python
"""Audit visible hardware for RAISE-ICT Tier-E eligibility."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path


EDGE_MODEL_PATTERNS = [
    "raspberry pi",
    "jetson",
    "orin",
    "nano",
    "nuc",
    "coral",
    "beaglebone",
    "rockchip",
]


def run_command(command: list[str]) -> tuple[int, str]:
    """Run a short hardware-inspection command and return status plus text."""
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip() or completed.stderr.strip()


def read_text(path: str) -> str:
    """Read a system text file, returning an empty string if unavailable."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip("\x00\n ")
    except OSError:
        return ""


def parse_lscpu(text: str) -> dict[str, str]:
    """Parse key-value rows from lscpu output."""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse integer kB values from /proc/meminfo text."""
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+)", line)
        if match:
            parsed[match.group(1)] = int(match.group(2))
    return parsed


def classify_hardware(uname: str, lscpu: dict[str, str], device_model: str, nvidia: str) -> dict[str, object]:
    """Classify visible hardware as Tier-E eligible or explain why not."""
    joined = " ".join([uname, lscpu.get("Model name", ""), lscpu.get("Architecture", ""), device_model, nvidia]).lower()
    is_wsl = "microsoft-standard-wsl" in joined or "microsoft" in lscpu.get("Hypervisor vendor", "").lower()
    matched = [pattern for pattern in EDGE_MODEL_PATTERNS if pattern in joined]
    architecture = lscpu.get("Architecture", platform.machine())
    eligible = bool(matched) and not is_wsl
    reasons = []
    if is_wsl:
        reasons.append("environment is WSL/virtualized, not a directly measured edge device")
    if not matched:
        reasons.append("no known edge-device model marker was detected")
    if nvidia and "geforce" in nvidia.lower():
        reasons.append("detected GPU is a desktop GeForce class device")
    return {
        "tier_e_eligible": eligible,
        "architecture": architecture,
        "matched_edge_markers": matched,
        "reasons": reasons,
    }


def build_audit() -> dict[str, object]:
    """Collect visible local hardware evidence into the audit schema."""
    uname_code, uname = run_command(["uname", "-a"])
    lscpu_code, lscpu_text = run_command(["lscpu"])
    nvidia_code, nvidia_text = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,power.draw,memory.total",
            "--format=csv,noheader",
        ]
    )
    meminfo = read_text("/proc/meminfo")
    device_model = read_text("/proc/device-tree/model")
    lscpu = parse_lscpu(lscpu_text if lscpu_code == 0 else "")
    nvidia = nvidia_text if nvidia_code == 0 else ""
    audit = {
        "schema_version": 1,
        "hardware_id": "local_visible_hardware",
        "measurement_scope": "hardware_visibility_audit",
        "uname": uname if uname_code == 0 else "",
        "lscpu": lscpu,
        "meminfo_kb": parse_meminfo(meminfo),
        "device_tree_model": device_model,
        "nvidia_smi": nvidia,
        "classification": classify_hardware(uname, lscpu, device_model, nvidia),
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="manifests/hardware/tier_e_hardware_audit.json")
    args = parser.parse_args()
    audit = build_audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
