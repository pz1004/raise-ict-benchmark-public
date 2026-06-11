#!/usr/bin/env bash
# Overnight Jetson Orin NX Super Tier-E run for RAISE-ICT.
#
# Default behavior:
# - Capture hardware/software evidence.
# - Discover INA3221 rails and require VDD_IN.
# - Download bounded Core4 dataset mirrors if missing.
# - Run a short synthetic INA3221 smoke window.
# - Run a 9-hour real-data inference-only INA3221 VDD_IN calibration.
# - Generate a measured onboard-sensor hardware YAML.
# - Validate the YAML and run the Tier-E Core4 orchestrator.
#
# Useful overrides:
#   RAISE_ICT_LONG_SECONDS=32400
#   RAISE_ICT_POWER_SAMPLE_INTERVAL_S=1.0
#   RAISE_ICT_HARDWARE_CONFIG=configs/hardware/jetson_orin_nx_super_measured_overnight.yaml
#   RAISE_ICT_DOWNLOAD_DATA=0
#   RAISE_ICT_RUN_CORE4=0
#   RAISE_ICT_POWER_MODE=MAXN_SUPER

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
RUN_ID="${RAISE_ICT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RAISE_ICT_RUN_DIR:-logs/overnight_${RUN_ID}}"
MAIN_LOG="${RUN_DIR}/overnight.log"

POWER_SYSFS_ROOT="${RAISE_ICT_POWER_SYSFS_ROOT:-/sys}"
POWER_RAIL="${RAISE_ICT_POWER_RAIL:-VDD_IN}"
POWER_SAMPLE_INTERVAL_S="${RAISE_ICT_POWER_SAMPLE_INTERVAL_S:-1.0}"
SMOKE_SECONDS="${RAISE_ICT_SMOKE_SECONDS:-120}"
LONG_SECONDS="${RAISE_ICT_LONG_SECONDS:-32400}"
LONG_CONFIG="${RAISE_ICT_LONG_CONFIG:-configs/experiments/tier_p_cse_cic_ids2018.yaml}"
LONG_DATASET_ID="${RAISE_ICT_LONG_DATASET_ID:-CSE-CIC-IDS2018}"
LONG_MODEL_ID="${RAISE_ICT_LONG_MODEL_ID:-extra_trees}"
LONG_SEED="${RAISE_ICT_LONG_SEED:-0}"
HARDWARE_CONFIG="${RAISE_ICT_HARDWARE_CONFIG:-configs/hardware/jetson_orin_nx_super_measured_overnight.yaml}"
DOWNLOAD_DATA="${RAISE_ICT_DOWNLOAD_DATA:-1}"
RUN_CORE4="${RAISE_ICT_RUN_CORE4:-1}"
RUN_ANALYSIS="${RAISE_ICT_RUN_ANALYSIS:-1}"
START_TEGRASTATS="${RAISE_ICT_START_TEGRASTATS:-1}"
POWER_MODE_FALLBACK="${RAISE_ICT_POWER_MODE:-MAXN_SUPER}"

RAIL_JSON="manifests/hardware/jetson_power_rails_${RUN_ID}.json"
SMOKE_JSON="manifests/hardware/jetson_inference_energy_smoke_${RUN_ID}.json"
SMOKE_POWER_CSV="manifests/hardware/jetson_power_ina3221_vdd_in_smoke_${RUN_ID}.csv"
LONG_JSON="manifests/hardware/jetson_inference_energy_window_${RUN_ID}.json"
LONG_POWER_CSV="manifests/hardware/jetson_power_ina3221_vdd_in_${RUN_ID}.csv"
TEGRALOG="${RUN_DIR}/tegrastats_${RUN_ID}.log"
TEGRAPID=""

mkdir -p \
  "$RUN_DIR" \
  logs \
  manifests/hardware \
  manifests/dataset_hashes \
  manifests/splits \
  manifests/feature_schemas \
  manifests/completion \
  results/raw \
  results/tables \
  results/figures \
  results/analysis

exec > >(tee -a "$MAIN_LOG") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

section() {
  printf '\n[%s] == %s ==\n' "$(timestamp)" "$*"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

run() {
  log "+ $*"
  "$@"
}

capture_cmd() {
  local out="$1"
  shift
  log "+ $* > ${out}"
  if "$@" >"$out" 2>&1; then
    log "wrote ${out}"
  else
    log "warning: command failed while writing ${out}"
  fi
}

stop_tegrastats() {
  if [[ -n "$TEGRAPID" ]] && kill -0 "$TEGRAPID" 2>/dev/null; then
    log "stopping tegrastats pid=${TEGRAPID}"
    kill "$TEGRAPID" 2>/dev/null || true
    wait "$TEGRAPID" 2>/dev/null || true
  fi
}

on_exit() {
  local status=$?
  stop_tegrastats
  if [[ "$status" -eq 0 ]]; then
    log "overnight run completed successfully"
  else
    log "overnight run failed with exit status ${status}"
  fi
  log "main log: ${MAIN_LOG}"
  exit "$status"
}

trap on_exit EXIT

start_tegrastats() {
  if [[ "$START_TEGRASTATS" != "1" ]]; then
    log "tegrastats disabled by RAISE_ICT_START_TEGRASTATS=${START_TEGRASTATS}"
    return
  fi
  if ! command -v tegrastats >/dev/null 2>&1; then
    log "tegrastats not found; continuing without thermal/frequency log"
    return
  fi
  log "starting tegrastats log at ${TEGRALOG}"
  if sudo -n true >/dev/null 2>&1; then
    sudo -n tegrastats --interval 1000 --logfile "$TEGRALOG" &
  else
    tegrastats --interval 1000 --logfile "$TEGRALOG" &
  fi
  TEGRAPID="$!"
  echo "$TEGRAPID" >"${RUN_DIR}/tegrastats.pid"
}

verify_energy_json() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
power = report.get("onboard_power") or {}
required = {
    "measurement_duration_s": report.get("measurement_duration_s", 0),
    "measured_flows": report.get("measured_flows", 0),
    "onboard_power.average_power_w": power.get("average_power_w", 0),
    "onboard_power.energy_per_flow_j": power.get("energy_per_flow_j", 0),
}
bad = {key: value for key, value in required.items() if float(value or 0) <= 0.0}
if bad:
    raise SystemExit(f"{path} has non-positive measurement fields: {bad}")
print(json.dumps(required, indent=2))
PY
}

write_hardware_config() {
  "$PYTHON" - "$LONG_JSON" "$HARDWARE_CONFIG" "$RUN_ID" "$POWER_MODE_FALLBACK" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

import yaml

energy_json = Path(sys.argv[1])
hardware_config = Path(sys.argv[2])
run_id = sys.argv[3]
power_mode_fallback = sys.argv[4]

report = json.loads(energy_json.read_text(encoding="utf-8"))
power = report.get("onboard_power") or {}
recommended = report.get("recommended_hardware_fields") or {}


def read_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace").strip()


def clean_optional(value: str) -> str:
    value = " ".join((value or "").replace("\x00", "").split())
    lowered = value.lower()
    if not value or any(marker in lowered for marker in ("unknown", "replace", "template", "none", "null")):
        return ""
    return value


def parse_jetson_linux_release(text: str) -> str:
    match = re.search(r"R(\d+).*REVISION:\s*([0-9.]+)", text)
    if not match:
        return ""
    return f"R{match.group(1)}.{match.group(2)}"


def parse_package_version(text: str, package_name: str) -> str:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == package_name:
            return parts[1]
    return ""


def parse_cuda_release(text: str) -> str:
    match = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", text)
    if match:
        return match.group(2)
    return ""


def parse_power_mode(*texts: str) -> str:
    for text in texts:
        for pattern in (r"NV Power Mode:\s*([^\n]+)", r"Power Mode:\s*([^\n]+)"):
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
    return power_mode_fallback


nv_tegra = read_text("manifests/hardware/jetson_nv_tegra_release.txt")
packages = read_text("manifests/hardware/jetson_l4t_jetpack_packages.txt")
nvcc = read_text("manifests/hardware/jetson_nvcc_version.txt")
device_tree_model = read_text("manifests/hardware/jetson_device_tree_model.txt")
nvpmodel = read_text("manifests/hardware/jetson_nvpmodel_selected.txt")
clocks = read_text("manifests/hardware/jetson_clocks_selected.txt")

average_power_w = float(recommended.get("average_power_w", power.get("average_power_w", 0.0)) or 0.0)
measurement_duration_s = float(
    recommended.get("measurement_duration_s", report.get("measurement_duration_s", 0.0)) or 0.0
)
measured_flows = int(float(recommended.get("measured_flows", report.get("measured_flows", 0)) or 0))

config = {
    "hardware_id": f"jetson_orin_nx_super_ina3221_{run_id.lower()}",
    "device_class": "physical_edge",
    "runtime": "python_sklearn_cpu",
    "thread_count": int(os.environ.get("RAISE_ICT_THREAD_COUNT") or (os.cpu_count() or 1)),
    "batch_size": 1,
    "power_mode": parse_power_mode(nvpmodel, clocks),
    "measurement_mode": "measured_onboard_sensor",
    "energy_source": "jetson_ina3221_vdd_in",
    "measurement_window": (
        f"{measurement_duration_s:.3f}s inference-only predict loop with INA3221 VDD_IN sampling; "
        f"run_id={run_id}"
    ),
    "average_power_w": average_power_w,
    "measurement_duration_s": measurement_duration_s,
    "measured_flows": measured_flows,
    "notes": (
        "Generated by scripts/jetson_overnight_tier_e.sh from software-observed "
        "Jetson INA3221 VDD_IN module-power telemetry. This is not external wall-power energy."
    ),
}

optional_values = {
    "jetson_linux_release": parse_jetson_linux_release(nv_tegra),
    "jetpack_release": parse_package_version(packages, "nvidia-jetpack"),
    "l4t_core_package": parse_package_version(packages, "nvidia-l4t-core"),
    "cuda_compiler_release": parse_cuda_release(nvcc),
    "device_tree_model": device_tree_model,
}
for key, value in optional_values.items():
    value = clean_optional(value)
    if value:
        config[key] = value

hardware_config.parent.mkdir(parents=True, exist_ok=True)
if hardware_config.exists():
    backup = hardware_config.with_suffix(hardware_config.suffix + f".bak_{run_id}")
    backup.write_text(hardware_config.read_text(encoding="utf-8"), encoding="utf-8")
config_text = yaml.safe_dump(config, sort_keys=False)
hardware_config.write_text(config_text, encoding="utf-8")
print(config_text)
PY
}

section "Start"
log "root: ${ROOT_DIR}"
log "run id: ${RUN_ID}"
log "main log: ${MAIN_LOG}"
log "long measurement seconds: ${LONG_SECONDS}"
log "long measurement config: ${LONG_CONFIG}"
log "hardware config output: ${HARDWARE_CONFIG}"

section "Preflight"
run "$PYTHON" --version
run "$PYTHON" -m compileall -q src scripts
capture_cmd "${RUN_DIR}/git_status.txt" git status --short
capture_cmd "${RUN_DIR}/git_head.txt" git rev-parse HEAD

section "Hardware Evidence"
capture_cmd manifests/hardware/jetson_nv_tegra_release.txt cat /etc/nv_tegra_release
capture_cmd manifests/hardware/jetson_l4t_jetpack_packages.txt dpkg-query -W nvidia-l4t-core nvidia-jetpack
capture_cmd manifests/hardware/jetson_nvcc_version.txt nvcc --version
capture_cmd manifests/hardware/jetson_uname.txt uname -a
if [[ -r /proc/device-tree/model ]]; then
  (tr -d '\000' </proc/device-tree/model; echo) >manifests/hardware/jetson_device_tree_model.txt
  log "wrote manifests/hardware/jetson_device_tree_model.txt"
else
  log "warning: /proc/device-tree/model is not readable"
fi
capture_cmd manifests/hardware/jetson_nproc.txt nproc
capture_cmd manifests/hardware/jetson_df_h.txt df -h
capture_cmd manifests/hardware/jetson_free_h.txt free -h
if sudo -n true >/dev/null 2>&1; then
  capture_cmd manifests/hardware/jetson_nvpmodel_selected.txt sudo -n nvpmodel -q --verbose
  capture_cmd manifests/hardware/jetson_clocks_selected.txt sudo -n jetson_clocks --show
else
  log "sudo password is not cached; skipping nvpmodel and jetson_clocks capture"
fi

section "Start Telemetry"
start_tegrastats

section "INA3221 Rail Discovery"
log "+ ${PYTHON} scripts/measure_inference_window.py --list-power-rails --power-sysfs-root ${POWER_SYSFS_ROOT} | tee ${RAIL_JSON}"
"$PYTHON" scripts/measure_inference_window.py \
  --list-power-rails \
  --power-sysfs-root "$POWER_SYSFS_ROOT" | tee "$RAIL_JSON"
"$PYTHON" - "$RAIL_JSON" "$POWER_RAIL" <<'PY'
import json
import sys
from pathlib import Path

rails = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = sys.argv[2]
labels = [str(item.get("label", "")) for item in rails]
if target not in labels:
    raise SystemExit(f"{target} not found in rails: {labels}")
print(f"found {target}; available rails={labels}")
PY

section "Synthetic Energy Smoke"
run "$PYTHON" scripts/measure_inference_window.py \
  --config configs/experiments/tier_s.yaml \
  --seconds "$SMOKE_SECONDS" \
  --warmup-iterations 3 \
  --start-delay-s 5 \
  --power-rail "$POWER_RAIL" \
  --power-sysfs-root "$POWER_SYSFS_ROOT" \
  --power-sample-interval-s "$POWER_SAMPLE_INTERVAL_S" \
  --power-log-out "$SMOKE_POWER_CSV" \
  --out "$SMOKE_JSON"
verify_energy_json "$SMOKE_JSON"

if [[ "$DOWNLOAD_DATA" == "1" ]]; then
  section "Dataset Downloads"
  run "$PYTHON" scripts/download_datasets.py --datasets UNSW-NB15 TON_IoT
  run "$PYTHON" scripts/download_datasets.py \
    --datasets CICIDS2017 \
    --manifest manifests/dataset_hashes/cicids2017_download_manifest.json
  run "$PYTHON" scripts/download_datasets.py \
    --datasets CSE-CIC-IDS2018 \
    --manifest manifests/dataset_hashes/cse_cic_ids2018_download_manifest.json
  run "$PYTHON" scripts/merge_artifacts.py \
    manifests/dataset_hashes/download_manifest.json \
    manifests/dataset_hashes/cicids2017_download_manifest.json \
    manifests/dataset_hashes/cse_cic_ids2018_download_manifest.json \
    --out manifests/dataset_hashes/tier_p_core4_download_manifest.json
else
  section "Dataset Downloads"
  log "skipped by RAISE_ICT_DOWNLOAD_DATA=${DOWNLOAD_DATA}"
fi

section "Long Real-Data Energy Window"
run "$PYTHON" scripts/measure_inference_window.py \
  --config "$LONG_CONFIG" \
  --dataset-id "$LONG_DATASET_ID" \
  --model-id "$LONG_MODEL_ID" \
  --seed "$LONG_SEED" \
  --seconds "$LONG_SECONDS" \
  --warmup-iterations 3 \
  --start-delay-s 5 \
  --power-rail "$POWER_RAIL" \
  --power-sysfs-root "$POWER_SYSFS_ROOT" \
  --power-sample-interval-s "$POWER_SAMPLE_INTERVAL_S" \
  --power-log-out "$LONG_POWER_CSV" \
  --out "$LONG_JSON"
verify_energy_json "$LONG_JSON"

section "Generate Hardware Config"
write_hardware_config
log "+ ${PYTHON} scripts/validate_hardware_config.py --config ${HARDWARE_CONFIG} | tee ${RUN_DIR}/hardware_validation.json"
"$PYTHON" scripts/validate_hardware_config.py --config "$HARDWARE_CONFIG" | tee "${RUN_DIR}/hardware_validation.json"

if [[ "$RUN_CORE4" == "1" ]]; then
  section "Tier-E Core4"
  run "$PYTHON" scripts/run_tier_e_core4.py \
    --hardware-config "$HARDWARE_CONFIG"
else
  section "Tier-E Core4"
  log "skipped by RAISE_ICT_RUN_CORE4=${RUN_CORE4}"
fi

if [[ "$RUN_CORE4" == "1" && "$RUN_ANALYSIS" == "1" ]]; then
  section "Tier-E Analysis"
  run "$PYTHON" scripts/analyze_results.py \
    --raw results/tables/tier_e_core4/table_raw_results.csv \
    --summary results/tables/tier_e_core4/table_main_results.csv \
    --out results/analysis/tier_e_core4 \
    --attack-threat a1_constrained_score_search \
    --label "Tier-E Core4 Jetson Orin NX" \
    --scope-note "Jetson Orin NX physical-edge run with INA3221 software-observed energy metadata" \
    --split-manifest manifests/splits/tier_e_core4_split_manifest.csv \
    --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json
fi

section "Outputs"
log "run directory: ${RUN_DIR}"
log "rail discovery: ${RAIL_JSON}"
log "smoke JSON: ${SMOKE_JSON}"
log "long energy JSON: ${LONG_JSON}"
log "long power CSV: ${LONG_POWER_CSV}"
log "hardware config: ${HARDWARE_CONFIG}"
log "main results: results/tables/tier_e_core4/table_main_results.csv"
log "completion audit: manifests/completion/benchmark_completion_audit_strict_tier_e.json"
