# RAISE-ICT Jetson Orin NX In Super Mode Experiment Guide

This guide gives a step-by-step path for rerunning the RAISE-ICT Core4 benchmark on a physical NVIDIA Jetson Orin NX target with NVIDIA Super Mode power profiles enabled. It covers hardware setup, software setup, dataset downloads, benchmark commands, software-observed energy metadata, and completion checks.

Naming boundary: NVIDIA markets a Jetson Orin Nano Super Developer Kit, while Jetson Orin NX uses Super Mode power profiles. This guide therefore treats "Jetson Orin NX Super" as shorthand for "Jetson Orin NX 8GB or 16GB module running a Super Mode power profile," not as a separate module SKU. Verify the actual module with `/proc/device-tree/model` before claiming results.

## 1. Scope And Evidence Target

The goal is a Tier-E RAISE-ICT run:

- Same Core4 datasets and threats as the current public evidence path.
- Physical edge hardware visible to `scripts/audit_hardware.py`.
- Non-CPU-proxy `hardware_id` in every result row.
- Positive software-observed `energy_per_flow_j`.
- Profile manifest at `manifests/hardware/tier_e_profile_manifest.json`.
- Strict completion audit passing with `scripts/check_completion.py --require-tier-e --strict`.

Important boundary: the current harness uses scikit-learn CPU inference. Jetson CUDA, TensorRT, and DLA acceleration are not used by this code path yet. The Jetson value in this run is physical edge-device profiling for the existing RAISE-ICT reference path, not GPU-accelerated model inference.

Stop conditions before any Tier-E claim:

- Super Mode or the declared power mode is not visible in `nvpmodel`.
- The run uses `cpu_proxy` or an unmeasured template hardware ID.
- Energy metadata comes from a guessed value rather than software-readable Jetson power telemetry or another documented measurement path.
- `scripts/check_completion.py --require-tier-e --strict` reports `complete=false`.

## 2. Official Jetson Facts To Record

Check these facts before running the benchmark:

- Date-sensitive release boundary checked on June 11, 2026: NVIDIA's current downloads page lists JetPack 7.2 with Jetson Linux 39.2 as the latest Jetson release. This guide still targets the JetPack 6.2.1 / Jetson Linux 36.4.x family as the conservative benchmark baseline because NVIDIA documents JetPack 6.2.1 as the latest production JetPack 6 release, and the Orin NX Super Mode instructions used here were introduced in JetPack 6.2. If you choose JetPack 7.x, record it as a separate software condition and rerun the smoke path before making Tier-E claims.
- NVIDIA lists Jetson Orin NX modules as up to 157 TOPS with configurable 10 W to 40 W power.
- NVIDIA JetPack 6.2 introduced Super Mode for Jetson Orin Nano and Jetson Orin NX production modules. For Orin NX 8GB and 16GB, the new modes include 40 W and labels such as `MAXN_SUPER`.
- NVIDIA states the new Super power modes require a new flashing configuration. Do not assume a normal runtime package update exposes Super Mode.
- NVIDIA JetPack 6.2.1 is documented with Jetson Linux 36.4.4, Linux kernel 5.15, and an Ubuntu 22.04 root file system. Apt-updated devices can report later 36.4.x package revisions, for example `nvidia-l4t-core 36.4.7-20250918154033` with `nvidia-jetpack 6.2.1+b38`. Record the exact package versions rather than normalizing them to 36.4.4.
- NVIDIA Jetson Linux documentation says `nvpmodel` is the supported command for displaying and changing power mode, and a reboot can be required after changing some modes.

Useful official pages:

- Jetson Orin product/spec page: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- JetPack current downloads page: https://developer.nvidia.com/embedded/jetpack/downloads
- JetPack 6.2.1 page: https://developer.nvidia.com/embedded/jetpack-sdk-621
- JetPack 6.2 Super Mode announcement: https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/
- JetPack SDK documentation: https://docs.nvidia.com/jetson/jetpack/
- Jetson Linux power/performance guide: https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
- Jetson Linux archive: https://developer.nvidia.com/embedded/jetson-linux-archive

## 3. Hardware Checklist

Use this minimum setup:

- Jetson Orin NX 16GB preferred. Jetson Orin NX 8GB can run the bounded Core4 path, but enable swap and expect longer runtime.
- Super Mode capable flash image/configuration.
- NVMe SSD strongly preferred. Keep at least 40 GB free for datasets, environments, logs, and regenerated artifacts.
- Active cooling. Avoid passive heatsink-only runs in 40 W or `MAXN_SUPER`.
- Software-readable Jetson power telemetry. For the default no-external-meter path, use INA3221 `VDD_IN` through sysfs and archive the raw CSV log.
- Stable power supply sized for the selected mode and carrier board.
- Wired Ethernet or reliable high-speed network for dataset downloads.
- SSH access plus `tmux` or `screen` for long runs.

Pre-board handoff checks on the workstation:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src scripts
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
python scripts/run_tier_e_core4.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_template.yaml \
  --dry-run >/tmp/raise_ict_tier_e_dry_run.json
du -sh .
```

Do not expect current `data/raw`, `results`, `manifests`, or `logs` directories to travel through a normal git clone. They are generated or local-evidence paths and are ignored by `.gitignore`. On the Jetson, rerun dataset downloads and regenerate manifests/results, or intentionally copy an archived evidence bundle if you are reproducing a completed run.

## 4. Flash Jetson And Enable Super Mode

Use NVIDIA SDK Manager or your lab's production flashing workflow. For reproducibility, record the exact JetPack and Jetson Linux release. JetPack 6.2.1 / Jetson Linux 36.4.x is a conservative target for this Python/scikit-learn benchmark because it uses Ubuntu 22.04. Newer major Jetson Linux releases may work, but record them explicitly and treat them as a different software condition in the result discussion.

After first boot:

```bash
mkdir -p manifests/hardware logs

cat /etc/nv_tegra_release | tee manifests/hardware/jetson_nv_tegra_release.txt
dpkg-query -W nvidia-l4t-core nvidia-jetpack | tee manifests/hardware/jetson_l4t_jetpack_packages.txt
nvcc --version | tee manifests/hardware/jetson_nvcc_version.txt
uname -a | tee manifests/hardware/jetson_uname.txt
(tr -d '\000' </proc/device-tree/model; echo) | tee manifests/hardware/jetson_device_tree_model.txt
nproc | tee manifests/hardware/jetson_nproc.txt
df -h | tee manifests/hardware/jetson_df_h.txt
free -h | tee manifests/hardware/jetson_free_h.txt
```

The `echo` after `/proc/device-tree/model` is intentional. Device-tree strings may not end with a newline; without `echo`, the model string can concatenate with the next command's output in terminal logs.

Observed preflight values from the current anonymous-jetson board log:

- Device-tree model: `NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super`.
- Jetson Linux: `R36.4.7`, with `REVISION: 4.7`.
- Current `nvpmodel` mode: mode ID `0`, label `MAXN_SUPER`.
- CPU cores online: 8.
- Root filesystem: 116 GB total, 93 GB available.
- Memory and swap: 15 GiB RAM, 7.6 GiB swap.

These values satisfy the board identity, power-mode visibility, thread-count, storage, and memory preconditions for the bounded Core4 path. They do not satisfy the energy-evidence requirement; that still requires software-readable power telemetry, such as INA3221 `VDD_IN`, over an inference-only window.

List power modes and choose exactly one declared mode for the paper run:

```bash
sudo nvpmodel -q --verbose | tee manifests/hardware/jetson_nvpmodel_available.txt
```

Find the mode ID for `40W`, `MAXN_SUPER`, or the mode your lab chooses. Then set it:

```bash
export RAISE_NVP_MODE=<MODE_ID_FOR_40W_OR_MAXN_SUPER>
sudo nvpmodel -m "$RAISE_NVP_MODE"
sudo reboot
```

After reboot:

```bash
mkdir -p manifests/hardware logs
sudo nvpmodel -q --verbose | tee manifests/hardware/jetson_nvpmodel_selected.txt
sudo jetson_clocks
sudo jetson_clocks --show | tee manifests/hardware/jetson_clocks_selected.txt
```

If Super modes are not visible, stop. Recheck the flash configuration and JetPack release before running benchmark claims. If `nvpmodel` asks for a reboot after changing modes, reboot and rerun the selected-mode and `jetson_clocks --show` capture.

Treat `jetson_clocks --show` as operating-state evidence, not as an energy measurement. If the reported `CurrentFreq` values are below the listed `MaxFreq` values, record that exact state in the evidence bundle and do not describe the run as fixed-frequency unless you deliberately rerun `sudo jetson_clocks` and archive a matching post-command `--show` output.

## 5. System Packages

Install operating-system packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential curl git git-lfs jq tmux wget unzip htop \
  libopenblas-dev gfortran pkg-config

git lfs install
```

Optional but recommended on 8GB modules:

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon --show
```

To make swap persistent, add this line to `/etc/fstab` only after confirming the swap file works:

```text
/swapfile none swap sw 0 0
```

## 6. Python Environment

The repository supports Python `>=3.10`. JetPack 6.x Ubuntu 22.04 often provides Python 3.10 by default, and that version is acceptable for the current RAISE-ICT harness.

Recommended on Jetson: Miniforge for Linux aarch64.

```bash
cd "$HOME"
wget -O Miniforge3-Linux-aarch64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -y -n raise-ict python=3.10 pip \
  numpy pandas pyarrow scipy scikit-learn matplotlib pyyaml tabulate pytest
conda activate raise-ict
```

Clone or copy the repository, then install it:

```bash
cd "$HOME"
test -n "$RAISE_ICT_REPO_URL"
git clone "$RAISE_ICT_REPO_URL" ml-benchmark-2
cd "$HOME/ml-benchmark-2"
python -m pip install -e .
```

Set `RAISE_ICT_REPO_URL` to your actual Git remote before running the clone command. If the repository is not on a remote yet, copy it from the workstation instead:

```bash
rsync -av --exclude '.venv' --exclude 'data/raw' --exclude 'data/processed' \
  --exclude 'results' --exclude 'manifests' --exclude 'logs' \
  /path/to/ml-benchmark-2/ jetson-user@jetson-host:~/ml-benchmark-2/
```

If the repository is already copied to the Jetson:

```bash
cd /path/to/ml-benchmark-2
conda activate raise-ict
python -m pip install -e .
```

Verify imports and tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src scripts
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## 7. Dataset Downloads

Raw datasets are not committed. Obtain datasets from the official provider pages first; see `DATASET_USAGE.md` for source and citation notes. The commands below use third-party mirror URLs only as an explicit opt-in convenience path and write hash manifests for the files actually downloaded.

```bash
cd /path/to/ml-benchmark-2
conda activate raise-ict

python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets UNSW-NB15 TON_IoT

python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets CICIDS2017 \
  --manifest manifests/dataset_hashes/cicids2017_download_manifest.json

python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets CSE-CIC-IDS2018 \
  --manifest manifests/dataset_hashes/cse_cic_ids2018_download_manifest.json

python scripts/merge_artifacts.py \
  manifests/dataset_hashes/download_manifest.json \
  manifests/dataset_hashes/cicids2017_download_manifest.json \
  manifests/dataset_hashes/cse_cic_ids2018_download_manifest.json \
  --out manifests/dataset_hashes/tier_p_core4_download_manifest.json
```

Expected local raw data footprint for the current bounded Core4 optional mirror path is about 1.3 GB. Keep extra disk space for environments and result artifacts.

Check local files:

```bash
du -sh data/raw
python -m json.tool manifests/dataset_hashes/tier_p_core4_download_manifest.json | head -80
```

## 8. Measurement Model Used By This Harness

RAISE-ICT does not automatically convert `tegrastats` output into energy. The profiler records latency and memory inside Python, and it records energy from the hardware config you provide. In this project, the Jetson energy path uses software-readable onboard telemetry only.

The default evidence path is:

1. Run a dedicated inference-only calibration window after datasets are downloaded and the Python environment is working.
2. Sample the Jetson INA3221 `VDD_IN` rail during that same prediction-only interval.
3. Record the window duration, average software-observed power, and number of inference flows.
4. Fill `average_power_w`, `measurement_duration_s`, and `measured_flows` in the hardware YAML.

Do not use a training-plus-download wall-clock window for `measurement_duration_s`. The result field is energy per inference flow, so the measurement window must correspond to model prediction work. `tegrastats` is useful for thermal throttling and clock evidence, but it is not treated as energy evidence.

NVIDIA documents that Jetson Orin NX modules expose INA3221 power-monitor rails through sysfs. For Jetson Orin NX and Orin Nano, the `1-0040` INA3221 monitor includes channel 1 `VDD_IN`, described as total module power. This is not identical to external board-input power, because carrier-board and adapter losses may be outside the module rail. It is still a usable fallback if the paper and hardware YAML label it as onboard-sensor/module-power measurement.

List readable rails:

```bash
python scripts/measure_inference_window.py \
  --list-power-rails \
  --power-sysfs-root /sys | tee manifests/hardware/jetson_power_rails.json
```

Run the same inference-only measurement with onboard `VDD_IN` sampling:

```bash
python scripts/measure_inference_window.py \
  --config configs/experiments/tier_p_cse_cic_ids2018.yaml \
  --dataset-id CSE-CIC-IDS2018 \
  --model-id extra_trees \
  --seed 0 \
  --seconds 60 \
  --warmup-iterations 3 \
  --start-delay-s 5 \
  --power-rail VDD_IN \
  --power-sample-interval-s 0.1 \
  --power-log-out manifests/hardware/jetson_power_ina3221_vdd_in.csv \
  --out manifests/hardware/jetson_inference_energy_window.json \
  2>&1 | tee logs/jetson_inference_energy_window.log
```

Then read `onboard_power.average_power_w`, `measurement_duration_s`, and `measured_flows` from `manifests/hardware/jetson_inference_energy_window.json`, and fill:

```yaml
measurement_mode: measured_onboard_sensor
energy_source: jetson_ina3221_vdd_in
measurement_window: "inference-only predict loop with INA3221 VDD_IN sampling"
average_power_w: <JSON_ONBOARD_POWER_AVERAGE_POWER_W>
measurement_duration_s: <JSON_MEASUREMENT_DURATION_S>
measured_flows: <JSON_MEASURED_FLOWS>
```

Do not describe this as external-meter energy. In the paper, use wording such as "Jetson INA3221 VDD_IN module-power estimate" and keep `manifests/hardware/jetson_power_ina3221_vdd_in.csv` as raw evidence. Do not use `tegrastats` temperature/frequency logs alone as energy evidence. Use them alongside the sensor log to show thermal and clock stability.

For a conservative paper run, repeat the calibration for the slowest or most power-intensive configured model/dataset pair you intend to report, or run three calibration windows and record whether the hardware YAML uses the mean, median, or maximum observed energy per flow. If you use a single scalar in the current harness, prefer a conservative maximum and archive all calibration JSON and raw power CSV files.

Minimum run note for energy evidence:

```text
device: Jetson Orin NX 16GB
carrier/power path: <carrier board and module-power sensor path>
power mode: <exact nvpmodel label>
clock state: jetson_clocks enabled/disabled
measurement tool: Jetson INA3221 VDD_IN sysfs
measurement window: inference-only / component / full-run explanation
average_power_w or energy_per_flow_j: <value>
measured_flows: <count if using average_power_w>
calculation: energy_per_flow_j = average_power_w * measurement_duration_s / measured_flows
ambient/cooling note: <fan/heatsink condition>
```

## 9. Create A Jetson Hardware Config

Copy the template:

```bash
cp configs/hardware/jetson_orin_nx_super_template.yaml \
   configs/hardware/jetson_orin_nx_super.yaml
```

Edit `configs/hardware/jetson_orin_nx_super.yaml`:

```yaml
hardware_id: jetson_orin_nx16_super_maxn_ina3221
device_class: physical_edge
runtime: python_sklearn_cpu
jetson_linux_release: "R36.4.7"
jetpack_release: "6.2.1+b38"
l4t_core_package: "36.4.7-20250918154033"
cuda_compiler_release: "12.6.68"
device_tree_model: "NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super"
carrier_board: "generic"  # From the observed nv_tegra BOARD field; replace if a more specific carrier is known.
thread_count: 8
batch_size: 1
power_mode: "MAXN_SUPER"
measurement_mode: measured_onboard_sensor
energy_source: jetson_ina3221_vdd_in
measurement_window: "inference-only predict loop with INA3221 VDD_IN sampling"
average_power_w: <JSON_ONBOARD_POWER_AVERAGE_POWER_W>
measurement_duration_s: <JSON_MEASUREMENT_DURATION_S>
measured_flows: <JSON_MEASURED_INFERENCE_FLOWS>
```

Do not leave `hardware_id` as `jetson_orin_nx_super_unmeasured`. Do not leave `power_mode`, `measurement_window`, `jetson_linux_release`, `jetpack_release`, or `device_tree_model` as `replace_with_*` placeholders. Do not leave `measurement_mode` as `unmeasured_template`, `proxy`, `not_measured_*`, or any label containing `guess`, `dummy`, or `template`. Do not use vague energy sources such as `external`; use the concrete software source `jetson_ina3221_vdd_in`. Do not use nonzero energy values unless they come from the inference-only software telemetry procedure recorded in the run note.

Validate the filled config before any smoke or full benchmark run:

```bash
python scripts/validate_hardware_config.py \
  --config configs/hardware/jetson_orin_nx_super.yaml
```

This command must pass. If it fails, fix the hardware YAML first; the Tier-E orchestrator runs the same validation before starting the heavy Core4 grid.

The validator rejects a config even when the energy numbers are positive if the power mode, inference-only measurement window, measurement mode, or energy source still contains untrusted wording. This is intentional: otherwise a reviewer cannot tell which operating condition produced the latency and energy values.

## 10. Smoke Run On Jetson

Before running the full Core4 grid, run a synthetic smoke path with the Jetson hardware config:

```bash
mkdir -p /tmp/raise_ict_jetson_smoke_raw /tmp/raise_ict_jetson_smoke_tables /tmp/raise_ict_jetson_smoke_figures

python scripts/run_benchmark.py \
  --config configs/experiments/tier_s.yaml \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --out-dir /tmp/raise_ict_jetson_smoke_raw \
  --split-manifest /tmp/raise_ict_jetson_smoke_splits.csv \
  --feature-schema /tmp/raise_ict_jetson_smoke_schema.json \
  --profile-manifest /tmp/raise_ict_jetson_smoke_profile.json

python scripts/aggregate_results.py \
  --results /tmp/raise_ict_jetson_smoke_raw \
  --out /tmp/raise_ict_jetson_smoke_tables \
  --figures /tmp/raise_ict_jetson_smoke_figures
```

Inspect the smoke row:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

paths = sorted(Path('/tmp/raise_ict_jetson_smoke_raw').glob('*.csv'))
assert len(paths) == 1, paths
row = pd.read_csv(paths[0]).iloc[0]
print(row[['hardware_id', 'measurement_mode', 'energy_source', 'energy_per_flow_j', 'p95_latency_ms']])
assert row['hardware_id'] != 'jetson_orin_nx_super_unmeasured'
assert row['measurement_mode'] != 'unmeasured_template'
assert float(row['energy_per_flow_j']) > 0.0
PY
```

If this smoke assertion fails, do not start the full Core4 run. Fix `configs/hardware/jetson_orin_nx_super.yaml` and repeat the smoke path.

## 11. Dry Run The Tier-E Command Graph

Use dry run first. It prints the exact commands without executing the heavy grid.

```bash
mkdir -p logs
python scripts/run_tier_e_core4.py \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --dry-run | tee logs/tier_e_core4_dry_run.json
```

Confirm the printed graph contains:

- `scripts/validate_hardware_config.py`
- `scripts/audit_hardware.py`
- Three `scripts/run_benchmark.py` calls
- Split and feature-schema merges
- `scripts/aggregate_results.py`
- `scripts/run_tier_e_core4.py --write-profile-manifest-only`
- `scripts/check_completion.py --require-tier-e --strict` with anonymous manuscript and bibliography paths

## 12. Full Tier-E Core4 Run

Start telemetry logging in one shell:

```bash
mkdir -p logs manifests/hardware
sudo tegrastats --interval 1000 --logfile logs/tegrastats_tier_e_core4.log &
echo $! > /tmp/raise_ict_tegrastats.pid
```

Before the full run, complete the inference-only INA3221 calibration in Section 8 and copy its `average_power_w`, `measurement_duration_s`, and `measured_flows` into `configs/hardware/jetson_orin_nx_super.yaml`. Archive `manifests/hardware/jetson_inference_energy_window.json` and `manifests/hardware/jetson_power_ina3221_vdd_in.csv` before starting the Core4 grid. Do not convert the whole training-plus-evaluation duration into `energy_per_flow_j`; the denominator must come from the prediction-only calibration window.

Run the benchmark in `tmux`:

```bash
tmux new -s raise_ict_tiere
cd /path/to/ml-benchmark-2
conda activate raise-ict
mkdir -p logs

python scripts/run_tier_e_core4.py \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --skip-completion-audit \
  2>&1 | tee "logs/tier_e_core4_$(date -u +%Y%m%dT%H%M%SZ).log"
```

Stop telemetry logging after the command exits:

```bash
kill "$(cat /tmp/raise_ict_tegrastats.pid)"
```

Confirm that the INA3221 calibration JSON and raw CSV remain archived under `manifests/hardware/`.

The orchestrator performs these actions:

1. Validates `configs/hardware/jetson_orin_nx_super.yaml`.
2. Writes `manifests/hardware/tier_e_hardware_audit.json`.
3. Runs `tier_p_expanded.yaml`, `tier_p_cicids2017.yaml`, and `tier_p_cse_cic_ids2018.yaml` with the Jetson hardware config.
4. Writes raw edge rows under `results/raw/tier_e_*`.
5. Merges edge split and feature-schema manifests.
6. Aggregates `results/tables/tier_e_core4`.
7. Writes `manifests/hardware/tier_e_profile_manifest.json`.
8. Runs the strict Tier-E completion audit.

Generate the Tier-E analysis bundle after the strict audit:

```bash
python scripts/analyze_results.py \
  --raw results/tables/tier_e_core4/table_raw_results.csv \
  --summary results/tables/tier_e_core4/table_main_results.csv \
  --out results/analysis/tier_e_core4 \
  --attack-threat a1_constrained_score_search \
  --label 'Tier-E Core4 Jetson Orin NX' \
  --scope-note 'Jetson Orin NX physical-edge run with INA3221 software-observed energy metadata' \
  --split-manifest manifests/splits/tier_e_core4_split_manifest.csv \
  --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json
```

## 13. Manual Component Commands

Use this section only if the orchestrator fails and you need to rerun one component.

```bash
python scripts/validate_hardware_config.py \
  --config configs/hardware/jetson_orin_nx_super.yaml

python scripts/audit_hardware.py \
  --out manifests/hardware/tier_e_hardware_audit.json

python scripts/run_benchmark.py \
  --config configs/experiments/tier_p_expanded.yaml \
  --out-dir results/raw/tier_e_expanded \
  --split-manifest manifests/splits/tier_e_expanded_split_manifest.csv \
  --feature-schema manifests/feature_schemas/tier_e_expanded_feature_schema.json \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --profile-manifest manifests/hardware/tier_e_expanded_profile_manifest.json

python scripts/run_benchmark.py \
  --config configs/experiments/tier_p_cicids2017.yaml \
  --out-dir results/raw/tier_e_cicids2017 \
  --split-manifest manifests/splits/tier_e_cicids2017_split_manifest.csv \
  --feature-schema manifests/feature_schemas/tier_e_cicids2017_feature_schema.json \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --profile-manifest manifests/hardware/tier_e_cicids2017_profile_manifest.json

python scripts/run_benchmark.py \
  --config configs/experiments/tier_p_cse_cic_ids2018.yaml \
  --out-dir results/raw/tier_e_cse_cic_ids2018 \
  --split-manifest manifests/splits/tier_e_cse_cic_ids2018_split_manifest.csv \
  --feature-schema manifests/feature_schemas/tier_e_cse_cic_ids2018_feature_schema.json \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --profile-manifest manifests/hardware/tier_e_cse_cic_ids2018_profile_manifest.json

python scripts/merge_artifacts.py \
  manifests/splits/tier_e_expanded_split_manifest.csv \
  manifests/splits/tier_e_cicids2017_split_manifest.csv \
  manifests/splits/tier_e_cse_cic_ids2018_split_manifest.csv \
  --out manifests/splits/tier_e_core4_split_manifest.csv

python scripts/merge_artifacts.py \
  manifests/feature_schemas/tier_e_expanded_feature_schema.json \
  manifests/feature_schemas/tier_e_cicids2017_feature_schema.json \
  manifests/feature_schemas/tier_e_cse_cic_ids2018_feature_schema.json \
  --out manifests/feature_schemas/tier_e_core4_feature_schema.json

python scripts/aggregate_results.py \
  --results results/raw/tier_e_expanded results/raw/tier_e_cicids2017 results/raw/tier_e_cse_cic_ids2018 \
  --out results/tables/tier_e_core4 \
  --figures results/figures/tier_e_core4

python scripts/run_tier_e_core4.py \
  --hardware-config configs/hardware/jetson_orin_nx_super.yaml \
  --write-profile-manifest-only

python scripts/check_completion.py \
  --require-tier-e \
  --raw-results results/tables/tier_e_core4/table_raw_results.csv \
  --summary-results results/tables/tier_e_core4/table_main_results.csv \
  --split-manifest manifests/splits/tier_e_core4_split_manifest.csv \
  --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json \
  --feature-schema manifests/feature_schemas/tier_e_core4_feature_schema.json \
  --profile-manifest manifests/hardware/tier_e_profile_manifest.json \
  --manuscript anonymous_manuscript.tex \
  --bibliography anonymous_references.bib \
  --out manifests/completion/benchmark_completion_audit_strict_tier_e.json \
  --strict

python scripts/analyze_results.py \
  --raw results/tables/tier_e_core4/table_raw_results.csv \
  --summary results/tables/tier_e_core4/table_main_results.csv \
  --out results/analysis/tier_e_core4 \
  --attack-threat a1_constrained_score_search \
  --label 'Tier-E Core4 Jetson Orin NX' \
  --scope-note 'Jetson Orin NX physical-edge run with INA3221 software-observed energy metadata' \
  --split-manifest manifests/splits/tier_e_core4_split_manifest.csv \
  --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json
```

## 14. Expected Outputs

If the full run succeeds:

- `results/raw/tier_e_expanded/*.csv`: 120 raw rows.
- `results/raw/tier_e_cicids2017/*.csv`: 60 raw rows.
- `results/raw/tier_e_cse_cic_ids2018/*.csv`: 60 raw rows.
- `results/tables/tier_e_core4/table_raw_results.csv`: 240 raw rows.
- `results/tables/tier_e_core4/table_main_results.csv`: 48 summary rows.
- `results/figures/tier_e_core4/figure_pipeline.pdf`.
- `results/figures/tier_e_core4/figure_pareto.pdf`.
- `results/analysis/tier_e_core4/analysis-report.md`.
- `results/analysis/tier_e_core4/stats-appendix.md`.
- `manifests/hardware/tier_e_hardware_audit.json`.
- `manifests/hardware/tier_e_profile_manifest.json`.
- `manifests/completion/benchmark_completion_audit_strict_tier_e.json`.

Run shape QA:

```bash
python - <<'PY'
import json
import pandas as pd

raw = pd.read_csv('results/tables/tier_e_core4/table_raw_results.csv')
summary = pd.read_csv('results/tables/tier_e_core4/table_main_results.csv')
audit = json.load(open('manifests/completion/benchmark_completion_audit_strict_tier_e.json'))

print('raw_rows', len(raw))
print('summary_rows', len(summary))
print('hardware_ids', sorted(raw['hardware_id'].unique()))
print('min_energy_per_flow_j', float(raw['energy_per_flow_j'].min()))
print('min_validity_rate', float(raw['validity_rate'].min()))
print('strict_complete', audit['complete'])
print('strict_summary', audit['summary'])
PY
```

Acceptance criteria for Tier-E claims:

- `raw_rows` is `240`.
- `summary_rows` is `48`.
- `hardware_ids` contains only the declared Jetson hardware ID, not `cpu_proxy`.
- `min_energy_per_flow_j` is greater than `0.0`.
- `min_validity_rate` is at least `0.95`.
- `strict_complete` is `True`.

## 15. Troubleshooting

### Super Mode Does Not Appear

Do not run Tier-E claims. Recheck the flashing configuration. NVIDIA states the new Super modes require the new flashing configuration.

### Python Version Is Too Old

If `python --version` is below 3.10, use the Miniforge environment above or another Python 3.10+ environment. The repo declares `requires-python = ">=3.10"`.

### CSE-CIC-IDS2018 Runs Out Of Memory

Use the existing bounded config first. On 8GB modules:

- Enable swap.
- Close desktop applications.
- Run from a text console or SSH session.
- Avoid parallel benchmark jobs.
- Keep `tier_p_cse_cic_ids2018.yaml` bounded unless the manuscript explicitly changes the evidence claim.

### Strict Audit Fails On Energy

Check:

```bash
python -m json.tool manifests/hardware/tier_e_profile_manifest.json | head -120
python - <<'PY'
import pandas as pd
raw = pd.read_csv('results/tables/tier_e_core4/table_raw_results.csv')
print(raw[['hardware_id', 'measurement_mode', 'energy_source', 'energy_per_flow_j']].drop_duplicates())
PY
```

The strict audit expects measured or software-observed mode/source metadata and positive energy values. The pre-run validator also expects a non-placeholder power mode and a non-placeholder inference-only measurement window.

Common causes:

- `configs/hardware/jetson_orin_nx_super.yaml` still has `measurement_mode: unmeasured_template`.
- `power_mode` or `measurement_window` still contains a `replace_with_*` placeholder.
- `energy_source` is `none`, `proxy`, or empty.
- `measurement_mode` or `energy_source` contains `not_measured`, `guess`, `dummy`, `template`, or another non-measured marker.
- `energy_source` is vague, for example `external`, instead of naming the concrete software source `jetson_ina3221_vdd_in`.
- `average_power_w` or `measurement_duration_s` is still `0.0`.
- `measured_flows` is `0` or does not describe the measured inference window.
- `manifests/hardware/tier_e_profile_manifest.json` was not regenerated after editing the hardware YAML.

### Hardware Audit Does Not Detect Jetson

Check:

```bash
(tr -d '\000' </proc/device-tree/model; echo)
python -m json.tool manifests/hardware/tier_e_hardware_audit.json | head -120
```

If the device-tree model does not contain Jetson/Orin markers because of a custom carrier/BSP, record the raw hardware evidence and update the audit logic only after confirming the board identity.

## 16. Archive The Evidence

After a successful run, preserve these files with the manuscript submission or experiment log:

```text
configs/hardware/jetson_orin_nx_super.yaml
manifests/hardware/jetson_nv_tegra_release.txt
manifests/hardware/jetson_l4t_jetpack_packages.txt
manifests/hardware/jetson_nvcc_version.txt
manifests/hardware/jetson_device_tree_model.txt
manifests/hardware/jetson_nproc.txt
manifests/hardware/jetson_df_h.txt
manifests/hardware/jetson_free_h.txt
manifests/hardware/jetson_nvpmodel_selected.txt
manifests/hardware/jetson_clocks_selected.txt
manifests/hardware/jetson_power_rails.json
manifests/hardware/jetson_power_ina3221_vdd_in.csv
manifests/hardware/jetson_inference_energy_window.json
manifests/hardware/tier_e_hardware_audit.json
manifests/hardware/tier_e_profile_manifest.json
manifests/dataset_hashes/tier_p_core4_download_manifest.json
manifests/splits/tier_e_core4_split_manifest.csv
manifests/feature_schemas/tier_e_core4_feature_schema.json
results/tables/tier_e_core4/table_raw_results.csv
results/tables/tier_e_core4/table_main_results.csv
results/analysis/tier_e_core4/   # if generated separately
logs/tier_e_core4_*.log
logs/jetson_inference_energy_window.log
logs/tegrastats_tier_e_core4.log
```

Only after these artifacts pass strict audit should the manuscript describe Tier-E Jetson Orin NX Super latency or energy results.
