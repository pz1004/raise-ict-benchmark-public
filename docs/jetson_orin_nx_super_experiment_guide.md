# Jetson Orin NX Super Timed Core4 Reproduction Guide

This guide describes the public timed Tier-E Core4 package. It is intended for rerunning or auditing the evidence bundle on a Jetson Orin NX Super-class host. The included archive contains result tables, timing sidecars, and manifests, not raw upstream IDS datasets.

## Scope

- Core4 datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15.
- Models: `extra_trees`, `hist_gradient_boosting`, `logistic_regression`, `mlp_sklearn`, and `random_forest`.
- Timed Tier-E Core4 evidence: 800 raw rows and 80 summary rows.
- CIC-style random controls: 100 raw rows and 10 summary rows.
- Seeds: `0..9`.
- Hardware evidence: Jetson Orin NX Super with shared INA3221 `VDD_IN` module-power provenance.
- Timing evidence: command-level and stage-level wall-clock sidecars.
- Energy boundary: shared module-power context from an inference window, not model-isolated energy.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Raw datasets are not redistributed. Use the official source notes in `DATASET_USAGE.md` and opt in explicitly before using the mirror helper:

```bash
python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets CICIDS2017 CSE-CIC-IDS2018 TON_IoT UNSW-NB15
```

## Verify Included Evidence

From the bundle root, rerun the strict audit against the manuscript and bibliography that contain the submitted claims:

```bash
python scripts/check_completion.py \
  --require-tier-e \
  --require-timing \
  --expected-raw-rows 800 \
  --expected-summary-rows 80 \
  --expected-split-rows 40 \
  --expected-feature-schema-records 40 \
  --expected-seeds 0 1 2 3 4 5 6 7 8 9 \
  --expected-models extra_trees hist_gradient_boosting logistic_regression mlp_sklearn random_forest \
  --raw-results results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv \
  --summary-results results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv \
  --split-manifest manifests/splits/tier_e_core4_hgb_mlp_timed_split_manifest.csv \
  --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json \
  --feature-schema manifests/feature_schemas/tier_e_core4_hgb_mlp_timed_feature_schema.json \
  --hardware-audit manifests/hardware/tier_e_core4_hgb_mlp_timed_hardware_audit.json \
  --profile-manifest manifests/hardware/tier_e_core4_hgb_mlp_timed_profile_manifest.json \
  --timing-events results/timing/tier_e_core4_hgb_mlp_timed/timing_events.csv \
  --timing-summary results/timing/tier_e_core4_hgb_mlp_timed/timing_summary.csv \
  --command-timeline results/timing/tier_e_core4_hgb_mlp_timed/command_timeline.json \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib \
  --out manifests/completion/public_recheck.json \
  --strict
```

The archived public audit is:

```text
manifests/completion/benchmark_completion_audit_strict_tier_e_core4_hgb_mlp_timed.json
```

## Dry Run The Single-Command Graph

Inspect the command graph before running the full benchmark:

```bash
python scripts/run_tier_e_core4_hgb_mlp_timed.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib \
  --dry-run
```

The graph should contain hardware validation, hardware audit, five-model ten-seed Core4 runs, CIC-style random-control rows, aggregation, analysis, profile-manifest generation, timing analysis, constructed admissibility rejection, and strict timed audit.

## Full Rerun

Run the full command only on the Jetson-class host after reconstructing the local datasets:

```bash
./scripts/run_tier_e_core4_hgb_mlp_timed.sh \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib
```

The command refuses Tier-E execution unless the hardware audit classifies the host as edge eligible. The wrapper uses GNU `/usr/bin/time` when available and falls back to shell wall-clock timing when it is not available.

## Expected Outputs

```text
results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv
results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv
results/tables/tier_e_random_control_hgb_mlp_timed/table_raw_results.csv
results/tables/tier_e_random_control_hgb_mlp_timed/table_main_results.csv
results/timing/tier_e_core4_hgb_mlp_timed/
results/analysis/tier_e_core4_hgb_mlp_timed/
results/figures/tier_e_core4_hgb_mlp_timed/
manifests/splits/tier_e_core4_hgb_mlp_timed_split_manifest.csv
manifests/feature_schemas/tier_e_core4_hgb_mlp_timed_feature_schema.json
manifests/hardware/tier_e_core4_hgb_mlp_timed_hardware_audit.json
manifests/hardware/tier_e_core4_hgb_mlp_timed_profile_manifest.json
manifests/completion/benchmark_completion_audit_strict_tier_e_core4_hgb_mlp_timed.json
```

Quick shape check:

```bash
python - <<'PY'
import json
import pandas as pd

raw = pd.read_csv('results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv')
summary = pd.read_csv('results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv')
random_raw = pd.read_csv('results/tables/tier_e_random_control_hgb_mlp_timed/table_raw_results.csv')
random_summary = pd.read_csv('results/tables/tier_e_random_control_hgb_mlp_timed/table_main_results.csv')
audit = json.load(open('manifests/completion/benchmark_completion_audit_strict_tier_e_core4_hgb_mlp_timed.json'))

print('core4_raw_rows', len(raw))
print('core4_summary_rows', len(summary))
print('random_raw_rows', len(random_raw))
print('random_summary_rows', len(random_summary))
print('models', sorted(raw['model_id'].unique()))
print('seeds', sorted(raw['seed'].unique()))
print('hardware_ids', sorted(raw['hardware_id'].unique()))
print('strict_complete', audit['complete'])
print('strict_summary', audit['summary'])
PY
```

Expected values:

- `core4_raw_rows`: 800.
- `core4_summary_rows`: 80.
- `random_raw_rows`: 100.
- `random_summary_rows`: 10.
- `models`: `extra_trees`, `hist_gradient_boosting`, `logistic_regression`, `mlp_sklearn`, `random_forest`.
- `seeds`: `0` through `9`.
- `hardware_ids`: `jetson_orin_nx_super_ina3221_20260608t140153z`.
- `strict_complete`: `True`.

## Archive Boundary

For Zenodo, upload a clean archive of the bundle contents after the timed artifacts are synchronized. Do not include local `.git/`, raw datasets, raw per-run outputs, Python caches, LaTeX build products, or private manuscript-review notes.
