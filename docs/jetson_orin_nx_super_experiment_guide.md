# Jetson Orin NX Super Core4+MLP Reproduction Guide

This guide describes the public Core4+MLP Tier-E package. It is intended for rerunning or auditing the evidence bundle on a Jetson Orin NX Super-class host. The included archive contains result tables and manifests, not raw upstream IDS datasets.

## Scope

- Core4 datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15.
- Models: `extra_trees`, `logistic_regression`, `mlp_sklearn`, and `random_forest`.
- Tier-E Core4+MLP evidence: 320 raw rows and 64 summary rows.
- CIC-style random controls: 40 raw rows and 8 summary rows.
- Hardware evidence: Jetson Orin NX Super with shared INA3221 `VDD_IN` module-power provenance.
- Energy boundary: shared module-power context from an inference window, not per-model isolated energy.

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
  --expected-raw-rows 320 \
  --expected-summary-rows 64 \
  --expected-models extra_trees logistic_regression mlp_sklearn random_forest \
  --raw-results results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv \
  --summary-results results/tables/tier_e_core4_mlp_challenger/table_main_results.csv \
  --split-manifest manifests/splits/tier_e_core4_mlp_challenger_split_manifest.csv \
  --dataset-manifest manifests/dataset_hashes/tier_p_core4_download_manifest.json \
  --feature-schema manifests/feature_schemas/tier_e_core4_mlp_challenger_feature_schema.json \
  --hardware-audit manifests/hardware/tier_e_mlp_challenger_hardware_audit.json \
  --profile-manifest manifests/hardware/tier_e_core4_mlp_challenger_profile_manifest.json \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib \
  --out manifests/completion/public_recheck.json \
  --strict
```

The archived public audit is:

```text
manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json
```

## Dry Run The Single-Command Graph

Inspect the command graph before running the full benchmark:

```bash
python scripts/run_tier_e_core4_mlp_challenger.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib \
  --dry-run
```

The graph should contain hardware validation, hardware audit, conditional classical Core4 regeneration, MLP Core4 rows, random-control rows, aggregation, analysis, profile-manifest generation, acceptance-evidence reporting, and strict audit.

## Full Rerun

Run the full command only on the Jetson-class host after reconstructing the local datasets:

```bash
python scripts/run_tier_e_core4_mlp_challenger.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --manuscript /path/to/anonymous_manuscript.tex \
  --bibliography /path/to/anonymous_references.bib
```

The command refuses Tier-E execution unless the hardware audit classifies the host as edge eligible. The classical Core4 path is reused only if its strict precheck passes; otherwise the orchestrator regenerates it before adding the MLP challenger rows.

## Expected Outputs

```text
results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv
results/tables/tier_e_core4_mlp_challenger/table_main_results.csv
results/tables/tier_e_random_control_mlp_challenger/table_raw_results.csv
results/tables/tier_e_random_control_mlp_challenger/table_main_results.csv
results/analysis/tier_e_core4_mlp_challenger/
results/figures/tier_e_core4_mlp_challenger/
manifests/splits/tier_e_core4_mlp_challenger_split_manifest.csv
manifests/feature_schemas/tier_e_core4_mlp_challenger_feature_schema.json
manifests/hardware/tier_e_mlp_challenger_hardware_audit.json
manifests/hardware/tier_e_core4_mlp_challenger_profile_manifest.json
manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json
```

Quick shape check:

```bash
python - <<'PY'
import json
import pandas as pd

raw = pd.read_csv('results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv')
summary = pd.read_csv('results/tables/tier_e_core4_mlp_challenger/table_main_results.csv')
random_raw = pd.read_csv('results/tables/tier_e_random_control_mlp_challenger/table_raw_results.csv')
random_summary = pd.read_csv('results/tables/tier_e_random_control_mlp_challenger/table_main_results.csv')
audit = json.load(open('manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json'))

print('core4_raw_rows', len(raw))
print('core4_summary_rows', len(summary))
print('random_raw_rows', len(random_raw))
print('random_summary_rows', len(random_summary))
print('models', sorted(raw['model_id'].unique()))
print('hardware_ids', sorted(raw['hardware_id'].unique()))
print('strict_complete', audit['complete'])
print('strict_summary', audit['summary'])
PY
```

Expected values:

- `core4_raw_rows`: 320.
- `core4_summary_rows`: 64.
- `random_raw_rows`: 40.
- `random_summary_rows`: 8.
- `models`: `extra_trees`, `logistic_regression`, `mlp_sklearn`, `random_forest`.
- `hardware_ids`: `jetson_orin_nx_super_ina3221_20260608t140153z`.
- `strict_complete`: `True`.

## Archive Boundary

For Zenodo, upload a clean archive of the bundle contents. Do not include local `.git/`, raw datasets, raw per-run outputs, Python caches, LaTeX build products, or private manuscript-review notes.
