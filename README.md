# RAISE-ICT Benchmark Public Bundle

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20637748.svg)](https://doi.org/10.5281/zenodo.20637748)

The badge identifies the Zenodo concept record for the public evidence bundle. If Zenodo assigns a new version-specific record DOI for this upload, use that version-specific DOI in submission metadata.

RAISE-ICT is a Python benchmark harness for intrusion-detection system (IDS) evaluation. An IDS is a model or rule system that flags malicious network activity. The main RAISE-ICT output is an auditable result-row contract: each reported score is tied to a CSV row plus machine-readable evidence for dataset source, split provenance, preprocessing state, model identity, threat setting, hardware profile, and audit status.

## What this bundle provides

A row-level result record is one CSV row for a specific dataset, split, seed, model, threat setting, and hardware profile. A summary row aggregates the five seed-level rows for one dataset-model-threat condition. A manifest is a machine-readable evidence file that records provenance, such as dataset hashes, split identifiers, feature schemas, or hardware profile metadata. An artifact is any generated evidence file in this bundle, such as a CSV table, JSON manifest, PDF figure, or Markdown analysis note.

Core4 means the four-dataset evidence set included here: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15. The Jetson Orin NX Super is the edge computer used for the included hardware profile. Tier-E means the edge-device evaluation tier, where runtime evidence is recorded on that Jetson profile. Core4+MLP Tier-E artifacts are the generated evidence files that describe this four-dataset edge-device evaluation with three classical baselines and one CPU-compatible MLP challenger.

The included Core4+MLP Tier-E evidence contains:

- 320 row-level result records in `results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv`.
- 64 aggregated summary rows in `results/tables/tier_e_core4_mlp_challenger/table_main_results.csv`.
- 4 datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15.
- 4 fixed-configuration models: Extra Trees, logistic regression, random forest, and `mlp_sklearn`.
- `mlp_sklearn` is a CPU-compatible scikit-learn MLP challenger, not a neural IDS survey or architecture search.
- 4 threat settings: clean evaluation, constrained feature attack, constrained score-search attack, and split-shift evaluation.
- 5 random seeds for each dataset-model-threat combination.

A positive-control random split is a sanity-check experiment where the split is intentionally easier than the held-out Core4 split. It helps confirm that the pipeline can recover high scores when train/test separation is relaxed; it should not be read as deployment evidence. The Core4+MLP positive-control tables contain 40 row-level records and 8 summary rows in `results/tables/tier_e_random_control_mlp_challenger/`.

## How to read the evidence

The raw table includes metrics such as clean macro-F1, robust utility, attack success rate, validity rate, latency, throughput, memory, energy, and the preprocessing-state hash.

The bundle includes these manifests:

- Dataset download manifests in `manifests/dataset_hashes/`.
- Split manifests in `manifests/splits/`.
- Feature-schema manifests in `manifests/feature_schemas/`.
- Hardware and runtime-profile manifests in `manifests/hardware/`.
- Completion-audit manifests in `manifests/completion/`.

The strict completion audit is the checker that verifies required files, expected row counts, expected model IDs, evidence paths, and manuscript-linked claim checks. The included strict Tier-E challenger audit record is `manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json`; it reports `complete=true`, 34 passed checks, 1 not-required check, and 0 incomplete checks for the 320-row Core4+MLP evidence path.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`.

## License and anonymity

The code in this anonymous-review bundle is released under the MIT License. The license applies to this benchmark code and documentation, not to the upstream datasets. The license currently uses `Anonymous Authors`; replace that holder before a non-anonymous public release.

## Dataset access

Raw IDS datasets are not redistributed in this bundle. Use `DATASET_USAGE.md` for official dataset sources, citation and redistribution notes, and the optional mirror-download policy. The download script is official-source-first: third-party mirror downloads require explicit opt-in after accepting the upstream dataset terms.

```bash
python scripts/download_datasets.py \
  --allow-third-party-mirrors \
  --datasets UNSW-NB15 TON_IoT
```

## Verify the current evidence

The command below reruns the strict audit. The `--manuscript` and `--bibliography` paths should point to the manuscript source and bibliography that contain the publication claims being checked.

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

## Re-run the Core4+MLP Tier-E path

The Jetson Orin NX Super is the edge device used for the included Tier-E hardware profile. Before rerunning the benchmark, inspect the command graph:

```bash
python scripts/run_tier_e_core4_mlp_challenger.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --dry-run
```

After reconstructing the local datasets from `data_registry.yaml`, run:

```bash
python scripts/run_tier_e_core4_mlp_challenger.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml
```

Run the strict audit separately with the anonymous manuscript and bibliography paths after benchmark regeneration.

The latency values are vectorized `model.predict(features)` timings normalized per flow with Python timing instrumentation. INA3221 `VDD_IN` is the Jetson onboard module-power sensor and rail used for this run; the energy values are shared module-power measurements from a repeated inference window, not isolated per-model energy measurements.

## Repository layout

```text
configs/            Benchmark, hardware, attack, and model configuration files.
data_registry.yaml  Dataset registry used to reconstruct local data inputs.
docs/               Jetson execution guide.
manifests/          Dataset, split, feature-schema, hardware, and audit evidence.
results/            Compact tables, figures, and analysis summaries.
scripts/            Reproduction, aggregation, profiling, and audit scripts.
src/                RAISE-ICT Python package.
```
