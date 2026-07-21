# RAISE-ICT Benchmark Public Bundle

RAISE-ICT is a Python benchmark harness for intrusion-detection system (IDS) evaluation. An IDS is a model or rule system that flags malicious network activity. The main RAISE-ICT output is an auditable result-row contract: each reported score is tied to a CSV row plus machine-readable evidence for dataset source, split provenance, preprocessing state, model identity, threat setting, hardware profile, timing sidecars, and audit status. The bundle also includes a claim-conditioned pairwise checker that determines whether two result rows support an ordering under a declared comparison context.

## What this bundle provides

A row-level result record is one CSV row for a specific dataset, split, seed, model, threat setting, and hardware profile. A summary row aggregates the ten seed-level rows for one dataset-model-threat condition. A manifest is a machine-readable evidence file that records provenance, such as dataset hashes, split identifiers, feature schemas, or hardware profile metadata. An artifact is any generated evidence file in this bundle, such as a CSV table, JSON manifest, PDF figure, or Markdown analysis note.

Core4 means the four-dataset evidence set included here: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15. The Jetson Orin NX Super is the edge computer used for the included hardware profile. Tier-E means the edge-device evaluation tier, where runtime evidence is recorded on that Jetson profile. The timed Core4 Tier-E artifacts describe this four-dataset edge-device evaluation with three classical baselines, one HistGradientBoosting challenger, and one CPU-compatible MLP challenger.

The included timed Tier-E evidence contains:

- 800 row-level result records in `results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv`.
- 80 aggregated summary rows in `results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv`.
- 4 datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, and UNSW-NB15.
- 5 fixed-configuration models: Extra Trees, HistGradientBoosting, logistic regression, random forest, and `mlp_sklearn`.
- `mlp_sklearn` is a CPU-compatible scikit-learn MLP challenger, not a neural IDS survey or architecture search.
- `hist_gradient_boosting` is a fixed scikit-learn tabular boosting challenger, not a boosting architecture search.
- 4 threat settings: clean evaluation, constrained feature attack, constrained score-search attack, and split-shift evaluation.
- 10 random seeds for each dataset-model-threat combination.
- Timing sidecars in `results/timing/tier_e_core4_hgb_mlp_timed/`.
- Constructed admissibility-rejection evidence in `results/analysis/tier_e_core4_hgb_mlp_timed/admissibility_rejection_report.md`.

A positive-control random split is a sanity-check experiment where the split is intentionally easier than the held-out Core4 split. It helps confirm that the pipeline can recover high scores when train/test separation is relaxed; it should not be read as deployment evidence. The timed positive-control tables contain 100 row-level records and 10 summary rows in `results/tables/tier_e_random_control_hgb_mlp_timed/`.

## How to read the evidence

The raw table includes metrics such as clean macro-F1, robust utility, attack success rate, validity rate, latency, throughput, memory, energy context, and the preprocessing-state hash.

The bundle includes these manifests:

- Dataset download manifests in `manifests/dataset_hashes/`.
- Split manifests in `manifests/splits/`.
- Feature-schema manifests in `manifests/feature_schemas/`.
- Hardware and runtime-profile manifests in `manifests/hardware/`.
- Completion-audit manifests in `manifests/completion/`.
- Timing sidecars in `results/timing/`.

The strict completion audit is the checker that verifies required files, expected row counts, expected model IDs, evidence paths, timing sidecars, and manuscript-linked claim checks. The included strict timed Tier-E audit record is `manifests/completion/benchmark_completion_audit_strict_tier_e_core4_hgb_mlp_timed.json`; it reports `complete=true`, 39 passed required checks, 1 not-required check, and 0 incomplete checks for the 800-row timed evidence path.

## Verify the pairwise admission rule

The pairwise checker treats a proposed comparison, rather than a paper or isolated score, as the unit of admission. A declared context fixes the permitted metric, focal comparison field, non-focal invariants, and required evidence fields before the candidate outcomes are evaluated. The checker returns `defined`, `context_mismatch`, or `insufficient_evidence`; an explicitly authorized `not_applicable` state is recorded separately from unresolved evidence.

The constructed test suite covers all three decisions, authorized `not_applicable`, unsupported metrics, row-to-request metric mismatches, and rejection of the withdrawn single-row claims interface:

```bash
python -m pytest -q tests/test_pairwise_admission.py
```

These tests establish the behavior of the software branches; they are not an external or independent validation of the comparison construct. To evaluate user-supplied rows, provide a context file, row file, and pair-request file:

```bash
python scripts/check_pairwise_admission.py \
  --contexts /path/to/contexts.yaml \
  --rows /path/to/rows.yaml \
  --pairs /path/to/pairs.yaml \
  --out-dir /path/to/pairwise-output
```

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

Raw IDS datasets are not redistributed in this bundle. Use `DATASET_USAGE.md` for official dataset sources, citation and redistribution notes, and the optional mirror-download policy. The download script is official-source-first: mirror downloads require explicit opt-in after accepting the upstream dataset terms.

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

## Re-run the timed Tier-E path

The Jetson Orin NX Super is the edge device used for the included Tier-E hardware profile. Before rerunning the benchmark, inspect the command graph:

```bash
python scripts/run_tier_e_core4_hgb_mlp_timed.py \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --dry-run
```

After reconstructing the local datasets from `data_registry.yaml`, run:

```bash
./scripts/run_tier_e_core4_hgb_mlp_timed.sh \
  --hardware-config configs/hardware/jetson_orin_nx_super_measured_20260608T140153Z.yaml \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Run the strict audit separately with the anonymous manuscript and bibliography paths after benchmark regeneration.

The latency values are vectorized `model.predict(features)` timings normalized per flow with Python timing instrumentation. The timing sidecars report command-level and stage-level wall-clock burden. INA3221 `VDD_IN` is the Jetson onboard module-power sensor and rail used for this run; the energy values are shared module-power measurements from a repeated inference window, not model-isolated energy measurements.

## Repository layout

```text
configs/            Benchmark, hardware, attack, and model configuration files.
data_registry.yaml  Dataset registry used to reconstruct local data inputs.
docs/               Jetson execution guide.
manifests/          Dataset, split, feature-schema, hardware, and audit evidence.
results/            Compact tables, figures, timing sidecars, and analysis summaries.
scripts/            Reproduction, aggregation, profiling, timing, and audit scripts.
src/                RAISE-ICT Python package.
```
