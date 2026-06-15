# Timed Tier-E Core4 HGB+MLP Acceptance Evidence

## Scope

- `results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv`: 800 rows; expected 800.
- `results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv`: 80 rows; expected 80.
- `results/tables/tier_e_random_control_hgb_mlp_timed/table_raw_results.csv`: 100 rows; expected 100.
- `results/tables/tier_e_random_control_hgb_mlp_timed/table_main_results.csv`: 10 rows; expected 10.
- Models: extra_trees, hist_gradient_boosting, logistic_regression, mlp_sklearn, random_forest.
- Energy remains shared INA3221 `VDD_IN` module-power context, not model-isolated energy.
- Timing sidecars report wall-clock execution burden and are not energy measurements.

## Timing Summary

- acceptance_report: events=1, total_s=0.027, max_s=0.027.
- admissibility_rejection: events=1, total_s=0.367, max_s=0.367.
- aggregate_results: events=2, total_s=7.706, max_s=5.399.
- analysis: events=1, total_s=3.630, max_s=3.630.
- artifact_merge: events=4, total_s=2.921, max_s=0.746.
- clean_profile: events=300, total_s=116.445, max_s=2.471.
- dataset_load: events=60, total_s=1015.798, max_s=44.427.
- hardware_audit: events=1, total_s=0.110, max_s=0.110.
- hardware_validation: events=1, total_s=0.108, max_s=0.108.
- model_training: events=300, total_s=2837.614, max_s=24.666.
- preprocess_manifest: events=60, total_s=191.879, max_s=5.266.
- preprocessing: events=300, total_s=1028.696, max_s=6.115.
- profile_manifest: events=1, total_s=0.004, max_s=0.004.
- result_writing: events=900, total_s=3.540, max_s=0.040.
- run_benchmark: events=5, total_s=5616.511, max_s=2114.609.
- split_construction: events=300, total_s=50.382, max_s=0.367.
- strict_audit: events=1, total_s=0.772, max_s=0.772.
- threat_evaluation: events=900, total_s=358.524, max_s=4.426.
- timed_strict_audit: events=1, total_s=0.810, max_s=0.810.

## Manuscript Boundary

- Allowed: constructed admissibility negative controls were rejected by the checker.
- Forbidden: do not describe the rejection suite as a test on independently submitted manuscripts.
- Forbidden: do not use timing evidence as calibrated energy or model-isolated power evidence.
