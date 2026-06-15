# Constructed Admissibility-Rejection Suite

- All constructed invalid bundles rejected: `true`.
- These are constructed negative controls, not tests on independently submitted manuscripts.

| Case | Rejected | Blocking checks |
|---|---:|---|
| missing_split_manifest | true | artifacts.split_manifest_present |
| wrong_row_count | true | core4.raw_rows |
| missing_model_rows | true | core4.raw_rows, core4.models |
| non_jetson_hardware | true | profiling.physical_edge_results_required, tier_e.profile_manifest_hardware_match |
| a1_validity_below_threshold | true | attacks.validity_threshold |
| missing_runtime_profile_metadata | true | profiling.profile_metadata_present |
