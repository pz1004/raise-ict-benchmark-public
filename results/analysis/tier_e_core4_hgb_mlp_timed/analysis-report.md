# RAISE-ICT Tier-E Timed Core4 HGB MLP Analysis Report

## Scope

- Raw rows: 800.
- Datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, UNSW-NB15.
- Models: extra_trees, hist_gradient_boosting, logistic_regression, mlp_sklearn, random_forest.
- Threat rows: a0_clean, a1_constrained_feature, a1_constrained_score_search, a4_split_shift.
- Primary constrained threat for paired analysis: a1_constrained_score_search.
- Seeds per dataset/model/threat: 10 to 10.

## Main Findings

- CICIDS2017: mlp_sklearn has the highest constrained robust utility (0.648 mean over 10 seeds).
- CSE-CIC-IDS2018: mlp_sklearn has the highest constrained robust utility (0.760 mean over 10 seeds).
- TON_IoT: mlp_sklearn has the highest constrained robust utility (0.906 mean over 10 seeds).
- UNSW-NB15: hist_gradient_boosting has the highest constrained robust utility (0.904 mean over 10 seeds).

Constrained perturbations produce measurable robust-utility changes in this run. These findings are from physical Jetson run with three classical baselines, HGB, and sklearn MLP over ten seeds, not field-wide ranking claims.

## Claim Candidates

- Claim: The harness executes the Tier-E Timed Core4 HGB MLP experiment grid on public intrusion-detection datasets.
  - Source evidence: `results/tables/tier_e_core4_hgb_mlp_timed/table_raw_results.csv`, `manifests/dataset_hashes/tier_p_core4_download_manifest.json`, and `manifests/splits/tier_e_core4_hgb_mlp_timed_split_manifest.csv`.
  - Allowed wording: "The Tier-E Timed Core4 HGB MLP run validates the RAISE-ICT execution path for physical Jetson run with three classical baselines, HGB, and sklearn MLP over ten seeds."
  - Forbidden stronger wording: "RAISE-ICT establishes a final field-wide ranking" or "model X is generally best."

- Claim: The constrained-attack rows in the Tier-E Timed Core4 HGB MLP run report explicit validity counts and pass rates for the implemented budget, bounds, immutable-field, and relation checks.
  - Source evidence: `valid_count`, `invalid_count`, `validity_rate`, and component pass-rate columns.
  - Allowed wording: "Generated constrained-feature examples are filtered by the implemented validity checks."
  - Forbidden stronger wording: "The attacks are packet-realizable."

## Caveats

- The run is an experiment-scope validation path, not a final field-wide ranking claim.
- Feature-space constrained attacks are not packet replay or simulator validated.
- Latency and energy claims are limited to the declared hardware tier and measurement metadata.
- Dataset-specific split limits are documented in the split manifest and should be carried into manuscript claims.
