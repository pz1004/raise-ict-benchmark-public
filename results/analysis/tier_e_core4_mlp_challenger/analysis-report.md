# RAISE-ICT Tier-E Core4 MLP Challenger Analysis Report

## Scope

- Raw rows: 320.
- Datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, UNSW-NB15.
- Models: extra_trees, logistic_regression, mlp_sklearn, random_forest.
- Threat rows: a0_clean, a1_constrained_feature, a1_constrained_score_search, a4_split_shift.
- Primary constrained threat for paired analysis: a1_constrained_score_search.
- Seeds per dataset/model/threat: 5 to 5.

## Main Findings

- CICIDS2017: mlp_sklearn has the highest constrained robust utility (0.647 mean over 5 seeds).
- CSE-CIC-IDS2018: mlp_sklearn has the highest constrained robust utility (0.747 mean over 5 seeds).
- TON_IoT: mlp_sklearn has the highest constrained robust utility (0.898 mean over 5 seeds).
- UNSW-NB15: random_forest has the highest constrained robust utility (0.862 mean over 5 seeds).

Constrained perturbations produce measurable robust-utility changes in this run. These findings are from physical Jetson Core4 run with three classical baselines and one sklearn MLP challenger, not final leaderboard claims.

## Claim Candidates

- Claim: The harness executes the Tier-E Core4 MLP Challenger experiment grid on public intrusion-detection datasets.
  - Source evidence: `results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv`, `manifests/dataset_hashes/tier_p_core4_download_manifest.json`, and `manifests/splits/tier_e_core4_mlp_challenger_split_manifest.csv`.
  - Allowed wording: "The Tier-E Core4 MLP Challenger run validates the RAISE-ICT execution path for physical Jetson Core4 run with three classical baselines and one sklearn MLP challenger."
  - Forbidden stronger wording: "RAISE-ICT establishes a final leaderboard" or "model X is generally best."

- Claim: The constrained-attack rows in the Tier-E Core4 MLP Challenger run report explicit validity counts and pass rates for the implemented budget, bounds, immutable-field, and relation checks.
  - Source evidence: `valid_count`, `invalid_count`, `validity_rate`, and component pass-rate columns.
  - Allowed wording: "Generated constrained-feature examples are filtered by the implemented validity checks."
  - Forbidden stronger wording: "The attacks are packet-realizable."

## Caveats

- The run is an experiment-scope validation path, not a final leaderboard claim.
- Feature-space constrained attacks are not packet replay or simulator validated.
- Latency and energy claims are limited to the declared hardware tier and measurement metadata.
- Dataset-specific split limits are documented in the split manifest and should be carried into manuscript claims.
