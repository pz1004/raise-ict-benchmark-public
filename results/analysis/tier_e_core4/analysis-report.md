# RAISE-ICT Tier-E Core4 Jetson Orin NX Analysis Report

## Scope

- Raw rows: 240.
- Datasets: CICIDS2017, CSE-CIC-IDS2018, TON_IoT, UNSW-NB15.
- Models: extra_trees, logistic_regression, random_forest.
- Threat rows: a0_clean, a1_constrained_feature, a1_constrained_score_search, a4_split_shift.
- Primary constrained threat for paired analysis: a1_constrained_score_search.
- Seeds per dataset/model/threat: 5 to 5.

## Main Findings

- CICIDS2017: logistic_regression has the highest constrained robust utility (0.642 mean over 5 seeds).
- CSE-CIC-IDS2018: logistic_regression has the highest constrained robust utility (0.567 mean over 5 seeds).
- TON_IoT: logistic_regression has the highest constrained robust utility (0.869 mean over 5 seeds).
- UNSW-NB15: random_forest has the highest constrained robust utility (0.862 mean over 5 seeds).

Constrained perturbations produce measurable robust-utility changes in this run. These findings are from Jetson Orin NX physical-edge run with INA3221 software-observed energy metadata, not final leaderboard claims.

## Claim Candidates

- Claim: The harness executes the Tier-E Core4 Jetson Orin NX experiment grid on public intrusion-detection datasets.
  - Source evidence: `results/tables/tier_e_core4/table_raw_results.csv`, `manifests/dataset_hashes/tier_p_core4_download_manifest.json`, and `manifests/splits/tier_e_core4_split_manifest.csv`.
  - Allowed wording: "The Tier-E Core4 Jetson Orin NX run validates the RAISE-ICT execution path for Jetson Orin NX physical-edge run with INA3221 software-observed energy metadata."
  - Forbidden stronger wording: "RAISE-ICT establishes a final leaderboard" or "model X is generally best."

- Claim: The constrained-attack rows in the Tier-E Core4 Jetson Orin NX run report explicit validity counts and pass rates for the implemented budget, bounds, immutable-field, and relation checks.
  - Source evidence: `valid_count`, `invalid_count`, `validity_rate`, and component pass-rate columns.
  - Allowed wording: "Generated constrained-feature examples are filtered by the implemented validity checks."
  - Forbidden stronger wording: "The attacks are packet-realizable."

## Caveats

- The run validates a physical Jetson Orin NX Tier-E execution path, not a final leaderboard claim.
- Feature-space constrained attacks are not packet replay or simulator validated.
- Energy is Jetson INA3221 VDD_IN module-power telemetry, not wall-power or calibrated board-input energy.
- Dataset-specific split limits are documented in the split manifest and should be carried into manuscript claims.
