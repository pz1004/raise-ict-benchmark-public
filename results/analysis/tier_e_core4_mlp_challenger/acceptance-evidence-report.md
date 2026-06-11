# Tier-E Core4 MLP Challenger Acceptance Evidence

## Scope

- `results/tables/tier_e_core4_mlp_challenger/table_raw_results.csv`: 320 rows (OK).
- `results/tables/tier_e_core4_mlp_challenger/table_main_results.csv`: 64 rows (OK).
- `results/tables/tier_e_random_control_mlp_challenger/table_raw_results.csv`: 40 rows (OK).
- `results/tables/tier_e_random_control_mlp_challenger/table_main_results.csv`: 8 rows (OK).
- Primary constrained threat: `a1_constrained_score_search`.
- Energy remains shared INA3221 `VDD_IN` module-power context, not per-model isolated energy.

## Pareto Frontier Check

- CICIDS2017: front=logistic_regression, mlp_sklearn; MLP on front=yes.
- CSE-CIC-IDS2018: front=mlp_sklearn; MLP on front=yes.
- TON_IoT: front=logistic_regression, mlp_sklearn; MLP on front=yes.
- UNSW-NB15: front=logistic_regression, mlp_sklearn, random_forest; MLP on front=yes.

## MLP Versus Best Classical Reference

| Dataset | Metric | MLP | Best classical | Classical model | Direction |
|---|---:|---:|---:|---|---|
| CICIDS2017 | clean_macro_f1 | 0.6512 | 0.7068 | logistic_regression | higher is better |
| CICIDS2017 | robust_utility | 0.6465 | 0.6416 | logistic_regression | higher is better |
| CICIDS2017 | asr | 0.696 | 0.657 | logistic_regression | lower is better |
| CICIDS2017 | p95_latency_ms | 0.0004786 | 0.0001513 | logistic_regression | lower is better |
| CICIDS2017 | peak_mem_mb | 44.63 | 2.062 | logistic_regression | lower is better |
| CICIDS2017 | service_cost | 6.844 | 5.487 | logistic_regression | lower is better |
| CSE-CIC-IDS2018 | clean_macro_f1 | 0.7782 | 0.6476 | logistic_regression | higher is better |
| CSE-CIC-IDS2018 | robust_utility | 0.7473 | 0.5672 | logistic_regression | higher is better |
| CSE-CIC-IDS2018 | asr | 0.4825 | 0.6658 | logistic_regression | lower is better |
| CSE-CIC-IDS2018 | p95_latency_ms | 0.00116 | 0.001333 | logistic_regression | lower is better |
| CSE-CIC-IDS2018 | peak_mem_mb | 2.506 | 0.1159 | logistic_regression | lower is better |
| CSE-CIC-IDS2018 | service_cost | 4.454 | 5.196 | logistic_regression | lower is better |
| TON_IoT | clean_macro_f1 | 0.9374 | 0.9862 | random_forest | higher is better |
| TON_IoT | robust_utility | 0.8984 | 0.869 | logistic_regression | higher is better |
| TON_IoT | asr | 0.174 | 0.2223 | logistic_regression | lower is better |
| TON_IoT | p95_latency_ms | 0.0004412 | 0.0001769 | logistic_regression | lower is better |
| TON_IoT | peak_mem_mb | 17.88 | 0.8265 | logistic_regression | lower is better |
| TON_IoT | service_cost | 0.9667 | 0.2021 | random_forest | lower is better |
| UNSW-NB15 | clean_macro_f1 | 0.8419 | 0.8667 | random_forest | higher is better |
| UNSW-NB15 | robust_utility | 0.8335 | 0.8619 | random_forest | higher is better |
| UNSW-NB15 | asr | 0.03287 | 0.01761 | random_forest | lower is better |
| UNSW-NB15 | p95_latency_ms | 0.00078 | 0.0002994 | logistic_regression | lower is better |
| UNSW-NB15 | peak_mem_mb | 40.83 | 1.886 | logistic_regression | lower is better |
| UNSW-NB15 | service_cost | 0.535 | 0.4147 | random_forest | lower is better |

## Random-Split Versus Held-Out Contrast

| Dataset | Model | Random-control clean F1 | Held-out score-search clean F1 |
|---|---|---:|---:|
| CICIDS2017 | extra_trees | 0.9946 | 0.5034 |
| CICIDS2017 | logistic_regression | 0.9389 | 0.7068 |
| CICIDS2017 | mlp_sklearn | 0.9839 | 0.6512 |
| CICIDS2017 | random_forest | 0.9964 | 0.5669 |
| CSE-CIC-IDS2018 | extra_trees | 0.8517 | 0.593 |
| CSE-CIC-IDS2018 | logistic_regression | 0.8168 | 0.6476 |
| CSE-CIC-IDS2018 | mlp_sklearn | 0.8726 | 0.7782 |
| CSE-CIC-IDS2018 | random_forest | 0.8652 | 0.5172 |

## Claim Candidates

- Allowed: The single-command Tier-E challenger run evaluates one CPU-compatible neural IDS challenger under the same dataset, split, threat, seed, hardware, validity, and profiling fields as the Core4 references.
- Allowed: The MLP challenger can be discussed only after the 320-row strict audit passes at `manifests/completion/benchmark_completion_audit_strict_tier_e_mlp_challenger.json`.
- Allowed: Random-control rows remain pipeline-sanity evidence and should be contrasted with held-out score-search rows.
- Forbidden: Do not claim a general neural IDS leaderboard, SOTA result, packet-level attack realizability, per-model isolated energy, or hardware-independent efficiency.

## Manuscript Use

- Do not update `jkics/jkics.tex` from the current 240-row claim until the 320-row audit is complete.
- If the 320-row audit passes, revise the manuscript from three classical baselines to three classical baselines plus one CPU-compatible MLP challenger.
