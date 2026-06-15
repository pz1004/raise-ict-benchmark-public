# RAISE-ICT Tier-E Timed Core4 HGB MLP Stats Appendix

## Paired Constrained-Attack Drop

| dataset         | model_id               |   n_pairs |   mean_robust_drop |   std_robust_drop |   ci_low |   ci_high |   mean_asr |   mean_validity_rate |   mean_valid_count |   mean_invalid_count |   wilcoxon_p |   holm_p |
|:----------------|:-----------------------|----------:|-------------------:|------------------:|---------:|----------:|-----------:|---------------------:|-------------------:|---------------------:|-------------:|---------:|
| CICIDS2017      | extra_trees            |        10 |             0.0211 |            0.0219 |   0.0075 |    0.0347 |     0.9004 |               0.9955 |         89595.2000 |             404.8000 |       0.0039 |   0.0391 |
| CICIDS2017      | hist_gradient_boosting |        10 |             0.0275 |            0.0453 |  -0.0006 |    0.0556 |     0.8348 |               0.9955 |         89595.2000 |             404.8000 |       0.0020 |   0.0391 |
| CICIDS2017      | logistic_regression    |        10 |             0.0598 |            0.0295 |   0.0416 |    0.0781 |     0.6634 |               0.9955 |         89595.2000 |             404.8000 |       0.0020 |   0.0391 |
| CICIDS2017      | mlp_sklearn            |        10 |             0.0047 |            0.0050 |   0.0016 |    0.0078 |     0.6922 |               0.9955 |         89595.2000 |             404.8000 |       0.0020 |   0.0391 |
| CICIDS2017      | random_forest          |        10 |             0.0993 |            0.0659 |   0.0585 |    0.1401 |     0.9384 |               0.9955 |         89595.2000 |             404.8000 |       0.0020 |   0.0391 |
| CSE-CIC-IDS2018 | extra_trees            |        10 |             0.0801 |            0.0393 |   0.0558 |    0.1044 |     0.8420 |               0.9980 |          4990.2000 |               9.8000 |       0.0020 |   0.0391 |
| CSE-CIC-IDS2018 | hist_gradient_boosting |        10 |             0.0082 |            0.0174 |  -0.0026 |    0.0189 |     0.9880 |               0.9980 |          4990.2000 |               9.8000 |       0.0137 |   0.0391 |
| CSE-CIC-IDS2018 | logistic_regression    |        10 |             0.0775 |            0.0343 |   0.0562 |    0.0988 |     0.6576 |               0.9980 |          4990.2000 |               9.8000 |       0.0020 |   0.0391 |
| CSE-CIC-IDS2018 | mlp_sklearn            |        10 |             0.0101 |            0.0299 |  -0.0084 |    0.0287 |     0.4427 |               0.9980 |          4990.2000 |               9.8000 |       0.0645 |   0.0645 |
| CSE-CIC-IDS2018 | random_forest          |        10 |             0.1087 |            0.0612 |   0.0707 |    0.1466 |     0.9647 |               0.9980 |          4990.2000 |               9.8000 |       0.0020 |   0.0391 |
| TON_IoT         | extra_trees            |        10 |             0.1056 |            0.0253 |   0.0899 |    0.1213 |     0.2523 |               0.9999 |         36039.3000 |               3.7000 |       0.0020 |   0.0391 |
| TON_IoT         | hist_gradient_boosting |        10 |             0.3025 |            0.0612 |   0.2646 |    0.3405 |     0.6025 |               0.9999 |         36039.3000 |               3.7000 |       0.0020 |   0.0391 |
| TON_IoT         | logistic_regression    |        10 |             0.0043 |            0.0007 |   0.0039 |    0.0047 |     0.2223 |               0.9999 |         36039.3000 |               3.7000 |       0.0020 |   0.0391 |
| TON_IoT         | mlp_sklearn            |        10 |             0.0492 |            0.0260 |   0.0331 |    0.0652 |     0.1601 |               0.9999 |         36039.3000 |               3.7000 |       0.0020 |   0.0391 |
| TON_IoT         | random_forest          |        10 |             0.1357 |            0.0331 |   0.1152 |    0.1562 |     0.2700 |               0.9999 |         36039.3000 |               3.7000 |       0.0020 |   0.0391 |
| UNSW-NB15       | extra_trees            |        10 |             0.0020 |            0.0005 |   0.0017 |    0.0023 |     0.0242 |               0.9991 |         82260.0000 |              72.0000 |       0.0020 |   0.0391 |
| UNSW-NB15       | hist_gradient_boosting |        10 |             0.0031 |            0.0004 |   0.0028 |    0.0033 |     0.0394 |               0.9991 |         82260.0000 |              72.0000 |       0.0020 |   0.0391 |
| UNSW-NB15       | logistic_regression    |        10 |             0.0127 |            0.0002 |   0.0125 |    0.0128 |     0.0960 |               0.9991 |         82260.0000 |              72.0000 |       0.0020 |   0.0391 |
| UNSW-NB15       | mlp_sklearn            |        10 |             0.0054 |            0.0008 |   0.0049 |    0.0059 |     0.0368 |               0.9991 |         82260.0000 |              72.0000 |       0.0020 |   0.0391 |
| UNSW-NB15       | random_forest          |        10 |             0.0015 |            0.0004 |   0.0012 |    0.0018 |     0.0178 |               0.9991 |         82260.0000 |              72.0000 |       0.0020 |   0.0391 |

Wilcoxon signed-rank tests compare matched seed-level `a0_clean` and `a1_constrained_score_search` robust utility within each dataset and model. Holm correction is applied across the reported contrasts.

## Summary Table Source

`results/tables/tier_e_core4_hgb_mlp_timed/table_main_results.csv` contains mean, standard deviation, and normal-approximation 95% confidence intervals over 10 seeds.
