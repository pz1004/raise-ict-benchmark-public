# RAISE-ICT Tier-E Core4 Jetson Orin NX Stats Appendix

## Paired Constrained-Attack Drop

| dataset         | model_id            |   n_pairs |   mean_robust_drop |   std_robust_drop |   ci_low |   ci_high |   mean_asr |   mean_validity_rate |   mean_valid_count |   mean_invalid_count |   wilcoxon_p |   holm_p |
|:----------------|:--------------------|----------:|-------------------:|------------------:|---------:|----------:|-----------:|---------------------:|-------------------:|---------------------:|-------------:|---------:|
| CICIDS2017      | extra_trees         |         5 |             0.0169 |            0.0195 |  -0.0002 |    0.0341 |     0.9021 |               0.9935 |         89418.4000 |             581.6000 |       0.1250 |   0.7500 |
| CICIDS2017      | logistic_regression |         5 |             0.0613 |            0.0293 |   0.0355 |    0.0870 |     0.6570 |               0.9935 |         89418.4000 |             581.6000 |       0.0625 |   0.7500 |
| CICIDS2017      | random_forest       |         5 |             0.0757 |            0.0522 |   0.0299 |    0.1215 |     0.9055 |               0.9935 |         89418.4000 |             581.6000 |       0.0625 |   0.7500 |
| CSE-CIC-IDS2018 | extra_trees         |         5 |             0.0618 |            0.0282 |   0.0371 |    0.0865 |     0.8230 |               0.9980 |          4989.8000 |              10.2000 |       0.0625 |   0.7500 |
| CSE-CIC-IDS2018 | logistic_regression |         5 |             0.0815 |            0.0458 |   0.0414 |    0.1217 |     0.6658 |               0.9980 |          4989.8000 |              10.2000 |       0.0625 |   0.7500 |
| CSE-CIC-IDS2018 | random_forest       |         5 |             0.0715 |            0.0510 |   0.0268 |    0.1162 |     0.9715 |               0.9980 |          4989.8000 |              10.2000 |       0.0625 |   0.7500 |
| TON_IoT         | extra_trees         |         5 |             0.1173 |            0.0191 |   0.1006 |    0.1340 |     0.2740 |               0.9999 |         36039.4000 |               3.6000 |       0.0625 |   0.7500 |
| TON_IoT         | logistic_regression |         5 |             0.0044 |            0.0009 |   0.0037 |    0.0052 |     0.2223 |               0.9999 |         36039.4000 |               3.6000 |       0.0625 |   0.7500 |
| TON_IoT         | random_forest       |         5 |             0.1493 |            0.0326 |   0.1207 |    0.1778 |     0.2954 |               0.9999 |         36039.4000 |               3.6000 |       0.0625 |   0.7500 |
| UNSW-NB15       | extra_trees         |         5 |             0.0021 |            0.0006 |   0.0016 |    0.0026 |     0.0243 |               0.9991 |         82260.0000 |              72.0000 |       0.0625 |   0.7500 |
| UNSW-NB15       | logistic_regression |         5 |             0.0126 |            0.0003 |   0.0124 |    0.0129 |     0.0959 |               0.9991 |         82260.0000 |              72.0000 |       0.0625 |   0.7500 |
| UNSW-NB15       | random_forest       |         5 |             0.0015 |            0.0005 |   0.0011 |    0.0019 |     0.0176 |               0.9991 |         82260.0000 |              72.0000 |       0.0625 |   0.7500 |

Wilcoxon signed-rank tests compare matched seed-level `a0_clean` and `a1_constrained_score_search` robust utility within each dataset and model. Holm correction is applied across the reported contrasts.

## Summary Table Source

`results/tables/tier_e_core4/table_main_results.csv` contains mean, standard deviation, and normal-approximation 95% confidence intervals over five seeds.
