# Model Prediction Accuracy and Loss Tables

- Forecast origin date: `2026-04-07`
- Truth date: `2026-04-08`
- Return/loss unit: log return. Percent-return columns are provided for readability.
- `base_view_step_00` is the app's base simulation view; `warm_refresh_step_25` is the final intraday refresh view.
- base artifact training metadata has blank/NaN CV metrics, so this report computes evaluation metrics from simulation predictions and actual bars.
- warm/current artifact training metadata has blank/NaN CV metrics, so this report computes evaluation metrics from simulation predictions and actual bars.

## Model Artifacts

| model_view | artifact_path | model_id | train_profile | production_mode | training_date | effective_as_of_date | n_rows | n_features | training_runtime_sec | stored_cv_metrics_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_artifact | model_artifacts/nibi_2026-04-16_job12292965/base | base_20260406_20260416T172942Z | base | base | 2026-04-16 | 2026-04-06 | 6396007 | 154 | 1702.225 | False |
| warm_current_artifact | model_artifacts/nibi_2026-04-16_job12292965/current | warm_refresh_20260406_20260416T211359Z | warm_refresh | refresh | 2026-04-16 | 2026-04-06 | 13104 | 154 | 8.166 | False |

## Base vs Warm Refresh

| model_view | step | slot_label | rows | direction_accuracy | final_mse_loss | final_horizon_rmse | final_horizon_mae | path_mse_loss | path_rmse | path_mae | final_price_rmse | final_price_mae | mean_predicted_return_pct | mean_actual_return_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_view_step_00 | 0 | 09:30 | 504 | 48.02% | 0.001887 | 0.043434 | 0.034640 | 0.001960 | 0.044277 | 0.034532 | 17.831468 | 8.370921 | 0.023449 | 2.572430 |
| warm_refresh_step_25 | 25 | 15:45 | 504 | 40.67% | 0.001668 | 0.040845 | 0.033424 | 0.001901 | 0.043597 | 0.034868 | 19.184186 | 8.252219 | -0.189029 | 2.336631 |

## Accuracy and Loss by Refresh Step

| step | slot_label | rows | direction_accuracy | final_horizon_rmse | final_mse_loss |
| --- | --- | --- | --- | --- | --- |
| 0 | 09:30 | 504 | 48.02% | 0.043434 | 0.001887 |
| 1 | 09:45 | 504 | 48.41% | 0.043680 | 0.001908 |
| 2 | 10:00 | 504 | 49.21% | 0.044134 | 0.001948 |
| 3 | 10:15 | 504 | 45.63% | 0.042326 | 0.001791 |
| 4 | 10:30 | 504 | 42.46% | 0.042918 | 0.001842 |
| 5 | 10:45 | 504 | 42.66% | 0.044298 | 0.001962 |
| 6 | 11:00 | 504 | 46.23% | 0.045832 | 0.002101 |
| 7 | 11:15 | 504 | 43.45% | 0.044699 | 0.001998 |
| 8 | 11:30 | 504 | 46.23% | 0.045124 | 0.002036 |
| 9 | 11:45 | 504 | 44.44% | 0.044619 | 0.001991 |
| 10 | 12:00 | 504 | 41.27% | 0.043687 | 0.001909 |
| 11 | 12:15 | 504 | 40.67% | 0.042831 | 0.001835 |
| 12 | 12:30 | 504 | 43.06% | 0.042300 | 0.001789 |
| 13 | 12:45 | 504 | 46.83% | 0.041606 | 0.001731 |
| 14 | 13:00 | 504 | 47.22% | 0.041791 | 0.001746 |
| 15 | 13:15 | 504 | 48.41% | 0.042015 | 0.001765 |
| 16 | 13:30 | 504 | 44.25% | 0.041277 | 0.001704 |
| 17 | 13:45 | 504 | 45.83% | 0.041306 | 0.001706 |
| 18 | 14:00 | 504 | 46.63% | 0.041499 | 0.001722 |
| 19 | 14:15 | 504 | 47.02% | 0.041712 | 0.001740 |
| 20 | 14:30 | 504 | 48.02% | 0.042027 | 0.001766 |
| 21 | 14:45 | 504 | 52.58% | 0.042846 | 0.001836 |
| 22 | 15:00 | 504 | 50.60% | 0.042903 | 0.001841 |
| 23 | 15:15 | 504 | 42.06% | 0.041844 | 0.001751 |
| 24 | 15:30 | 504 | 44.84% | 0.042046 | 0.001768 |
| 25 | 15:45 | 504 | 40.67% | 0.040845 | 0.001668 |

## Top Warm-Refresh Predictions With Actual Outcomes

| symbol | slot_label | cutoff_close | predicted_final_close | actual_final_close | predicted_final_return_pct | actual_final_return_pct | predicted_direction | actual_direction | correct_direction | final_abs_log_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSKY | 15:45 | 10.91 | 11.874041814997696 | 10.82 | 8.836314 | -0.824931 | up | down | no | 0.092958 |
| CF | 15:45 | 133.79 | 138.90865029548098 | 126.17 | 3.825884 | -5.695493 | up | down | no | 0.096186 |
| MCD | 15:45 | 304.89 | 314.36980968102966 | 307.06 | 3.109256 | 0.711732 | up | up | yes | 0.023527 |
| KEYS | 15:45 | 300.65 | 309.6656734357932 | 318.06 | 2.998727 | 5.790787 | up | up | yes | 0.026747 |
| UNH | 15:45 | 307.73 | 316.7766904517145 | 305.97 | 2.939814 | -0.571930 | up | down | no | 0.034710 |
| DPZ | 15:45 | 368.15 | 378.39064669262405 | 376.28 | 2.781651 | 2.208339 | up | up | yes | 0.005594 |
| CI | 15:45 | 274.26 | 281.8773256480632 | 277.52 | 2.777410 | 1.188653 | up | up | yes | 0.015579 |
| ELV | 15:45 | 311.97 | 320.5654041131862 | 318.2 | 2.755202 | 1.996987 | up | up | yes | 0.007406 |
| GE | 15:45 | 288.69 | 296.27492663413784 | 308.1 | 2.627360 | 6.723475 | up | up | yes | 0.039137 |
| MKC | 15:45 | 51.0 | 52.18841857700453 | 50.5 | 2.330233 | -0.980392 | up | down | no | 0.032887 |
| CBOE | 15:45 | 297.18 | 304.0220309850293 | 296.91 | 2.302319 | -0.090854 | up | down | no | 0.023671 |
| OMC | 15:45 | 75.65 | 77.24856002321589 | 76.84 | 2.113100 | 1.573034 | up | up | yes | 0.005303 |
| APA | 15:45 | 42.975 | 43.853895834296736 | 38.75 | 2.045133 | -9.831297 | up | down | no | 0.123733 |
| HUM | 15:45 | 197.13 | 201.10349725647174 | 198.36 | 2.015674 | 0.623954 | up | up | yes | 0.013736 |
| CCI | 15:45 | 84.93 | 86.58798553189281 | 85.55 | 1.952179 | 0.730013 | up | up | yes | 0.012060 |
| DVN | 15:45 | 49.95 | 50.89566262874153 | 47.93 | 1.893218 | -4.044044 | up | down | no | 0.060036 |
| CVNA | 15:45 | 320.22 | 325.85682527219177 | 338.7 | 1.760298 | 5.771032 | up | up | yes | 0.038657 |
| PM | 15:45 | 157.5 | 159.9282825531099 | 160.94 | 1.541767 | 2.184127 | up | up | yes | 0.006306 |
| XOM | 15:45 | 163.83 | 166.32509723538973 | 156.23 | 1.522979 | -4.638955 | up | down | no | 0.062615 |
| INTC | 15:45 | 52.91 | 53.71010538146672 | 58.94 | 1.512201 | 11.396711 | up | up | yes | 0.092919 |
