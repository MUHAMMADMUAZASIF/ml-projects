# Results (demo/synthetic data, sklearn gradient-boosting fallback)

> These numbers come from the generated demo datasets, NOT real-world data. Re-run on real data (see each README) for real scores. With LightGBM/XGBoost installed, numbers will differ slightly.

## 1. Traffic Forecaster (congestion = speed < 45 mph)

| Horizon | MAE (mph) | R2 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 15min | 1.305 | 0.8888 | 0.9865 | 0.9032 | 0.775 | 0.8342 |
| 30min | 1.538 | 0.8206 | 0.9817 | 0.8791 | 0.6783 | 0.7658 |
| 60min | 1.886 | 0.7147 | 0.9796 | 0.8919 | 0.6111 | 0.7253 |

## 2. Airbnb Pricing (premium = price above median $130.0)

| MAE ($) | R2 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 28.91 | 0.891 | 0.939 | 0.9385 | 0.9366 | 0.9376 |

## 3. Flight Delay (ArrDel15)

ROC-AUC: 0.645 | delay rate: 0.1928

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.50 | 0.8071 | 0.4833 | 0.01 | 0.0197 |
| tuned_0.18 | 0.5861 | 0.2621 | 0.6323 | 0.3706 |
