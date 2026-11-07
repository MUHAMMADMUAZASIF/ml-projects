"""Traffic speed forecaster: predicts speed 15 / 30 / 60 min ahead for every sensor.

Model : LightGBM (falls back to sklearn HistGradientBoosting if LightGBM is missing)
Input : wide CSV  ->  timestamp, <sensor_1>, <sensor_2>, ...  (5-minute data)
Run   : python train.py --csv data/traffic_speeds.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

STEP_MIN = 5
HORIZONS = {"15min": 3, "30min": 6, "60min": 12}   # in 5-min steps
LAGS = [0, 1, 2, 3, 6, 12]
CONGESTION_MPH = 45                                  # "congested" if speed < 45 mph


def make_regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=42, verbose=-1), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                             random_state=42), "sklearn HistGradientBoosting (fallback)"


def build_features(df, horizon):
    sensors = list(df.columns)
    net_mean = df.mean(axis=1)
    h = df.index.hour + df.index.minute / 60
    frames = []
    for i, s in enumerate(sensors):
        x = pd.DataFrame(index=df.index)
        for k in LAGS:
            x[f"lag_{k}"] = df[s].shift(k)
        x["roll_mean_6"] = df[s].rolling(6).mean()
        x["roll_std_6"] = df[s].rolling(6).std()
        up = df[sensors[i - 1]] if i > 0 else df[s]
        down = df[sensors[i + 1]] if i < len(sensors) - 1 else df[s]
        x["upstream"], x["downstream"] = up, down
        x["upstream_lag3"] = up.shift(3)
        x["network_mean"] = net_mean
        x["hour_sin"] = np.sin(2 * np.pi * h / 24)
        x["hour_cos"] = np.cos(2 * np.pi * h / 24)
        x["dow"] = df.index.dayofweek
        x["sensor_id"] = i
        x["current"] = df[s]
        x["target"] = df[s].shift(-horizon)
        x["timestamp"] = df.index
        frames.append(x)
    return pd.concat(frames).dropna()


def main(csv_path="data/traffic_speeds.csv", out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    cutoff = df.index[int(len(df) * 0.75)]            # time-based split (no leakage)
    results = {}
    for name, steps in HORIZONS.items():
        data = build_features(df, steps)
        train = data[data.timestamp <= cutoff - pd.Timedelta(minutes=steps * STEP_MIN)]
        test = data[data.timestamp > cutoff]
        feats = [c for c in data.columns if c not in ("target", "current", "timestamp")]
        model, model_name = make_regressor()
        model.fit(train[feats], train["target"])
        pred = np.clip(model.predict(test[feats]), 0, 85)
        y = test["target"].values

        yc, pc = y < CONGESTION_MPH, pred < CONGESTION_MPH
        base_mae = mean_absolute_error(y, test["current"])   # "speed stays the same" baseline
        results[name] = {
            "MAE_mph": round(mean_absolute_error(y, pred), 3),
            "RMSE_mph": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
            "R2": round(r2_score(y, pred), 4),
            "baseline_persistence_MAE_mph": round(base_mae, 3),
            "congestion_accuracy": round(accuracy_score(yc, pc), 4),
            "congestion_precision": round(precision_score(yc, pc, zero_division=0), 4),
            "congestion_recall": round(recall_score(yc, pc, zero_division=0), 4),
            "congestion_f1": round(f1_score(yc, pc, zero_division=0), 4),
            "congested_share_of_test": round(float(yc.mean()), 4),
            "train_rows": len(train), "test_rows": len(test),
        }
    results["_model"] = model_name
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Model: {model_name}   |   congestion = speed < {CONGESTION_MPH} mph")
    print(pd.DataFrame({k: v for k, v in results.items() if k != "_model"}).T.to_string())
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/traffic_speeds.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.csv, a.out)
