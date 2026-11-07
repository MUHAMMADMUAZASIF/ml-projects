"""Flight delay predictor: probability that a flight arrives 15+ minutes late.

Model : LightGBM classifier (falls back to sklearn HistGradientBoosting if LightGBM is missing)
Input : CSV with BTS columns (Reporting_Airline, Origin, Dest, Month, DayOfWeek, CRSDepTime,
        Distance, ArrDel15) and optionally weather columns (precip_mm, wind_kts, visibility_mi, temp_f)
Run   : python train.py --csv data/flights.csv
Output: outputs/model.joblib (used by app.py), outputs/metrics.json
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

CATEGORICAL = ["Reporting_Airline", "Origin", "Dest"]
NUMERIC = ["Month", "DayOfWeek", "dep_hour", "Distance"]
WEATHER = ["precip_mm", "wind_kts", "visibility_mi", "temp_f"]


def make_classifier(pos_weight=1.0):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=42,
                              verbose=-1), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42,
                                              categorical_features="from_dtype"), "sklearn HistGradientBoosting (fallback)"


def prepare(df, categories=None):
    X = df.copy()
    X["dep_hour"] = (X["CRSDepTime"] // 100).clip(0, 23)
    num = NUMERIC + [c for c in WEATHER if c in X.columns]
    X = X[CATEGORICAL + num].copy()
    cats = categories or {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X, cats


def scores(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4), "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4), "f1": round(f1_score(y, p, zero_division=0), 4)}


def main(csv_path="data/flights.csv", out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["ArrDel15"])
    X, cats = prepare(df)
    y = df["ArrDel15"].astype(int).values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp)

    model, model_name = make_classifier()
    model.fit(Xtr, ytr)
    pva, pte = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]

    # delays are the minority class -> also pick the threshold that maximises F1 on validation data
    pr, rc, th = precision_recall_curve(yva, pva)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    best = float(th[np.argmax(f1s)])

    metrics = {"_model": model_name, "delay_rate": round(float(y.mean()), 4),
               "roc_auc": round(roc_auc_score(yte, pte), 4),
               "threshold_0.50": scores(yte, pte >= 0.5),
               f"threshold_tuned_{best:.2f}": scores(yte, pte >= best),
               "best_threshold": round(best, 4), "train_rows": len(Xtr), "test_rows": len(Xte)}

    route_dist = df.groupby(["Origin", "Dest"])["Distance"].mean().round(0)
    meta = {"categories": cats, "best_threshold": best, "weather_cols": [c for c in WEATHER if c in df.columns],
            "route_distance": {f"{o}-{d}": v for (o, d), v in route_dist.items()},
            "mean_distance": float(df.Distance.mean())}
    joblib.dump({"model": model, "meta": meta}, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/flights.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.csv, a.out)
