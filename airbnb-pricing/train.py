"""Airbnb pricing engine: predict nightly price + explain what drives it.

Model : XGBoost (falls back to sklearn HistGradientBoosting if XGBoost is missing)
Explain: SHAP summary plot if `shap` is installed, else permutation importance
Run   : python train.py --csv data/airbnb_listings.csv
Output: outputs/model.joblib (used by app.py), outputs/metrics.json, feature importance
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split

CATEGORICAL = ["neighbourhood", "room_type"]
AMENITY_COLS = ["wifi", "kitchen", "air_conditioning", "washer", "free_parking", "gym", "pool", "hot_tub"]


def make_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                             random_state=42), "sklearn HistGradientBoosting (fallback)"


def prepare(df):
    X = df.drop(columns=["price"]).copy()
    X["review_score"] = X["review_score"].fillna(X["review_score"].median())
    X = pd.get_dummies(X, columns=CATEGORICAL).astype(float)
    return X


def main(csv_path="data/airbnb_listings.csv", out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["price"])
    X, y = prepare(df), df["price"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model, model_name = make_regressor()
    model.fit(Xtr, np.log1p(ytr))                       # log-price target
    pred = np.expm1(model.predict(Xte))

    # classification view: is the listing "premium" (above the median training price)?
    thr = float(np.median(ytr))
    yc, pc = yte > thr, pred > thr
    metrics = {
        "_model": model_name, "premium_threshold_price": thr,
        "MAE_dollars": round(mean_absolute_error(yte, pred), 2),
        "RMSE_dollars": round(float(np.sqrt(mean_squared_error(yte, pred))), 2),
        "R2": round(r2_score(yte, pred), 4),
        "MAPE_percent": round(float(np.mean(np.abs(yte - pred) / yte) * 100), 2),
        "premium_accuracy": round(accuracy_score(yc, pc), 4),
        "premium_precision": round(precision_score(yc, pc), 4),
        "premium_recall": round(recall_score(yc, pc), 4),
        "premium_f1": round(f1_score(yc, pc), 4),
        "train_rows": len(Xtr), "test_rows": len(Xte),
    }

    # ---- explainability: which features add the most money ----
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sv = shap.TreeExplainer(model).shap_values(Xte.iloc[:1000])
        shap.summary_plot(sv, Xte.iloc[:1000], show=False)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150); plt.close()
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.Series(mean_abs, index=X.columns, name="mean_abs_shap_log_price")
    except Exception:
        r = permutation_importance(model, Xte.iloc[:1500], np.log1p(yte[:1500]), n_repeats=3, random_state=0)
        imp = pd.Series(r.importances_mean, index=X.columns, name="permutation_importance")
    imp.sort_values(ascending=False).to_csv(os.path.join(out_dir, "feature_importance.csv"))

    # ---- save everything the Streamlit app needs ----
    meta = {"columns": list(X.columns), "amenities": AMENITY_COLS,
            "neighbourhoods": sorted(df.neighbourhood.unique().tolist()),
            "room_types": sorted(df.room_type.unique().tolist()),
            "nb_latlon": df.groupby("neighbourhood")[["latitude", "longitude"]].mean().to_dict("index"),
            "review_score_median": float(df.review_score.median())}
    joblib.dump({"model": model, "meta": meta}, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nTop 10 price drivers:\n", imp.sort_values(ascending=False).head(10).round(4).to_string())
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/airbnb_listings.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.csv, a.out)
