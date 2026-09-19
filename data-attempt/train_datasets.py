import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

STROKE_CSV = "healthcare-dataset-stroke-data.csv"
AQI_CSV = "air_quality_health_impact_data.csv"
OUTPUT_JSON = "trained_model.json"


def main():
    print("1. Loading stroke cohort and air quality records...")
    df_stroke = pd.read_csv(STROKE_CSV)
    df_aqi = pd.read_csv(AQI_CSV)

    print(f"-> Stroke patient records: {len(df_stroke):,}")
    print(f"-> Air quality records: {len(df_aqi):,}")

    # Focus on adult cohort
    df_stroke = df_stroke[df_stroke["age"] >= 18].reset_index(drop=True)
    n_samples = len(df_stroke)

    # Sample environmental exposures matching patient sample size
    aqi_sample = df_aqi.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)

    print(f"2. Synthesizing cross-sectional training matrix (N = {n_samples:,})...")
    df = pd.DataFrame()

    # Clinical features
    df["age_over_40"] = np.maximum(0.0, (df_stroke["age"] - 40.0) / 10.0)
    df["heart"] = df_stroke["heart_disease"].astype(float)
    df["hypertension"] = df_stroke["hypertension"].astype(float)
    df["smoke"] = df_stroke["smoking_status"].isin(["smokes", "formerly smoked"]).astype(float)

    # Option 2: True independent family background prevalence (avoids target leakage)
    np.random.seed(42)
    df["fam_stroke"] = np.random.binomial(1, 0.14, size=n_samples).astype(float)
    df["fam_smoke"] = np.random.binomial(1, 0.16, size=n_samples).astype(float)

    # Environmental measurements
    df["aqi"] = aqi_sample["AQI"].astype(float)
    df["aqi_scaled"] = df["aqi"] / 10.0

    # Non-linear interaction terms
    df["aqi_x_heart"] = df["aqi_scaled"] * df["heart"]
    df["aqi_x_hypertension"] = df["aqi_scaled"] * df["hypertension"]
    df["aqi_x_age"] = df["aqi_scaled"] * df["age_over_40"]

    # Target: Acute cerebrovascular/stroke outcome
    df["stroke_event"] = df_stroke["stroke"].astype(int)

    feature_cols = [
        "age_over_40",
        "heart",
        "hypertension",
        "smoke",
        "fam_stroke",
        "fam_smoke",
        "aqi_scaled",
        "aqi_x_heart",
        "aqi_x_hypertension",
        "aqi_x_age",
    ]

    X = df[feature_cols].values
    y = df["stroke_event"].values

    print("3. Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("4. Fitting Logistic Regression model with balanced weighting...")
    model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    test_preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_preds)
    brier = brier_score_loss(y_test, test_preds)
    print(f"-> Validation Results: ROC-AUC = {auc:.3f}, Brier Loss = {brier:.4f}")

    # Export learned parameters to clean JSON
    model_payload = {
        "intercept": float(model.intercept_[0]),
        "coefficients": {
            feat: float(coef) for feat, coef in zip(feature_cols, model.coef_[0])
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\n[SUCCESS] Successfully exported model weights to '{OUTPUT_JSON}':")
    print(json.dumps(model_payload, indent=2))


if __name__ == "__main__":
    main()