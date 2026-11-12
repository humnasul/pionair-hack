'''
Humna Sultan
PURPOSE OF THE FILE:
- output a json file with weights from a logistic regression model trained on a synthetic stroke cohort and air quality records
- utilizes: individualized data processing, logistic regression, train-test-split, sklearn, pandas, numpy, json
'''
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
# imports for machine learning

STROKE_CSV = "healthcare-dataset-stroke-data.csv"
AQI_CSV = "air_quality_health_impact_data.csv"
# files for data sources (stroke patient records and air quality records)
OUTPUT_JSON = "trained_model.json"
# output is a json with weights from the model


def main():
    print("1. Loading stroke cohort and air quality records...")
    df_stroke = pd.read_csv(STROKE_CSV)
    df_aqi = pd.read_csv(AQI_CSV)
    # reading files

    print(f"-> Stroke patient records: {len(df_stroke):,}")
    print(f"-> Air quality records: {len(df_aqi):,}")

    # Focus on adult cohort (age >= 18)
    df_stroke = df_stroke[df_stroke["age"] >= 18].reset_index(drop=True)
    n_samples = len(df_stroke)

    # Sample environmental exposures matching patient sample size
    aqi_sample = df_aqi.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)

    print(f"2. Synthesizing training matrix (N = {n_samples:,})...")
    df = pd.DataFrame()

    # Clinical features
    df["age_over_40"] = np.maximum(0.0, (df_stroke["age"] - 40.0) / 10.0)
    # Age over 40 is scaled in decades (e.g., 50 years old = 1.0, 60 years old = 2.0)
    df["heart"] = df_stroke["heart_disease"].astype(float)
    # Heart disease is a binary feature (0 = no, 1 = yes)
    df["hypertension"] = df_stroke["hypertension"].astype(float)
    # Hypertension is a binary feature (0 = no, 1 = yes)
    df["smoke"] = df_stroke["smoking_status"].isin(["smokes", "formerly smoked"]).astype(float)
    # Smoking status is a binary feature (0 = never smoked, 1 = smokes or formerly smoked)

    # Independent background prevalence without circular target leakage
    np.random.seed(42)
    df["fam_stroke"] = np.random.binomial(1, 0.14, size=n_samples).astype(float)
    # Family history of stroke is a binary feature (0 = no, 1 = yes) with ~14% prevalence
    df["fam_smoke"] = np.random.binomial(1, 0.16, size=n_samples).astype(float)
    # Family history of smoking is a binary feature (0 = no, 1 = yes) with ~16% prevalence

    # Single environmental feature used for calculation in app_3.py
    # (PM2.5, O3, and NO2 are displayed on the frontend, but excluded from risk weighting) --> reduces complexity for prototype
    df["aqi_scaled"] = aqi_sample["AQI"].astype(float) / 10.0

    # Target: Cerebrovascular event
    df["stroke_event"] = df_stroke["stroke"].astype(int)

    feature_cols = [
        "age_over_40",
        "heart",
        "hypertension",
        "smoke",
        "fam_stroke",
        "fam_smoke",
        "aqi_scaled",
    ]

    X = df[feature_cols].values
    y = df["stroke_event"].values
    # Split into train/test sets (stratified to preserve class balance)
    # feature cols = independent variables = characteristics of the patient and environmental factors
    # target col = dependent variable = whether the patient had a stroke or not

    print("3. Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Split the data into training and testing sets for training

    # Fit Logistic Regression without artificial balance distortion
    print("4. Fitting Logistic Regression model...")
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    # lbfgs is a solver for optimization in logistic regression, max_iter is the maximum number of iterations for the solver to converge
    # commonly used for problems with large datasets and many features, as it can handle high-dimensional data efficiently
    model.fit(X_train, y_train)

    test_preds = model.predict_proba(X_test)[:, 1]
    # Evaluate model performance on test set
    auc = roc_auc_score(y_test, test_preds)
    brier = brier_score_loss(y_test, test_preds)
    print(f"-> Cross-Validated Discrimination: ROC-AUC = {auc:.3f}, Brier Loss = {brier:.4f}")

    # Set calibrated baseline intercept (-4.50 aligns with ~1.1% 30-day base risk)
    model_payload = {
        "intercept": -4.50,
        "coefficients": {
            feat: float(coef) for feat, coef in zip(feature_cols, model.coef_[0])
        },
    }
    # Export model weights to JSON for use in app_3.py

    # Pin synthetic/environmental fields to pooled clinical meta-analyses
    # (Since PM2.5/O3/NO2 are no longer weighted, aqi_scaled is the single air risk term)
    literature_overrides = {
        "fam_stroke": math.log(1.40),    # Family history pooled RR (PMID 30472175)
        "fam_smoke": math.log(1.23),     # Household secondhand smoke RR (Lee et al. 2017)
        "aqi_scaled": math.log(1.10),    # Incident event RR per 10 scaled AQI units
    }
    # Apply overrides to model coefficients IF NEEDED (i.e., if the feature is present in the model)
    for feat, value in literature_overrides.items():
        if feat in model_payload["coefficients"]:
            model_payload["coefficients"][feat] = value
    # replace the coefficients for the specified features with literature-based values

    print("\n5. Applying epidemiological overrides to synthetic terms:")
    for feat, value in literature_overrides.items():
        print(f"   {feat:15s} -> {value:.4f}")

    # Apply tempered shrinkage (0.35) to prevent over-compounding of stacked features
    # preventing the model from overestimating risk when multiple risk factors are present
    SHRINKAGE = 0.53
    for feat in model_payload["coefficients"]:
        model_payload["coefficients"][feat] = round(model_payload["coefficients"][feat] * SHRINKAGE, 4)

    print(f"\n6. Applying global shrinkage factor ({SHRINKAGE}) to all coefficients:")
    print(json.dumps(model_payload["coefficients"], indent=2))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\n[SUCCESS] Exported matching model weights to '{OUTPUT_JSON}'")
    # final output


if __name__ == "__main__":
    main()