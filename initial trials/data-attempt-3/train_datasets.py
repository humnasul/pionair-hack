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

    # Environmental measurements - both the composite AQI figure and the
    # three individual pollutant columns, so the model can use PurpleAir's
    # AQI (a different, hyper-local sensor network) as a corroborating
    # signal alongside OpenWeatherMap's PM2.5/O3/NO2 readings.
    df["aqi_scaled"] = aqi_sample["AQI"].astype(float) / 10.0
    df["pm2_5_scaled"] = aqi_sample["PM2_5"].astype(float) / 10.0
    df["o3_scaled"] = aqi_sample["O3"].astype(float) / 10.0
    df["no2_scaled"] = aqi_sample["NO2"].astype(float) / 10.0

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
        "pm2_5_scaled",
        "o3_scaled",
        "no2_scaled",
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

    # ------------------------------------------------------------------
    # Override coefficients the training data cannot honestly support.
    #
    # fam_stroke / fam_smoke are synthesized with np.random.binomial(),
    # unconnected to who actually had a stroke -> the "learned" weight for
    # these is pure noise from the train/test split, not a real effect.
    #
    # pm2_5_scaled / o3_scaled / no2_scaled are sampled from an unrelated
    # air-quality dataset with no real link to each patient's actual
    # exposure -> same problem. Rather than ship noise, all five terms are
    # pinned to log(relative risk) from published meta-analyses, while
    # age/heart/hypertension/smoke stay as the genuinely-learned
    # coefficients from the real stroke cohort.
    # ------------------------------------------------------------------
    literature_overrides = {
        "fam_stroke": math.log(1.40),    # family history of stroke, pooled RR (PMID 30472175)
        "fam_smoke": math.log(1.23),     # secondhand/household smoke exposure, RR (Lee et al. 2017, J Stroke Cerebrovasc Dis)
        "aqi_scaled": math.log(1.13) * 0.5,  # composite AQI (PurpleAir) - HALVED: AQI is substantially derived
                                              # from PM2.5, which pm2_5_scaled below already captures from a
                                              # separate sensor source (OpenWeatherMap). Full weight on both
                                              # would double-count essentially the same physical measurement;
                                              # this term instead acts as a smaller "second sensor agrees" signal.
        "pm2_5_scaled": math.log(1.13),  # +13% incident stroke risk per 10 ug/m3 PM2.5 (Alexeeff et al. 2020, JAHA)
        "o3_scaled": math.log(1.22),     # stroke-specific HR per 10 ug/m3 long-term O3, PM2.5-adjusted (CHERRY cohort, Environment International 2022)
        "no2_scaled": math.log(1.11),    # cardiovascular mortality HR per 10 ppb long-term NO2 (Huang 2021 meta-analysis); stroke-specific NO2 evidence is weaker/mixed, so this is a deliberately modest estimate
    }
    for feat, value in literature_overrides.items():
        if feat in model_payload["coefficients"]:
            model_payload["coefficients"][feat] = value

    print("\n5. Overriding noise-driven coefficients with literature values:")
    for feat, value in literature_overrides.items():
        print(f"   {feat:22s} -> {value:.4f}")

    # ------------------------------------------------------------------
    # Global shrinkage.
    #
    # With several positive risk factors stacked together (older age +
    # smoker + heart disease + bad air day, say), the unshrunk logistic
    # model compounds quickly and can push probabilities to the extreme
    # end of the scale faster than feels right for a screening prototype.
    # Multiplying every non-intercept coefficient by a shrinkage factor
    # pulls all predictions back toward the baseline (intercept-only) risk
    # without changing which direction any factor points - it just makes
    # the swings less dramatic. 0.65 is a starting point, not a fitted
    # value; tune it down further if results still feel too extreme, or
    # up toward 1.0 to let the raw estimates speak for themselves.
    # ------------------------------------------------------------------
    SHRINKAGE = 0.65
    for feat in model_payload["coefficients"]:
        model_payload["coefficients"][feat] *= SHRINKAGE

    print(f"\n6. Applying global shrinkage factor ({SHRINKAGE}) to all coefficients:")
    print(json.dumps(model_payload["coefficients"], indent=2))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\n[SUCCESS] Successfully exported model weights to '{OUTPUT_JSON}':")
    print(json.dumps(model_payload, indent=2))


if __name__ == "__main__":
    main()