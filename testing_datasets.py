'''
Humna Sultan
PURPOSE OF THE FILE:
- output a json file with weights from a logistic regression model trained on a stroke cohort (patient terms are fitted;
  family-history terms are simulated and pinned to literature) plus a literature-based PM2.5 air term (not fitted)
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
# file for the data source (stroke patient records). The Kaggle air quality CSV is no longer used: its AQI values were
# assigned to patients at random, so they could not teach the model anything about air.
OUTPUT_JSON = "trained_model.json"
# output is a json with weights from the model

# Air term, taken from a published meta-analysis instead of being fitted (see README references):
# Shah et al., BMJ 2015;350:h1295 - short-term PM2.5 exposure and stroke hospital admission or mortality,
# pooled relative risk 1.011 (95% CI 1.011 to 1.012) per 10 ug/m3. The studies it pooled had typical PM2.5 levels of
# roughly 6 to 34 ug/m3, so effects at much higher PM2.5 (e.g., wildfire smoke) are extrapolated.
AIR_RR_PER_10UG = 1.011
# Reference PM2.5 level: air at or below this adds no risk. 5 ug/m3 is close to the cleanest NYC neighborhood
# average in the NYC Open Data file (5.26 ug/m3 in 2024).
AIR_REFERENCE_PM25 = 5.0


def main():
    print("1. Loading stroke cohort...")
    df_stroke = pd.read_csv(STROKE_CSV)
    # reading file

    print(f"-> Stroke patient records: {len(df_stroke):,}")

    # Focus on adult cohort (age >= 18)
    df_stroke = df_stroke[df_stroke["age"] >= 18].reset_index(drop=True)
    n_samples = len(df_stroke)

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

    # Air quality is NOT a training feature: the patient file has no air exposure, so training could not learn an air
    # effect. The air term is added after training from the published estimate above (PM2.5, O3, and NO2 are displayed on
    # the frontend; only PM2.5 feeds the risk model).

    # Target: Cerebrovascular event
    df["stroke_event"] = df_stroke["stroke"].astype(int)

    feature_cols = [
        "age_over_40",
        "heart",
        "hypertension",
        "smoke",
        "fam_stroke",
        "fam_smoke",
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
    print(f"-> Held-out test set (20% split, fitted model BEFORE overrides/shrinkage): ROC-AUC = {auc:.3f}, Brier Loss = {brier:.4f}")

    # Hand-set baseline intercept: sigmoid(-4.50) = ~1.1% for someone age 40 or under with no risk factors at the
    # reference PM2.5 level. This is a design assumption for a short-term risk. It is not fitted to, or validated against,
    # incidence data.
    model_payload = {
        "intercept": -4.50,
        "coefficients": {
            feat: float(coef) for feat, coef in zip(feature_cols, model.coef_[0])
        },
    }
    # Export model weights to JSON for use in app.py

    # Pin the simulated family-history fields to pooled clinical meta-analyses
    literature_overrides = {
        "fam_stroke": math.log(1.40),    # Family history pooled RR (PMID 30472175)
        "fam_smoke": math.log(1.23),     # Household secondhand smoke RR (Lee et al. 2017)
    }
    # Apply overrides to model coefficients IF NEEDED (i.e., if the feature is present in the model)
    for feat, value in literature_overrides.items():
        if feat in model_payload["coefficients"]:
            model_payload["coefficients"][feat] = value
    # replace the coefficients for the specified features with literature-based values

    print("\n5. Applying epidemiological overrides to simulated terms:")
    for feat, value in literature_overrides.items():
        print(f"   {feat:15s} -> {value:.4f}")

    # Apply shrinkage (0.53) to prevent over-compounding of stacked features
    # preventing the model from overestimating risk when multiple risk factors are present
    # NOTE: this is applied AFTER the literature overrides, so the shipped family-history effects are also shrunk.
    # The air term below is deliberately NOT shrunk: it is an already-pooled published estimate, and shrinking it would
    # stop it matching the study it is cited from.
    SHRINKAGE = 0.53
    for feat in model_payload["coefficients"]:
        model_payload["coefficients"][feat] = round(model_payload["coefficients"][feat] * SHRINKAGE, 4)

    print(f"\n6. Applying global shrinkage factor ({SHRINKAGE}) to the six fitted/pinned coefficients:")
    print(json.dumps(model_payload["coefficients"], indent=2))

    # Literature air term (not fitted, not shrunk): risk rises by ln(RR) per 10 ug/m3 of PM2.5 above the reference level
    model_payload["air"] = {
        "pm25_log_rr_per_10ug": round(math.log(AIR_RR_PER_10UG), 6),
        "pm25_rr_per_10ug": AIR_RR_PER_10UG,
        "reference_pm25": AIR_REFERENCE_PM25,
        "source": "Shah et al., BMJ 2015;350:h1295 (doi:10.1136/bmj.h1295): short-term PM2.5 and stroke admission or mortality, RR 1.011 (95% CI 1.011 to 1.012) per 10 ug/m3",
    }
    print(f"\n6b. Air term from literature: RR {AIR_RR_PER_10UG} per 10 ug/m3 PM2.5 above {AIR_REFERENCE_PM25} ug/m3 (ln = {model_payload['air']['pm25_log_rr_per_10ug']})")

    # Evaluate the model that is actually shipped (same test rows, final intercept and coefficients)
    shipped_logit = model_payload["intercept"] + X_test @ np.array([model_payload["coefficients"][f] for f in feature_cols])
    shipped_preds = 1.0 / (1.0 + np.exp(-shipped_logit))
    shipped_auc = roc_auc_score(y_test, shipped_preds)
    shipped_brier = brier_score_loss(y_test, shipped_preds)
    print(f"\n7. Shipped model on the same held-out rows (air at the reference level, so the air term adds nothing): ROC-AUC = {shipped_auc:.3f}, Brier Loss = {shipped_brier:.4f}")
    # Same evaluation using only the 4 terms that come from real patient rows
    real_terms = ["age_over_40", "heart", "hypertension", "smoke"]
    real_idx = [feature_cols.index(f) for f in real_terms]
    real_coefs = np.array([model_payload["coefficients"][f] for f in real_terms])
    real_preds = 1.0 / (1.0 + np.exp(-(model_payload["intercept"] + X_test[:, real_idx] @ real_coefs)))
    real_auc = roc_auc_score(y_test, real_preds)
    print(f"   Using only the 4 patient-data terms (age, heart disease, hypertension, smoking): ROC-AUC = {real_auc:.3f}")
    print("   The family-history terms are simulated independently of the outcome, so they can only lower these metrics.")
    print(f"   Mean predicted risk = {shipped_preds.mean():.1%} vs observed 'stroke' rate in the test rows = {y_test.mean():.1%}.")
    print("   These are not directly comparable: the dataset label is 'has had a stroke', while the intercept is a hand-set short-term baseline.")

    # Provenance block for the paper/README. app.py only reads 'intercept' and 'coefficients', so this is safe to add.
    model_payload["metadata"] = {
        "training_rows": int(n_samples),
        "shrinkage": SHRINKAGE,
        "literature_overrides_before_shrinkage": {k: round(v, 4) for k, v in literature_overrides.items()},
        "held_out_fitted_model": {"roc_auc": round(float(auc), 3), "brier": round(float(brier), 4)},
        "held_out_shipped_model": {"roc_auc": round(float(shipped_auc), 3), "brier": round(float(shipped_brier), 4)},
        "held_out_shipped_model_patient_terms_only": {"roc_auc": round(float(real_auc), 3)},
        "air_term": "literature, not fitted and not shrunk (see 'air')",
        "note": "Prototype weights: family history is simulated and pinned to literature; the air term is from a published meta-analysis; intercept is hand-set; not clinically validated.",
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\n[SUCCESS] Exported matching model weights to '{OUTPUT_JSON}'")
    # final output


if __name__ == "__main__":
    main()