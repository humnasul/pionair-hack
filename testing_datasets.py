'''
Humna Sultan
PURPOSE OF THE FILE:
- Build the weights for the prototype logistic regression model and write them to a JSON for app.py.
- Patient terms fitted from a stroke cohort:
    * age_over_40, heart disease, hypertension
- Pinned to published literature (not fitted):
    * smoke_current (INTERSTROKE OR 1.67 for ischemic stroke)
    * smoke_former  (near-baseline after 5+ years of quitting, Surgeon General reports)
    * fam_stroke    (pooled RR 1.40, PMID 30472175)
    * fam_smoke     (household secondhand smoke RR 1.23, Lee et al. 2017)
- PM2.5 air term added from a published meta-analysis (Shah et al., BMJ 2015), not fitted and not shrunk.
- utilizes: pandas, numpy, scikit-learn, json
'''
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

STROKE_CSV = "healthcare-dataset-stroke-data.csv"
OUTPUT_JSON = "trained_model.json"

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
    print(f"-> Stroke patient records: {len(df_stroke):,}")

    # Focus on adult cohort (age >= 18)
    df_stroke = df_stroke[df_stroke["age"] >= 18].reset_index(drop=True)
    n_samples = len(df_stroke)

    print(f"2. Building training matrix (N = {n_samples:,})...")
    df = pd.DataFrame()

    # ----- Patient terms that are FITTED from Kaggle -----
    df["age_over_40"] = np.maximum(0.0, (df_stroke["age"] - 40.0) / 10.0)
    df["heart"] = df_stroke["heart_disease"].astype(float)
    df["hypertension"] = df_stroke["hypertension"].astype(float)

    # ----- Smoking as a three-level categorical (never / former / current) -----
    # The Kaggle dataset DOES record "smokes" vs "formerly smoked" vs "never smoked" vs "Unknown",
    # but it is a small observational snapshot and the two smoker categories carry very different
    # real-world risk. We therefore do NOT fit the smoking coefficients from the Kaggle outcome
    # column; we pin them to published literature below, same as the family history terms.
    smoking = df_stroke["smoking_status"]
    df["smoke_current"] = (smoking == "smokes").astype(float)
    df["smoke_former"] = (smoking == "formerly smoked").astype(float)
    # "never smoked" and "Unknown" -> both stay 0 (treated the same as never)

    # ----- Simulated family-history fields, independent of the outcome -----
    np.random.seed(42)
    df["fam_stroke"] = np.random.binomial(1, 0.14, size=n_samples).astype(float)
    df["fam_smoke"] = np.random.binomial(1, 0.16, size=n_samples).astype(float)

    # Air quality is NOT a training feature: the patient file has no air exposure, so training could not learn an air
    # effect. The air term is added after training from the published estimate above (PM2.5, O3, and NO2 are displayed on
    # the frontend; only PM2.5 feeds the risk model).

    df["stroke_event"] = df_stroke["stroke"].astype(int)

    # Only three patient terms are fitted from data.
    fitted_feature_cols = ["age_over_40", "heart", "hypertension"]
    # The shipped model also uses the four pinned terms below.
    shipped_feature_cols = fitted_feature_cols + [
        "smoke_current", "smoke_former", "fam_stroke", "fam_smoke",
    ]

    X_fit = df[fitted_feature_cols].values
    y = df["stroke_event"].values

    print("3. Splitting into train/test sets (row-based, stratified)...")
    row_idx = np.arange(n_samples)
    idx_train, idx_test = train_test_split(row_idx, test_size=0.20, random_state=42, stratify=y)
    X_fit_train, X_fit_test = X_fit[idx_train], X_fit[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print("4. Fitting Logistic Regression model (3 patient terms, from data)...")
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_fit_train, y_train)

    test_preds = model.predict_proba(X_fit_test)[:, 1]
    auc = roc_auc_score(y_test, test_preds)
    brier = brier_score_loss(y_test, test_preds)
    print(f"-> Held-out test set (fitted 3-term model, BEFORE overrides/shrinkage): ROC-AUC = {auc:.3f}, Brier Loss = {brier:.4f}")

    # Hand-set baseline intercept: sigmoid(-4.50) = ~1.1% for someone age 40 or under with no risk factors at the
    # reference PM2.5 level. This is a design assumption for a short-term risk. It is not fitted to, or validated against,
    # incidence data.
    model_payload = {
        "intercept": -4.50,
        "coefficients": {
            feat: float(coef) for feat, coef in zip(fitted_feature_cols, model.coef_[0])
        },
    }

    # Pin the smoking and family-history fields to published literature.
    # These replace any value for the same feature. Smoking is added fresh because it is not fitted.
    literature_overrides = {
        "smoke_current": math.log(1.67),   # INTERSTROKE OR ~1.67 for ischemic stroke (current smoker vs never)
        "smoke_former":  math.log(1.05),   # Former smoker risk returns to about baseline after 5+ years (Surgeon General)
        "fam_stroke":    math.log(1.40),   # Family history pooled RR (PMID 30472175)
        "fam_smoke":     math.log(1.23),   # Household secondhand smoke RR (Lee et al. 2017)
    }
    for feat, value in literature_overrides.items():
        model_payload["coefficients"][feat] = value

    print("\n5. Applying literature anchors to the four pinned coefficients:")
    for feat, value in literature_overrides.items():
        print(f"   {feat:15s} -> {value:.4f}  (= ln of the cited RR)")

    # Apply shrinkage (0.53) to prevent over-compounding of stacked features.
    # NOTE: this is applied AFTER the literature overrides, so the shipped family-history and smoking effects are also
    # shrunk. The air term below is deliberately NOT shrunk: it is an already-pooled published estimate, and shrinking it
    # would stop it matching the study it is cited from.
    SHRINKAGE = 0.53
    for feat in model_payload["coefficients"]:
        model_payload["coefficients"][feat] = round(model_payload["coefficients"][feat] * SHRINKAGE, 4)

    print(f"\n6. Applying global shrinkage factor ({SHRINKAGE}) to all coefficients:")
    print(json.dumps(model_payload["coefficients"], indent=2))

    # Literature air term (not fitted, not shrunk): risk rises by ln(RR) per 10 ug/m3 of PM2.5 above the reference level
    model_payload["air"] = {
        "pm25_log_rr_per_10ug": round(math.log(AIR_RR_PER_10UG), 6),
        "pm25_rr_per_10ug": AIR_RR_PER_10UG,
        "reference_pm25": AIR_REFERENCE_PM25,
        "source": "Shah et al., BMJ 2015;350:h1295 (doi:10.1136/bmj.h1295): short-term PM2.5 and stroke admission or mortality, RR 1.011 (95% CI 1.011 to 1.012) per 10 ug/m3",
    }
    print(f"\n6b. Air term from literature: RR {AIR_RR_PER_10UG} per 10 ug/m3 PM2.5 above {AIR_REFERENCE_PM25} ug/m3 (ln = {model_payload['air']['pm25_log_rr_per_10ug']})")

    # Evaluate the model that is actually shipped (same test rows, final intercept and coefficients, air at reference).
    X_shipped_test = df.loc[idx_test, shipped_feature_cols].values
    shipped_coefs = np.array([model_payload["coefficients"][f] for f in shipped_feature_cols])
    shipped_logit = model_payload["intercept"] + X_shipped_test @ shipped_coefs
    shipped_preds = 1.0 / (1.0 + np.exp(-shipped_logit))
    shipped_auc = roc_auc_score(y_test, shipped_preds)
    shipped_brier = brier_score_loss(y_test, shipped_preds)
    print(f"\n7. Shipped model on the same held-out rows (air at the reference level, so the air term adds nothing): ROC-AUC = {shipped_auc:.3f}, Brier Loss = {shipped_brier:.4f}")

    # Same evaluation using only patient terms (no family history)
    patient_terms = ["age_over_40", "heart", "hypertension", "smoke_current", "smoke_former"]
    X_patient_test = df.loc[idx_test, patient_terms].values
    patient_coefs = np.array([model_payload["coefficients"][f] for f in patient_terms])
    patient_preds = 1.0 / (1.0 + np.exp(-(model_payload["intercept"] + X_patient_test @ patient_coefs)))
    patient_auc = roc_auc_score(y_test, patient_preds)
    print(f"   Using only the 5 patient-data terms (age, heart disease, hypertension, smoke_current, smoke_former): ROC-AUC = {patient_auc:.3f}")
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
        "held_out_shipped_model_patient_terms_only": {"roc_auc": round(float(patient_auc), 3)},
        "air_term": "literature, not fitted and not shrunk (see 'air')",
        "note": "Prototype weights: smoking (current/former) and family history are pinned to literature; the air term is from a published meta-analysis; intercept is hand-set; not clinically validated.",
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=4)

    print(f"\n[SUCCESS] Exported model weights to '{OUTPUT_JSON}'")


if __name__ == "__main__":
    main()