# Pionair: Personalized Air Quality Communication Tool
## What's Pionair?
Pionair integrates live air pollution data into daily life through an app, care circle, and portable personalized device, making air quality information more accessible to vulnerable populations.
## File Directory:  
- The main branch (outside any folders) includes...   
The stroke dataset CSV (from Kaggle) is used by `testing_datasets.py`. `air_quality_health_impact_data.csv` is kept in the repo for reference but is no longer used (see Datasets).   
`testing_datasets.py` creates weights utilizing the CSV that is outputted as a file, `trained_model.json`. These weights are used as a basis of logistic regression in `app.py`.    
`app.py` is the interactive web app.    
NYC Open Data context (display only, does not change the risk score): `export.csv` is the NYC Open Data "Air Quality" export, `preprocess_nyc_data.py` turns it into the small `nyc_baseline.json`, `nyc_context.py` looks up a user's borough/neighborhood in it, and `verify_nyc_data.py` checks the data, the location lookup, and the model against it. For finding the user's ZIP/borough without a network call, `ZIP_Code_Tabulation_Areas_20261108.csv` (NYC ZIP boundaries) is turned into `nyc_zip_lookup.json` by `preprocess_nyc_zips.py`.    
- `initial trials` folder: includes all code that was created during the duration of the hackathon that was updated throughout the 25 hours. **The final results are in the main folder, initial trials is old code**    
- `hardware integration` folder: folder that contains a `.java` file that is a demonstration (**not tested during the duration of this hackathon**) for what we intend our product to be. **This is solely for showing the potential of our product + how easy and cheap it can be for SCALE.**
## How to Run It:  
1. Make sure you have the required packages installed: Python, Numpy, Pandas, SKLearn, math, etc. (`python-dotenv` is optional, for a `.env` file with your API keys)
2. Update the API keys for Purple Air and OpenWeather to your own.
3. Enter the directory for the project and type `py .\testing_datasets.py\` into the terminal to generate an updated `trained_model.json`. Ensure the stroke dataset CSV is in the directory
4. (Optional, NYC context) With `export.csv` and `ZIP_Code_Tabulation_Areas_20261108.csv` in the directory, type `py .\preprocess_nyc_data.py` to generate `nyc_baseline.json` and `py .\preprocess_nyc_zips.py` to generate `nyc_zip_lookup.json`. Then `py .\verify_nyc_data.py` checks those files against the CSVs, tests the location lookup, and prints how the model behaves on NYC data.
5. Once the `json` files are generated, type `py .\app.py` to open an interactive web page. The application is interactive and will work with your inputs! (note that AQI can be entered manually as the third option for AQI input, this was for testing purposes. Use the GPS feature.)
## Our Workflow
1. The user enters (or shares GPS for) their location, plus a short set of health questions: age, hypertension, heart disease, three-way smoking status (never / former / current), family history of stroke, household secondhand smoke exposure.
2. Pionair looks up the nearest real-time air quality sensor and current pollutant levels for that location.
3. A logistic regression model, trained offline and shipped as a small JSON file, combines those inputs into a predicted probability of a near-term cerebrovascular event.
4. That probability is mapped to one of five color-coded levels (from "Clear Sailing" to "Rest & Recharge Indoors"), each with a plain-language headline and a concrete suggestion (stay inside, mask up, it's fine to go for a walk, etc.). The policy layer also enforces EPA AQI-band guardrails and an EPA sensitive-group rule (age 65+ or heart disease) so that clean air can never trigger a high alert and sensitive users get an extra step of caution at AQI 101 and above.
5. An optional hardware companion: an air-quality sensor wired to an LED strip — mirrors the same red-to-green scale physically, so the signal doesn't require opening an app at all.   
**Why software & hardware?** Increased reliability, accessibility to all populations, ease-of-use, convenient, and portable
## UI Mockup:  
https://www.figma.com/proto/tnyfL6ssnZGn7j6YS148RZ/Traffic-Air?node-id=87-392&t=90HhvGNjr8lBRnZ7-1&scaling=scale-down&content-scaling=fixed&page-id=56%3A481&starting-point-node-id=87%3A392
## How it works

Pionair is two independent pieces that share one file:

​```
        Patient data (CSV)
                │
                ▼
        testing_datasets.py
   (fits 3 patient terms from data,
    pins smoking and family history to literature,
    adds a literature PM2.5 air term,
    saves the weights)
                     │
                     ▼
            trained_model.json
                     │
                     ▼
                  app.py
   (loads the weights, takes a live location
    + health inputs, calls PurpleAir /
    OpenWeatherMap / Nominatim, computes and
    serves a risk score in the browser)
​```
## Datasets
**The model learns from existing published population-level evidence, not from the user's personal health data**
### 1. Stroke Prediction Dataset (`healthcare-dataset-stroke-data.csv`)

- **5,110 rows × 12 columns**: `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`
- Patient records, originally published on Kaggle by fedesoriano.
- Pionair filters this to adults (age ≥ 18) before using it for training. The `smoking_status` column is used to derive two indicator variables, `smoke_current` and `smoke_former`, but their coefficients are pinned to the literature rather than fit from this cohort (see The model).

### 2. Air Quality and Health Impact Dataset (`air_quality_health_impact_data.csv`) — no longer used

- 5,811 rows from Kaggle (Rabie El Kharoua). An earlier version of the model drew AQI values from this file at random and assigned them to patients.
- Those values were not linked to any patient's outcome, so training could never learn an air effect from them, and the AQI weight had to be hand-set. The air effect now comes from a published meta-analysis (see The model), so this file is not read by any script. It is kept for reference only.

### 3. NYC Open Data "Air Quality" (`export.csv`)

- NYC Department of Health and Mental Hygiene indicators by neighborhood (UHF42), borough, and citywide, downloaded from NYC Open Data (dataset id `c3uy-2p5r`).
- Pionair uses two indicators from it: **annual mean PM2.5** (latest year in the file) and **modeled cardiovascular hospitalizations due to PM2.5, age 40+** (latest period in the file).
- These are long-term averages and modeled estimates by area. They are **not live air quality** and they **do not feed the risk model** directly. They are shown as context and used as a flagged fallback PM2.5 value when no live PurpleAir sensor is nearby in NYC.

### 4. NYC ZIP Code Tabulation Areas (`ZIP_Code_Tabulation_Areas_20261108.csv`)

- 221 ZIP areas with boundary polygons, from NYC Open Data. The polygons are **clipped to the city**, so a point inside any of them is inside NYC.
- The file lists ZIP areas that *touch* NYC, so it includes ZIPs that are mostly outside the city (for example 07305 Jersey City, 10550 Mount Vernon, 10704/10705 Yonkers, 10803 Pelham, and Nassau ZIPs 11001, 11020, 11021, 11040, 11580). `preprocess_nyc_zips.py` flags these as `partial`.
- It has no borough column: the borough comes from the ZIP prefix where unambiguous, and for the 12 other ZIPs it is inferred from the nearest ZIP with a known borough (listed in the preprocessing printout and in `verify_nyc_data.py`).
- Used only to locate the user. It does not feed the risk model.

## The model

`testing_datasets.py` trains a **logistic regression** (`scikit-learn`, `lbfgs` solver) on 3 fitted patient terms and ships 4 additional patient terms whose coefficients are pinned to published literature, for a total of 7 user features plus 1 literature-anchored air term:

| Feature | Meaning | Source |
|---|---|---|
| `age_over_40` | Years above 40, scaled in decades | Fitted from Kaggle |
| `heart` | Diagnosed heart disease (0/1) | Fitted from Kaggle |
| `hypertension` | Diagnosed hypertension (0/1) | Fitted from Kaggle |
| `smoke_current` | Currently smokes (0/1) | Pinned to literature |
| `smoke_former` | Used to smoke, quit 5+ years ago (0/1) | Pinned to literature |
| `fam_stroke` | Family history of stroke (simulated at ~14% prevalence) | Pinned to literature |
| `fam_smoke` | Household secondhand smoke (simulated at ~16% prevalence) | Pinned to literature |
| `air` | PM2.5 above 5 µg/m³ reference level, per 10 µg/m³ | Pinned to literature, not shrunk |

The pipeline then does four things beyond a plain `model.fit()`:

1. **Sets a hand-chosen intercept** (`-4.50`), which corresponds to about 1.1% for someone aged 40 or under with no risk factors at clean air. This is a design assumption for a short-term baseline. It is not fitted to or validated against incidence data.
2. **Pins four coefficients to published relative-risk estimates** instead of fitting them from the Kaggle data. Smoking is treated as a three-level categorical (never / former / current) because the two smoker levels carry very different real-world risk:
   - `smoke_current → ln(1.67)` — INTERSTROKE OR for current smokers vs never smokers (O'Donnell et al., 2016)
   - `smoke_former → ln(1.05)` — near-baseline for long-quit former smokers (Surgeon General reports)
   - `fam_stroke → ln(1.40)` — pooled family-history meta-analysis
   - `fam_smoke → ln(1.23)` — household secondhand smoke (Lee et al., 2017)
3. **Applies a global shrinkage factor (0.53)** to all seven patient-term coefficients (both fitted and pinned), so that a person with several risk factors at once doesn't get an implausibly compounded score.
4. **Adds a literature air term that is not fitted and not shrunk:** risk rises by the published relative risk of **1.011 per 10 µg/m³ of PM2.5** above a reference level of 5 µg/m³ (about the cleanest NYC neighborhood average); air at or below the reference adds nothing. Source: Shah et al., *BMJ* 2015;350:h1295 — short-term PM2.5 exposure and stroke hospital admission or mortality (95% CI 1.011 to 1.012). The studies it pooled had typical PM2.5 of roughly 6 to 34 µg/m³ and high heterogeneity (I² = 86%), so effects at much higher PM2.5 (for example wildfire smoke) are extrapolated.

The result — an intercept, 7 coefficients, and an `air` block — is written to `trained_model.json`. `app.py` loads that file once at startup and evaluates it live for each request as:

​```
probability = sigmoid(intercept + Σ (coefficient_i × feature_i) + ln(1.011)/10 × max(0, PM2.5 − 5))
​```

That probability is then bucketed into 5 levels (green to red) with the thresholds defined in `app.py`.

### Alert levels: AQI-band guardrails
The probability alone is dominated by age and health history, which could make clean air look alarming (an 80-year-old with hypertension at AQI 20) or dangerous air look fine (a healthy 30-year-old at AQI 250). So `app.py` also keeps each level inside the range allowed for today's EPA AQI band, and the model's score decides where a person lands inside that range:

| AQI band | Allowed levels |
|---|---|
| 0-50 (Good) | 1-2 |
| 51-100 (Moderate) | 1-3 |
| 101-150 (Unhealthy for Sensitive Groups) | 2-4 |
| 151-200 (Unhealthy) | 3-5 |
| 201 and above | 4-5 |

Sensitive groups get one more step. For particle pollution the EPA lists people with heart or lung disease, older adults, and children and teenagers as sensitive. The app can identify two of these, so a person aged **65 or older** or with **heart disease** has their lowest allowed level raised by one (capped at Level 5) when the AQI is 101 or higher. The EPA gives no age for "older adult", so 65 is an assumption. Hypertension is not on the EPA list, so it is not used for this rule. Adults only are supported, so children are not covered.

The care-circle popup appears only when the level is 4 or 5 **and** the AQI is 101 or higher, and never when the AQI is a neighborhood-average estimate. The probability thresholds (0.02, 0.04, 0.07, 0.12) were set empirically by inspecting outputs for hypothetical profiles, and the AQI-band ranges are a design rule anchored to the EPA category cutoffs; neither is clinically validated. `verify_nyc_data.py` (Part 4) tests these rules for every persona across AQI 0 to 500.

### Evaluation note
`testing_datasets.py` prints (and saves into `trained_model.json` under `metadata`, which `app.py` ignores) held-out test metrics for the fitted model **and** for the model that is actually shipped. On a stratified 80/20 split of the adult Kaggle cohort, the fitted 3-term model reaches a ROC-AUC of 0.796, and the full shipped model (3 fitted + 4 pinned, with the 0.53 shrinkage) reaches a ROC-AUC of 0.795 with a Brier score of 0.053 and a bootstrap calibration slope near 0.99. Patient-data terms only (age, heart, hypertension, smoke_current, smoke_former) reach a ROC-AUC of 0.802; the family-history values are simulated independently of the outcome, so they can only lower these metrics. These are prototype numbers, not clinical validation.

## APIs used

`app.py` calls three external services on every request:

- **[Nominatim](https://nominatim.org/)** (OpenStreetMap) — turns a typed address or zip code into latitude/longitude.
- **[PurpleAir API](https://api.purpleair.com/)** — finds the nearest community air-quality sensor to those coordinates and returns its live AQI reading and station name. The raw PM2.5 is converted to EPA-reference values using the humidity-corrected formula `max(0.5, 0.524 × raw − 0.0862 × humidity + 5.75)`. Sensor rows that have no PM2.5 reading or that have not reported in the last hour are rejected.
- **[OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution)** — supplies ozone (O₃) and NO₂ readings shown as context tiles on the results screen. If this call fails, or if the API key is missing, the tiles display **N/A** rather than `0.0`, so the user is never shown a missing reading as if it were clean air.

Only PM2.5 feeds the risk model's air term (the AQI derived from it also sets the alert-level ranges); the O₃ and NO₂ values are for display only. The PM2.5 tile is always tagged with its source — "live sensor", "approx from AQI" (for manual mode), or "neighborhood avg" (for the NYC fallback) — so the user can see where the value came from.

## NYC Open Data context
When the user's location is in one of the five NYC boroughs, `app.py` adds a card, "Your neighborhood, long term", showing the area's average PM2.5 and the modeled cardiovascular burden figure. The location is resolved locally from `nyc_zip_lookup.json`: a typed 5-digit ZIP inside NYC gives its borough directly, and GPS or a typed address is located inside a ZIP boundary polygon, so no extra network call is needed. A typed ZIP that is mostly outside NYC (such as Floral Park 11001 or Yonkers 10704) hides the card; use GPS or a street address instead. Only a point inside the city's bounding box but in no boundary (for example on the water) falls back to the county from Nominatim. If a `uhf42_zip_crosswalk.csv` file (columns `zip,uhf42`, using the same 101 to 504 IDs as the CSV) is present, the card becomes neighborhood-level; otherwise it is borough-level. Outside NYC the card is simply hidden.

If PurpleAir has no usable sensor in NYC (no reading, or no update in the last hour, or no API key configured), `app.py` falls back to the neighborhood's average PM2.5, clearly labeled as an estimate on the PM2.5 tile ("neighborhood avg"), and does not trigger the care-circle popup. Outside NYC, when no live air data is available, `app.py` shows a friendly "PionAir isn't working right now" box with an explanation of what is missing (typically a missing or rate-limited PurpleAir key), rather than exposing a raw Python error on the page. `verify_nyc_data.py` rechecks `nyc_baseline.json` against `export.csv` and reports how the model's alert levels behave across NYC neighborhoods.

**Limits:** the NYC figures are modeled neighborhood estimates, so they can sanity-check exposure inputs but cannot validate an individual's risk.

## Hardware component
We have code in the `hardware integration` folder that includes RGB LED light programming, allowing for a gradient that switches between red and green values. This code would be ideally integrated with a hardware component, in which the output from the ML model in `app.py` is used to change the RGB output on a device.
**not implemented in the hackathon, but the code is present to demonstrate the intended approach and scale**
## References

**Coefficients that were pinned to external literature** rather than fit **only** from the training data; **Clinical evaluation on literature was essential**:

- O'Donnell, M.J., Chin, S.L., Rangarajan, S., et al. (2016). *Global and regional effects of potentially modifiable risk factors associated with acute stroke in 32 countries (INTERSTROKE): a case-control study.* The Lancet, 388(10046), 761–775. — basis for the `smoke_current` relative risk (OR 1.67 for current smokers vs never smokers).
- U.S. Department of Health and Human Services. (2014). *The Health Consequences of Smoking — 50 Years of Progress: A Report of the Surgeon General.* — basis for the `smoke_former` relative risk (near baseline after 5+ years of quitting).
- Yu, S., Su, Z., Miao, J., et al. (2019). *Different Types of Family History of Stroke and Stroke Risk: Results Based on 655,552 Individuals.* — basis for the `fam_stroke` relative risk. (PMID: 30472175)
- Lee et al. (2017) — cited in code as the basis for the household/secondhand-smoke relative risk used for `fam_smoke`.
- Alonso et al. and the Framingham AF cohort papers on atrial fibrillation's 5-fold stroke risk
- Peters et al., The relationship between smoking and stroke: a meta-analysis (Medicine, 2019)
- Steyerberg, E.W., Eijkemans, M.J.C., Van Houwelingen, J.C., Lee, K.L., & Habbema, J.D.F. (2000). "Prognostic models based on literature and individual patient data in logistic regression analysis." Statistics in Medicine, 19(2), 141–160.
- Newcombe, P.J., Reck, B.H., Sun, J., Platek, G.T., Verzilli, C., Kader, A.K., Kim, S.T., Hsu, F.C., Zhang, Z., Zheng, S.L., Mooser, V.E., Condreay, L.D., Spraggs, C.F., Whittaker, J.C., Rittmaster, R.S., & Xu, J. (2012). "A comparison of Bayesian and frequentist approaches to incorporating external information for the prediction of prostate cancer risk." Genetic Epidemiology, 36(1), 71–83.
- Chatterjee, N., Chen, Y.-H., Maas, P., & Carroll, R.J. (2016). "Constrained Maximum Likelihood Estimation for Model Calibration Using Summary-Level Information From External Big Data Sources." Journal of the American Statistical Association, 111(513), 107–117.
- Shah, A.S.V., Lee, K.K., McAllister, D.A., Hunter, A., Nair, H., Whiteley, W., Langrish, J.P., Newby, D.E., & Mills, N.L. (2015). "Short term exposure to air pollution and stroke: systematic review and meta-analysis." *BMJ*, 350, h1295. doi:10.1136/bmj.h1295 — basis for the PM2.5 air term (RR 1.011 per 10 µg/m³). (Corrected in *BMJ* 2016;354:i4851, a licensing correction.)
- U.S. EPA AirNow, *Air Quality Guide for Particle Pollution* (https://www.airnow.gov/sites/default/files/2021-03/air-quality-guide_pm_2015_0.pdf) — AQI categories and the sensitive groups used for the alert-level ranges and the sensitive-group rule.

Dataset sources:
- fedesoriano. *Stroke Prediction Dataset.* Kaggle. ; https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?
- (No longer used) Rabie El Kharoua. *Air Quality and Health Impact Dataset.* Kaggle. ; https://www.kaggle.com/datasets/tfisthis/global-air-quality-and-persirator-health-outcomes
