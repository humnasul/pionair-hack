# Pionair: Personalized Air Quality Communication Tool
## What's Pionair?
Pionair integrates live air pollution data into daily life through an app, care circle, and portable personalized device, making air quality information more accessible to vulnerable populations.
## File Directory:  
- The main branch (outside any folders) includes...   
2 dataset CSVs used by `testing_datasets.py`. These datasets are provided from Kaggle.   
`testing_datasets.py` creates weights utilizing the CSVs that are outputted as a file, `trained_model.json`. These weights are used as a basis of logistic regression in `app.py`.    
`app.py` is the interactive web app.    
- `initial trials` folder: includes all code that was created during the duration of the hackathon that was updated throughout the 25 hours. **The final results are in the main folder, initial trials is old code**    
- `hardware integration` folder: folder that contains a `.java` file that is a demonstration (**not tested during the duration of this hackathon**) for what we intend our product to be. **This is solely for showing the potential of our product + how easy and cheap it can be for SCALE.**
## How to Run It:  
1. Make sure you have the required packages installed: Python, Numpy, SKLearn, math, etc.
2. Update the API keys for Purple Air and OpenWeather to your own.
3. Enter the directory for the project and type `py .\testing_datasets.py\` into the terminal to generate an updated `trained_model.json`. Ensure the 2 dataset CSVs are in the directory
4. Once the `json` is generated, type `py .\app.py` to open an interactive web page. The application is interactive and will work with your inputs! (note that AQI can be entered manually as the third option for AQI input, this was for testing purposes. Use the GPS feature.)
## Our Workflow
1. The user enters (or shares GPS for) their location, plus a short set of health questions: age, hypertension, heart disease, smoking status, family history of stroke, family history of smoking.
2. Pionair looks up the nearest real-time air quality sensor and current pollutant levels for that location.
3. A logistic regression model, trained offline and shipped as a small JSON file, combines those inputs into a predicted probability of a near-term cerebrovascular event.
4. That probability is mapped to one of five color-coded levels (from "Clear Sailing" to "Rest & Recharge Indoors"), each with a plain-language headline and a concrete suggestion (stay inside, mask up, it's fine to go for a walk, etc.).
5. An optional hardware companion: an air-quality sensor wired to an LED strip — mirrors the same red-to-green scale physically, so the signal doesn't require opening an app at all.   
**Why software & hardware?** Increased reliability, accessibility to all populations, ease-of-use, convenient, and portable
## UI Mockup:  
https://www.figma.com/proto/tnyfL6ssnZGn7j6YS148RZ/Traffic-Air?node-id=87-392&t=90HhvGNjr8lBRnZ7-1&scaling=scale-down&content-scaling=fixed&page-id=56%3A481&starting-point-node-id=87%3A392
## How it works

Pionair is two independent pieces that share one file:

```
 Patient data (CSV)         Air quality data (CSV)
        │                            │
        └───────────┬────────────────┘
                     ▼
            testing_datasets.py
        (trains a logistic regression,
         applies literature overrides,
         saves the learned weights)
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
```
## Datasets
**The model learns from existing published population-level evidence, not from the user’s personal health data**
### 1. Stroke Prediction Dataset (`healthcare-dataset-stroke-data.csv`)

- **5,110 rows × 12 columns**: `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`
- Patient records, originally published on Kaggle by fedesoriano.
- Pionair filters this to adults (age ≥ 18) before using it for training.

### 2. Air Quality and Health Impact Dataset (`air_quality_health_impact_data.csv`)

- **5,811 rows × 15 columns**: `RecordID`, `AQI`, `PM10`, `PM2_5`, `NO2`, `SO2`, `O3`, `Temperature`, `Humidity`, `WindSpeed`, `RespiratoryCases`, `CardiovascularCases`, `HospitalAdmissions`, `HealthImpactScore`, `HealthImpactClass`
- Originally published on Kaggle by Rabie El Kharoua.
- Used here only as a source of realistic AQI values (`AQI` column) — sampled with replacement to size-match the stroke cohort. Everything else in the file (respiratory/cardiovascular case counts, the health impact score/class) is not used, since it's synthetic and not causally tied to the real stroke cohort.

## The model

`testing_datasets.py` trains a **logistic regression** (`scikit-learn`, `lbfgs` solver) on 7 features:

| Feature | Meaning |
|---|---|
| `age_over_40` | Years above 40, scaled in decades |
| `heart` | Diagnosed heart disease (0/1) |
| `hypertension` | Diagnosed hypertension (0/1) |
| `smoke` | Currently or formerly smokes (0/1) |
| `fam_stroke` | Family history of stroke (simulated at ~14% prevalence) |
| `fam_smoke` | Family history / household smoking (simulated at ~16% prevalence) |
| `aqi_scaled` | Local AQI ÷ 10 |

The pipeline then does three things beyond a plain `model.fit()`:

1. **Sets a calibrated intercept** (`-4.50`) so the baseline predicted risk lines up with a realistic ~1% short-term population base rate, rather than whatever the raw training data happens to imply.
2. **Overrides three coefficients with published relative-risk estimates** instead of trusting the fitted values for features that were only loosely simulated (family history, secondhand smoke, and AQI itself weren't independently validated in the source data):
   - `fam_stroke → ln(1.40)`
   - `fam_smoke → ln(1.23)`
   - `aqi_scaled → ln(1.10)` per 10-point AQI increase
3. **Applies a global shrinkage factor (0.53)** to every coefficient, so that a person with several risk factors at once doesn't get an implausibly compounded score — a known failure mode of naively stacking independent odds ratios.

The result — an intercept and 7 coefficients — is written to `trained_model.json`. `app.py` loads that file once at startup and evaluates it live for each request as:

```
probability = sigmoid(intercept + Σ (coefficient_i × feature_i))
```

That probability is then bucketed into 5 levels (green to red) with the thresholds defined in `app.py`.

## APIs used

`app.py` calls three external services on every request:

- **[Nominatim](https://nominatim.org/)** (OpenStreetMap) — turns a typed address or zip code into latitude/longitude.
- **[PurpleAir API](https://api.purpleair.com/)** — finds the nearest community air-quality sensor to those coordinates and returns its live AQI reading and station name.
- **[OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution)** — supplies a pollutant breakdown (PM2.5, O₃, NO₂) shown on the results screen. If this call fails, `app.py` falls back to an AQI-derived PM2.5 estimate so the app still returns a result.

Only the AQI value itself feeds the risk model; the individual pollutant breakdown is currently for display only.

## Hardware component
We have code in the `hardware integration` folder that includes RGB LED light programming, allowing for a gradient that switches between red and green values. This code would be ideally integrated with a hardware component, in which the output from the ML model in `app.py` is used to change the RGB output on a device.
**not implemented in the hackathon, but the code is present to demonstrate the intended approach and scale**
## References

**Coefficients that were pinned to external literature** rather than fit **only** from the training data; **Clinical evaluation on literature was essential**:

- Yu, S., Su, Z., Miao, J., et al. (2019). *Different Types of Family History of Stroke and Stroke Risk: Results Based on 655,552 Individuals.* — basis for the `fam_stroke` relative risk. (PMID: 30472175)
- Lee et al. (2017) — cited in code as the basis for the household/secondhand-smoke relative risk used for `fam_smoke`.
- Alonso et al. and the Framingham AF cohort papers on atrial fibrillation's 5-fold stroke risk
- Peters et al., The relationship between smoking and stroke: a meta-analysis (Medicine, 2019)
- Steyerberg, E.W., Eijkemans, M.J.C., Van Houwelingen, J.C., Lee, K.L., & Habbema, J.D.F. (2000). "Prognostic models based on literature and individual patient data in logistic regression analysis." Statistics in Medicine, 19(2), 141–160.
- Newcombe, P.J., Reck, B.H., Sun, J., Platek, G.T., Verzilli, C., Kader, A.K., Kim, S.T., Hsu, F.C., Zhang, Z., Zheng, S.L., Mooser, V.E., Condreay, L.D., Spraggs, C.F., Whittaker, J.C., Rittmaster, R.S., & Xu, J. (2012). "A comparison of Bayesian and frequentist approaches to incorporating external information for the prediction of prostate cancer risk." Genetic Epidemiology, 36(1), 71–83.
- Chatterjee, N., Chen, Y.-H., Maas, P., & Carroll, R.J. (2016). "Constrained Maximum Likelihood Estimation for Model Calibration Using Summary-Level Information From External Big Data Sources." Journal of the American Statistical Association, 111(513), 107–117.
- AQI–stroke incidence relative risk (per 10-unit AQI increase) used for `aqi_scaled`.

Dataset sources:
- fedesoriano. *Stroke Prediction Dataset.* Kaggle. ; https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?
- Rabie El Kharoua. *Air Quality and Health Impact Dataset.* Kaggle. ; https://www.kaggle.com/datasets/tfisthis/global-air-quality-and-respiratory-health-outcomes
