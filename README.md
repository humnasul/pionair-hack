## Pionair: Personalized Air Quality Communication Tool
# What It Does?  
1. The user enters (or shares GPS for) their location, plus a short set of health questions: age, hypertension, heart disease, smoking status, family history of stroke, family history of smoking.
2. Pionair looks up the nearest real-time air quality sensor and current pollutant levels for that location.
3. A logistic regression model — trained offline and shipped as a small JSON file — combines those inputs into a predicted probability of a near-term cerebrovascular event.
4. That probability is mapped to one of five color-coded levels (from "Clear Sailing" to "Rest & Recharge Indoors"), each with a plain-language headline and a concrete suggestion (stay inside, mask up, it's fine to go for a walk, etc.).
5. An optional hardware companion — an air-quality sensor wired to an LED strip — mirrors the same red-to-green scale physically, so the signal doesn't require opening an app at all.
