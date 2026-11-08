"""
verify_nyc_data.py

PURPOSE:
- Verify the NYC data pipeline and check how the PionAir risk model behaves against NYC data.
- Part 1 (data integrity): recompute values straight from export.csv with independent code and compare
  them to nyc_baseline.json, plus range, coverage, and staleness checks.
- Part 2 (model checks): run the real app.py scoring code on NYC neighborhood PM2.5 values and report
  (a) whether risk rises with air quality, (b) alert levels across neighborhoods, (c) an order-of-magnitude
  effect-size comparison with NYC's reported PM2.5 share of cardiovascular hospitalizations,
  (d) how neighborhood PM2.5 relates to NYC's modeled cardiovascular burden (informational only).

IMPORTANT LIMITS (also written into the report):
- NYC figures are modeled, neighborhood-level estimates. They can check exposure inputs and plausibility of
  the air-related part of the model. They cannot validate an individual's stroke risk.
- NYC's health-burden figures are modeled rates that, as I understand the method, combine a PM2.5-attributable share
  with each neighborhood's baseline hospitalization rate (confirm in the DOHMH technical notes). In this export
  they are NOT well correlated with PM2.5 alone, so Check 2.4 is informational and is not a validation of the model.

USAGE:
    python verify_nyc_data.py                    # uses export.csv and nyc_baseline.json
    python verify_nyc_data.py path/to/export.csv

Exit code is 1 if any check FAILS (warnings do not fail the run). A JSON report is saved to
nyc_verification_report.json.
"""

import datetime
import json
import math
import sys

import pandas as pd

import app  # the real scoring code: pm25_to_aqi, compute_detailed_risk, INTERCEPT, COEFS

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "export.csv"
BASELINE_PATH = "nyc_baseline.json"
REPORT_PATH = "nyc_verification_report.json"

PM25_NAME, PM25_MEASURE = "Fine particles (PM 2.5)", "Annual mean"
CV_NAME = "Cardiovascular hospitalizations due to PM2.5 (age 40+)"

# NYC DOHMH data story: about 1 in 140 cardiovascular hospitalizations is caused by PM2.5.
# Different outcome (hospitalizations, not stroke), period, and method than this model, so this is
# an order-of-magnitude reference only. Source: a816-dohbesp.nyc.gov (traffic and air pollution data story).
NYC_REFERENCE_ATTRIBUTABLE_FRACTION = 1 / 140

PERSONAS = {
    "healthy 30yo": dict(age=30, hypertension=False, heart=False, smoke=False, fam_smoke=False, fam_stroke=False),
    "healthy 50yo": dict(age=50, hypertension=False, heart=False, smoke=False, fam_smoke=False, fam_stroke=False),
    "65yo hypertension": dict(age=65, hypertension=True, heart=False, smoke=False, fam_smoke=False, fam_stroke=False),
    "80yo hypertension": dict(age=80, hypertension=True, heart=False, smoke=False, fam_smoke=False, fam_stroke=False),
    "55yo smoker": dict(age=55, hypertension=False, heart=False, smoke=True, fam_smoke=False, fam_stroke=False),
}

results = []  # (status, check name, detail)


def record(status, name, detail=""):
    results.append({"status": status, "check": name, "detail": detail})
    print(f"[{status:4}] {name}" + (f" - {detail}" if detail else ""))


# ----------------------------------------------------------------------------------------------
# Part 1: data integrity (independent recomputation from the raw CSV)
# ----------------------------------------------------------------------------------------------
def latest_rows(df, geo):
    sub = df[df["Geo Type Name"] == geo].copy()
    sub["_d"] = pd.to_datetime(sub["Start_Date"], format="%m/%d/%Y", errors="coerce")
    return sub[sub["_d"] == sub["_d"].max()]


def part1(raw, base):
    pm_raw = raw[(raw["Name"] == PM25_NAME) & (raw["Measure"] == PM25_MEASURE)]
    cv_raw = raw[raw["Name"] == CV_NAME]
    pm_b, cv_b = base["pm25_annual_mean"], base["cv_hospitalizations_due_to_pm25_age40plus"]

    # coverage
    record("PASS" if len(pm_b["uhf42"]) == 42 else "FAIL", "1.1 PM2.5 covers all 42 UHF neighborhoods", f"found {len(pm_b['uhf42'])}")
    record("PASS" if len(pm_b["borough"]) == 5 else "FAIL", "1.2 PM2.5 covers all 5 boroughs", f"found {len(pm_b['borough'])}")
    missing_cv = sorted(set(pm_b["uhf42"]) - set(cv_b["uhf42"]))
    record("PASS" if not missing_cv else "WARN", "1.3 every neighborhood with PM2.5 also has a CV burden value",
           f"missing: {missing_cv}" if missing_cv else "")

    # exact match against the raw file, value by value
    mismatches = 0
    checked = 0
    for table, raw_sub in ((pm_b, pm_raw), (cv_b, cv_raw)):
        u = latest_rows(raw_sub, "UHF42")
        for _, r in u.iterrows():
            checked += 1
            stored = table["uhf42"].get(str(int(r["Geo Join ID"])), {}).get("value")
            if stored is None or abs(stored - round(float(r["Data Value"]), 2)) > 1e-9:
                mismatches += 1
        b = latest_rows(raw_sub, "Borough")
        for _, r in b.iterrows():
            checked += 1
            stored = table["borough"].get(r["Geo Place Name"])
            if stored is None or abs(stored - round(float(r["Data Value"]), 2)) > 1e-9:
                mismatches += 1
    record("PASS" if mismatches == 0 else "FAIL", "1.4 stored values match export.csv (independent recompute)",
           f"{checked} values checked, {mismatches} mismatches")

    # plausibility ranges
    pm_vals = [v["value"] for v in pm_b["uhf42"].values()] + list(pm_b["borough"].values()) + [pm_b["citywide"]]
    ok = all(2 <= v <= 40 for v in pm_vals)
    record("PASS" if ok else "FAIL", "1.5 PM2.5 annual means are in a plausible range (2 to 40 ug/m3)",
           f"min {min(pm_vals):.2f}, max {max(pm_vals):.2f}")
    cv_vals = [v["value"] for v in cv_b["uhf42"].values()]
    record("PASS" if all(v >= 0 for v in cv_vals) else "FAIL", "1.6 CV burden values are non-negative",
           f"min {min(cv_vals):.1f}, max {max(cv_vals):.1f}")

    # borough consistency: citywide should sit within the range of the boroughs
    lo, hi = min(pm_b["borough"].values()), max(pm_b["borough"].values())
    record("PASS" if lo <= pm_b["citywide"] <= hi else "WARN", "1.7 citywide PM2.5 lies within the borough range",
           f"citywide {pm_b['citywide']}, boroughs {lo} to {hi}")

    # staleness
    this_year = datetime.date.today().year
    pm_year = int(str(pm_b["period"])[:4])
    record("PASS" if this_year - pm_year <= 2 else "WARN", "1.8 PM2.5 data is recent (within 2 years)",
           f"period {pm_b['period']}")
    cv_year = int(str(cv_b["period"])[-4:])
    record("PASS" if this_year - cv_year <= 3 else "WARN", "1.9 CV burden data is recent (within 3 years)",
           f"period {cv_b['period']} (modeled estimates are published with a delay; label as such in the app)")


# ----------------------------------------------------------------------------------------------
# Part 2: model checks against NYC data (runs the real app.py functions)
# ----------------------------------------------------------------------------------------------
def risk(persona, aqi):
    p = app.compute_detailed_risk(**persona, aqi=aqi)
    return p[0], p[1]  # probability, level


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(ra.corr(rb))


def part2(base):
    pm_b, cv_b = base["pm25_annual_mean"], base["cv_hospitalizations_due_to_pm25_age40plus"]

    # 2.1 monotonicity: risk must not fall as air quality worsens
    ok = True
    for name, persona in PERSONAS.items():
        probs = [risk(persona, aqi)[0] for aqi in range(0, 501, 10)]
        if any(b < a - 1e-12 for a, b in zip(probs, probs[1:])):
            ok = False
    record("PASS" if ok else "FAIL", "2.1 risk never decreases as AQI increases (all personas, AQI 0 to 500)")

    # 2.2 neighborhood alert levels using each neighborhood's annual-average PM2.5 (typical-day scenario)
    summary = {}
    for name, persona in PERSONAS.items():
        levels = [risk(persona, app.pm25_to_aqi(v["value"]))[1] for v in pm_b["uhf42"].values()]
        counts = {lv: levels.count(lv) for lv in sorted(set(levels))}
        summary[name] = counts
    print("\n      Alert levels on a typical-day (annual-average PM2.5) across the 42 neighborhoods:")
    for name, counts in summary.items():
        print(f"        {name:18} {counts}")
    # How much of the level is explained by the air? Compare the same persona at the cleanest vs dirtiest neighborhood.
    vals = {k: v["value"] for k, v in pm_b["uhf42"].items()}
    clean_k, dirty_k = min(vals, key=vals.get), max(vals, key=vals.get)
    swing = []
    for name, persona in PERSONAS.items():
        p_clean, l_clean = risk(persona, app.pm25_to_aqi(vals[clean_k]))
        p_dirty, l_dirty = risk(persona, app.pm25_to_aqi(vals[dirty_k]))
        swing.append((name, l_clean, l_dirty))
    moved = sum(1 for _, a, b in swing if a != b)
    record("INFO", "2.2 cleanest vs dirtiest NYC neighborhood changes the alert level for how many personas?",
           f"{moved} of {len(swing)} ({pm_b['uhf42'][clean_k]['name']} {vals[clean_k]} vs {pm_b['uhf42'][dirty_k]['name']} {vals[dirty_k]} ug/m3). "
           "If this is low, the alert is driven mostly by the user's profile, not the air.")

    # 2.3 effect-size comparison with NYC's reported attributable share (order of magnitude only)
    aqi_city = app.pm25_to_aqi(pm_b["citywide"])
    coef = app.COEFS.get("aqi_scaled", 0.035)
    odds_ratio = math.exp(coef * aqi_city / 10.0)
    implied_af = 1 - 1 / odds_ratio
    ratio = implied_af / NYC_REFERENCE_ATTRIBUTABLE_FRACTION
    status = "PASS" if 0.2 <= ratio <= 5 else "WARN"
    record(status, "2.3 model's implied air-attributable share at NYC's average PM2.5 vs NYC's reported share",
           f"model implies {implied_af*100:.1f}% (about 1 in {1/implied_af:.0f}) vs NYC about {NYC_REFERENCE_ATTRIBUTABLE_FRACTION*100:.2f}% (1 in 140) "
           f"= {ratio:.1f}x. Different outcome and method, so only an order-of-magnitude check. "
           "A large ratio suggests the AQI coefficient overstates the short-term effect.")

    # 2.4 informational: how does neighborhood PM2.5 relate to modeled CV burden?
    keys = [k for k in pm_b["uhf42"] if k in cv_b["uhf42"]]
    rho = spearman([pm_b["uhf42"][k]["value"] for k in keys], [cv_b["uhf42"][k]["value"] for k in keys])
    top = sorted(keys, key=lambda k: cv_b["uhf42"][k]["value"], reverse=True)[:3]
    top_txt = ", ".join(f"{pm_b['uhf42'][k]['name']} ({pm_b['uhf42'][k]['value']} ug/m3)" for k in top)
    record("INFO", "2.4 neighborhood PM2.5 vs modeled CV burden: Spearman rank correlation",
           f"rho = {rho:.2f} across {len(keys)} neighborhoods. Highest burden: {top_txt}. A weak correlation is expected if the burden "
           "mostly reflects local baseline disease rates. Do not use this as a test of the model; the integrity checks in Part 1 cover the join.")
    return summary, {"implied_attributable_fraction": implied_af, "nyc_reference": NYC_REFERENCE_ATTRIBUTABLE_FRACTION, "spearman_pm25_vs_cv": rho}


def main():
    raw = pd.read_csv(CSV_PATH)
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)
    print(f"Verifying {BASELINE_PATH} against {CSV_PATH}\n")
    print("PART 1: data integrity")
    part1(raw, base)
    print("\nPART 2: model checks against NYC data")
    summary, extra = part2(base)

    fails = [r for r in results if r["status"] == "FAIL"]
    warns = [r for r in results if r["status"] == "WARN"]
    print(f"\nSummary: {sum(r['status']=='PASS' for r in results)} passed, {len(warns)} warnings, {len(fails)} failed")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "csv": CSV_PATH, "results": results, "typical_day_levels_by_persona": summary, "effect_size": extra,
            "limits": "NYC figures are modeled neighborhood estimates; they cannot validate individual risk. Check 2.4 is informational only.",
        }, f, indent=2)
    print(f"Report saved to {REPORT_PATH}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
