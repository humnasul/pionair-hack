"""
preprocess_nyc_data.py

PURPOSE:
- Read the NYC Open Data "Air Quality" export (export.csv) and write a small
  lookup file, nyc_baseline.json, that app.py can load at startup.
- The output is CONTEXT ONLY (neighborhood long-term PM2.5 and a health-burden
  indicator). It is not fed into the risk model.

USAGE:
    python preprocess_nyc_data.py                       # uses export.csv
    python preprocess_nyc_data.py path/to/export.csv

Rows used (names/measures verified against the exported file):
    PM2.5:   Name == "Fine particles (PM 2.5)", Measure == "Annual mean"
    Burden:  Name == "Cardiovascular hospitalizations due to PM2.5 (age 40+)"
             (modeled estimate per 100,000 adults, not an observed count)
Geographies kept: UHF42 neighborhoods, boroughs, and citywide.
"""

import json
import sys

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "export.csv"
OUT = "nyc_baseline.json"

PM25_NAME = "Fine particles (PM 2.5)"
PM25_MEASURE = "Annual mean"
CV_NAME = "Cardiovascular hospitalizations due to PM2.5 (age 40+)"
GEO_KEEP = ["UHF42", "Borough", "Citywide"]


def latest_period(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows from the most recent period (by Start_Date) in df."""
    start = pd.to_datetime(df["Start_Date"], format="%m/%d/%Y", errors="coerce")
    newest = start.max()
    return df[start == newest]


def build_table(df: pd.DataFrame) -> dict:
    """Turn rows for ONE indicator into {citywide, borough{}, uhf42{}} lookups."""
    out = {"period": None, "citywide": None, "borough": {}, "uhf42": {}}
    for geo in GEO_KEEP:
        sub = latest_period(df[df["Geo Type Name"] == geo])
        if sub.empty:
            continue
        out["period"] = str(sub["Time Period"].iloc[0])
        for _, r in sub.iterrows():
            value = round(float(r["Data Value"]), 2)
            if geo == "Citywide":
                out["citywide"] = value
            elif geo == "Borough":
                out["borough"][r["Geo Place Name"]] = value
            else:  # UHF42
                out["uhf42"][str(int(r["Geo Join ID"]))] = {
                    "name": r["Geo Place Name"],
                    "value": value,
                }
    return out


def main():
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC}")

    pm = df[(df["Name"] == PM25_NAME) & (df["Measure"] == PM25_MEASURE)]
    pm = pm.dropna(subset=["Data Value"])
    cv = df[df["Name"] == CV_NAME].dropna(subset=["Data Value"])

    pm_table = build_table(pm)
    cv_table = build_table(cv)

    # Sanity checks so a changed export format fails loudly, not silently
    assert pm_table["citywide"] is not None, "No citywide PM2.5 row found"
    assert len(pm_table["borough"]) == 5, f"Expected 5 boroughs, got {len(pm_table['borough'])}"
    assert len(pm_table["uhf42"]) == 42, f"Expected 42 UHF neighborhoods, got {len(pm_table['uhf42'])}"
    assert all(2 < v < 40 for v in pm_table["borough"].values()), "PM2.5 values look implausible"

    payload = {
        "source": "NYC Open Data - Air Quality (DOHMH), data.cityofnewyork.us/d/c3uy-2p5r",
        "note": "Long-term neighborhood averages and modeled burden estimates. Not live air quality.",
        "pm25_annual_mean": {"units": "ug/m3", **pm_table},
        "cv_hospitalizations_due_to_pm25_age40plus": {
            "units": "estimated annual rate per 100,000 adults",
            **cv_table,
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"PM2.5 period: {pm_table['period']}  |  citywide: {pm_table['citywide']} ug/m3")
    print(f"  boroughs: {pm_table['borough']}")
    print(f"  UHF42 neighborhoods: {len(pm_table['uhf42'])}")
    print(f"CV burden period: {cv_table['period']}  |  citywide: {cv_table['citywide']} per 100k")
    print(f"[SUCCESS] Wrote {OUT}")


if __name__ == "__main__":
    main()
