"""
nyc_context.py

PURPOSE:
- Given a latitude/longitude, return NYC neighborhood context (long-term PM2.5
  average and a modeled cardiovascular burden figure) from nyc_baseline.json.
- CONTEXT ONLY: nothing here feeds the risk model.
- Never raises: returns None outside NYC or on any failure, so app.py keeps working.

LOOKUP ORDER:
1. Neighborhood (UHF42) if an optional ZIP crosswalk file is present
   (uhf42_zip_crosswalk.csv with columns: zip,uhf42) and the postcode is known.
2. Borough, from the county returned by Nominatim reverse geocoding.
3. None (card is hidden) if the point is not in one of the five NYC counties.
"""

import csv
import json
import os
import urllib.parse
import urllib.request

BASELINE_PATH = "nyc_baseline.json"
CROSSWALK_PATH = "uhf42_zip_crosswalk.csv"  # optional upgrade, see app_changes.md

COUNTY_TO_BOROUGH = {
    "bronx county": "Bronx",
    "kings county": "Brooklyn",
    "new york county": "Manhattan",
    "queens county": "Queens",
    "richmond county": "Staten Island",
}

# Loaded once at import, like trained_model.json in app.py
try:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        _BASE = json.load(f)
except (OSError, ValueError):
    _BASE = None

_ZIP_TO_UHF = {}
if os.path.exists(CROSSWALK_PATH):
    try:
        with open(CROSSWALK_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _ZIP_TO_UHF[row["zip"].strip()[:5]] = str(int(row["uhf42"]))
    except (OSError, KeyError, ValueError):
        _ZIP_TO_UHF = {}


_ADDRESS_CACHE = {}  # rounded (lat, lon) -> address dict; Nominatim asks for ~1 request/second


def _reverse_geocode(lat: float, lon: float) -> dict:
    """Nominatim reverse lookup; returns the 'address' dict (cached by ~100 m cell)."""
    key = (round(lat, 3), round(lon, 3))
    if key in _ADDRESS_CACHE:
        return _ADDRESS_CACHE[key]
    params = urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "PlayfulHealthAir/9.0 (community-check)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        address = json.loads(resp.read().decode("utf-8")).get("address", {})
    _ADDRESS_CACHE[key] = address
    return address


def nyc_context(lat: float, lon: float):
    """Return a context dict for the card, or None if not in NYC / data missing."""
    if not _BASE:
        return None
    try:
        addr = _reverse_geocode(lat, lon)
        borough = COUNTY_TO_BOROUGH.get(str(addr.get("county", "")).strip().lower())
        if not borough:
            return None  # outside the five boroughs: hide the card

        pm = _BASE["pm25_annual_mean"]
        cv = _BASE["cv_hospitalizations_due_to_pm25_age40plus"]

        uhf_id = _ZIP_TO_UHF.get(str(addr.get("postcode", ""))[:5])
        if uhf_id and uhf_id in pm["uhf42"]:
            return {
                "level": "neighborhood",
                "name": pm["uhf42"][uhf_id]["name"],
                "borough": borough,
                "pm25_annual_mean": pm["uhf42"][uhf_id]["value"],
                "pm25_period": pm["period"],
                "cv_burden_per_100k": cv["uhf42"].get(uhf_id, {}).get("value"),
                "cv_period": cv["period"],
                "citywide_pm25": pm["citywide"],
            }

        return {
            "level": "borough",
            "name": borough,
            "borough": borough,
            "pm25_annual_mean": pm["borough"][borough],
            "pm25_period": pm["period"],
            "cv_burden_per_100k": cv["borough"].get(borough),
            "cv_period": cv["period"],
            "citywide_pm25": pm["citywide"],
        }
    except Exception:
        return None
