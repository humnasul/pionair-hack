"""
nyc_context.py

PURPOSE:
- Given a latitude/longitude (and optionally what the user typed), return NYC neighborhood context
  (long-term PM2.5 average and a modeled cardiovascular burden figure) from nyc_baseline.json.
- CONTEXT ONLY: nothing here feeds the risk model.
- Never raises: returns None outside NYC or on any failure, so app.py keeps working.

HOW THE LOCATION IS RESOLVED (in this order):
1. A typed 5-digit ZIP is classified with nyc_zip_lookup.json (built by preprocess_nyc_zips.py):
     - a ZIP inside NYC -> borough, no network call
     - a ZIP mostly outside NYC (Nassau, Westchester, New Jersey) or not an NYC ZIP -> card hidden, no network call
2. GPS or a typed address: the point is located inside a ZIP boundary polygon locally (no network call).
   The boundaries are clipped to NYC, so a point inside any of them is inside the city.
   A point outside the city's bounding box is hidden immediately, also without a network call.
3. If the point is inside the bounding box but in no polygon (water, gaps, or nyc_zip_lookup.json missing),
   the county from Nominatim reverse geocoding decides (one network call, cached by ~100 m cell).
4. Neighborhood (UHF42) if an optional crosswalk file is present (uhf42_zip_crosswalk.csv, columns: zip,uhf42)
   and the ZIP is known. Otherwise the card is borough-level.
Whenever the card is hidden because of an error or a non-NYC location, the reason is printed in the terminal.
"""

import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BASELINE_PATH = "nyc_baseline.json"
ZIP_LOOKUP_PATH = "nyc_zip_lookup.json"
CROSSWALK_PATH = "uhf42_zip_crosswalk.csv"  # optional upgrade to neighborhood-level

COUNTY_TO_BOROUGH = {
    "bronx county": "Bronx",
    "kings county": "Brooklyn",
    "new york county": "Manhattan",
    "queens county": "Queens",
    "richmond county": "Staten Island",
}

# Used only for ZIPs that are missing from the boundary file (PO box and single-building ZIPs such as 10118)
PREFIX_TO_BOROUGH = {
    "100": "Manhattan", "101": "Manhattan", "102": "Manhattan",
    "103": "Staten Island",
    "104": "Bronx",
    "112": "Brooklyn",
    "111": "Queens", "113": "Queens", "114": "Queens", "116": "Queens",
}

# Loaded once at import, like trained_model.json in app.py
try:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        _BASE = json.load(f)
except (OSError, ValueError):
    _BASE = None

try:
    with open(ZIP_LOOKUP_PATH, "r", encoding="utf-8") as f:
        _ZIPS = json.load(f)["zips"]
except (OSError, ValueError, KeyError):
    _ZIPS = {}

# Bounding box of everything inside NYC (a little padding): a point outside it is not in the city
if _ZIPS:
    _NYC_BBOX = (
        min(e["bbox"][0] for e in _ZIPS.values()) - 0.01, min(e["bbox"][1] for e in _ZIPS.values()) - 0.01,
        max(e["bbox"][2] for e in _ZIPS.values()) + 0.01, max(e["bbox"][3] for e in _ZIPS.values()) + 0.01,
    )
else:
    _NYC_BBOX = None

_ZIP_TO_UHF = {}
if os.path.exists(CROSSWALK_PATH):
    try:
        with open(CROSSWALK_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _ZIP_TO_UHF[row["zip"].strip()[:5]] = str(int(row["uhf42"]))
    except (OSError, KeyError, ValueError):
        _ZIP_TO_UHF = {}

_ADDRESS_CACHE = {}  # rounded (lat, lon) -> address dict; Nominatim asks for ~1 request/second


def parse_typed_zip(text):
    """Return the 5-digit ZIP if the user typed one (optionally ZIP+4), else None."""
    m = re.match(r"^\s*(\d{5})(?:-\d{4})?\s*$", str(text or ""))
    return m.group(1) if m else None


def classify_zip(z):
    """Return ("nyc", borough), ("partial", None), ("outside", None), or ("unknown", None) for a typed 5-digit ZIP.

    partial: the ZIP is mostly outside NYC (only a sliver is inside), so a typed one is not treated as NYC.
    unknown: no boundary file is loaded, so the ZIP cannot be ruled in or out locally.
    """
    entry = _ZIPS.get(z)
    if entry:
        if entry.get("partial"):
            return "partial", None
        return "nyc", entry["borough"]
    borough = PREFIX_TO_BOROUGH.get(z[:3])  # ZIPs missing from the boundary file (PO box, single-building ZIPs)
    if borough:
        return "nyc", borough
    if not _ZIPS:
        return "unknown", None
    return "outside", None


def zip_for_point(lat, lon):
    """Find the ZIP boundary polygon containing a point, using only local data. Returns a ZIP or None."""
    matches = []
    for z, entry in _ZIPS.items():
        x0, y0, x1, y1 = entry["bbox"]
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        inside = False
        for ring in entry["rings"]:  # even-odd rule: rings include holes
            j = len(ring) - 1
            for i in range(len(ring)):
                xi, yi = ring[i]
                xj, yj = ring[j]
                if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                    inside = not inside
                j = i
        if inside:
            matches.append(z)
    if not matches:
        return None
    whole = [z for z in matches if not _ZIPS[z].get("partial")]
    return whole[0] if whole else matches[0]  # polygons should not overlap; if they do, prefer a ZIP that is inside NYC


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
    try:
        resp = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as err:
        if err.code != 429:
            raise
        time.sleep(1.5)  # Nominatim rate limit (about 1 request/second): wait once, then retry
        resp = urllib.request.urlopen(req, timeout=10)
    with resp:
        address = json.loads(resp.read().decode("utf-8")).get("address", {})
    _ADDRESS_CACHE[key] = address
    return address


def nyc_context(lat: float, lon: float, zip_hint=None):
    """Return a context dict for the card, or None if not in NYC / data missing.

    zip_hint: what the user typed, if anything. Typed ZIPs and GPS points inside a clearly-NYC ZIP
    area resolve without any network call.
    """
    if not _BASE:
        print("[NYC context] nyc_baseline.json not found or unreadable - card hidden. "
              "Run preprocess_nyc_data.py and keep the file next to app.py.")
        return None
    try:
        zip_code = parse_typed_zip(zip_hint)
        borough = None

        # 1. typed ZIP
        if zip_code:
            status, borough = classify_zip(zip_code)
            if status in ("outside", "partial"):
                why = "is mostly outside NYC" if status == "partial" else "is not an NYC ZIP"
                print(f"[NYC context] ZIP {zip_code} {why} - card hidden. (Use GPS or a street address inside NYC.)")
                return None

        # 2. locate the coordinates inside a ZIP boundary, locally (polygons are clipped to NYC)
        if not borough and _ZIPS:
            x0, y0, x1, y1 = _NYC_BBOX
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                print("[NYC context] location is outside the NYC area - card hidden.")
                return None
            found = zip_for_point(lat, lon)
            if found:
                borough, zip_code = _ZIPS[found]["borough"], found

        # 3. last resort: county from Nominatim (point in no polygon, or no boundary file loaded)
        if not borough:
            addr = _reverse_geocode(lat, lon)
            county = str(addr.get("county", "")).strip()
            borough = COUNTY_TO_BOROUGH.get(county.lower())
            zip_code = str(addr.get("postcode", ""))[:5] or zip_code
            if not borough:
                print(f"[NYC context] location is not in NYC (county: '{county or 'unknown'}') - card hidden.")
                return None  # outside the five boroughs: hide the card

        pm = _BASE["pm25_annual_mean"]
        cv = _BASE["cv_hospitalizations_due_to_pm25_age40plus"]

        uhf_id = _ZIP_TO_UHF.get(str(zip_code or "")[:5])
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
    except Exception as exc:
        print(f"[NYC context] lookup failed ({type(exc).__name__}: {exc}) - card hidden.")
        return None
