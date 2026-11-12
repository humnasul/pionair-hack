'''
Humna Sultan
PURPOSE OF THE FILE:
- utilize trained_model.json (if available) to compute a personalized stroke risk score based on user inputs and local air quality data
- utilizes: logistic regression, PurpleAir API, OpenWeatherMap API, geocoding, HTML/CSS/JS frontend
- NYC Open Data context: shows the user's neighborhood long-term PM2.5 average and a modeled health-burden figure
  (display only, never used in the risk score), and falls back to the neighborhood average, clearly flagged, when
  no live sensor is available in NYC. Built from nyc_baseline.json (see preprocess_nyc_data.py, nyc_context.py)
- AQI-band guardrails keep each alert level inside the range allowed for today's EPA AQI band, and the care-circle
  popup needs Level 4+ AND AQI 101+ (see AQI_LEVEL_RANGES and CARE_CIRCLE_MIN_AQI)
- the air effect in the score is a published estimate (short-term PM2.5 and stroke admission or mortality, Shah et al.
  2015, RR 1.011 per 10 ug/m3), not a fitted or hand-picked weight; EPA-listed sensitive groups (age 65+ or heart
  disease) get one extra step of caution at AQI 101 and above
'''
import http.server
import json
import math
import os
import time
import traceback
import urllib.parse
import urllib.request
import webbrowser
# imports for ml processing, web server, and API requests

# NYC neighborhood context (display only). The app still runs if nyc_context.py is missing.
try:
    from nyc_context import nyc_context
except ImportError:
    def nyc_context(lat, lon):
        return None

try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv(override=True)
except ImportError:
    _dotenv_loaded = False  # python-dotenv not installed - run: pip install python-dotenv
# Load environment variables from .env file if available, otherwise continue without it
# the env vars are API keys for PurpleAir and OpenWeatherMap, which are used to fetch air quality data

MODEL_JSON_PATH = "trained_model.json"
PURPLEAIR_API_READ_KEY = os.getenv("PURPLEAIR_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
# manually set default API keys for PurpleAir and OpenWeatherMap if not found in environment variables
MAX_SENSOR_AGE_SECONDS = 3600  # ignore PurpleAir sensors that have not reported in the last hour

# if needed, print startup diagnostics to verify environment variables and working directory
def _print_startup_diagnostics():
    print("-" * 60)
    print("STARTUP DIAGNOSTICS")
    print(f"  Working directory:      {os.getcwd()}")
    print(f"  .env file found here:   {os.path.exists('.env')}")
    print(f"  python-dotenv loaded:   {_dotenv_loaded}")

    def _status(name, value):
        if not value:
            return "NOT SET (empty)"
        return f"SET ({value[:4]}...{value[-4:]}, {len(value)} chars)"

    print(f"  PURPLEAIR_API_KEY:      {_status('PURPLEAIR_API_KEY', PURPLEAIR_API_READ_KEY)}")
    print(f"  OPENWEATHER_API_KEY:    {_status('OPENWEATHER_API_KEY', OPENWEATHER_API_KEY)}")
    print("-" * 60)

# ------------------------------------------------------------------------------
# 1. MODEL COEFFICIENT LOADER (only PM2.5 feeds the air term; NO2 and O3 are display-only)
# ------------------------------------------------------------------------------
# Fallback values mirror the current trained_model.json, so the app behaves the same even if that file is missing
DEFAULT_INTERCEPT = -4.50
DEFAULT_COEFS = {
    "age_over_40": 0.4125,
    "heart": 0.1033,
    "hypertension": 0.2944,
    # Smoking is a three-level categorical (never / former / current). Former-smoker stroke
    # risk returns to roughly the never-smoker baseline within ~5 years of quitting
    # (Surgeon General reports; INTERSTROKE), so the two levels get very different weights.
    # Each is anchored to literature and shrunk by 0.53 like the other pinned terms:
    #   - current: ln(1.67) * 0.53 ~= 0.272 (INTERSTROKE OR ~1.67 for ischemic stroke)
    #   - former:  ln(1.05) * 0.53 ~= 0.026 (near-baseline for long-quit smokers)
    "smoke_current": 0.272,
    "smoke_former": 0.026,
    "fam_stroke": 0.1783,
    "fam_smoke": 0.1097,
}
# Air term (published estimate, not fitted): Shah et al., BMJ 2015;350:h1295, short-term PM2.5 and stroke admission or
# mortality, RR 1.011 per 10 ug/m3. Air at or below the reference level adds no risk.
DEFAULT_AIR = {"pm25_log_rr_per_10ug": 0.01094, "reference_pm25": 5.0}
INTERCEPT = DEFAULT_INTERCEPT
COEFS = dict(DEFAULT_COEFS)
AIR = dict(DEFAULT_AIR)
if os.path.exists(MODEL_JSON_PATH): # if the trained_model.json file exists, load the model coefficients from it
    with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
        _loaded = json.load(f)
    INTERCEPT = _loaded.get("intercept", DEFAULT_INTERCEPT)
    COEFS.update({k: v for k, v in _loaded.get("coefficients", {}).items() if k in DEFAULT_COEFS})  # missing ones keep defaults
    AIR.update(_loaded.get("air", {}))
    if "aqi_scaled" in _loaded.get("coefficients", {}):
        print("[NOTE] trained_model.json is from an older version with an AQI weight; that weight is ignored. "
              "Rerun testing_datasets.py to rebuild it.")
    _loaded_coefs = _loaded.get("coefficients", {})
    if "smoke" in _loaded_coefs and "smoke_current" not in _loaded_coefs:
        print("[NOTE] trained_model.json uses the old single 'smoke' coefficient. Smoking is now a "
              "three-level categorical (never / former / current); using the literature-anchored "
              "defaults for smoke_current and smoke_former instead. Rerun testing_datasets.py to rebuild.")
else:
    print("[WARNING] trained_model.json not found - using the built-in default coefficients.")

# thresholds for risk levels based on the model's predicted probability
# these cutoffs are hand-chosen design thresholds (not derived from data and not clinically validated)
LEVEL_THRESHOLDS = [
    (0.12, 5, "#ef4444", "Rest & Recharge Indoors", "🚨 High Environmental Caution", "Your body is working harder today due to outside air. Close windows, turn on indoor air cooling or filtering, and relax comfortably inside. If you feel sudden weakness, dizziness, or shortness of breath, please check in with a doctor or health professional right away."),
    (0.07, 4, "#f97316", "Stay Indoors or Mask Up", "⚠️ Elevated Indoor Caution", "Skip heavier outdoor tasks or workouts today. Stay comfortable inside with clean air, drink fresh water, and keep your regular prescriptions nearby."),
    (0.04, 3, "#f59e0b", "Pace Yourself", "👀 Yellow Caution", "If you have heart or blood pressure concerns, take things slow outside. Avoid long walks along busy roads while the air is hazy or dusty."),
    (0.02, 2, "#10b981", "Looking Safe", "👍 Green Light", "Your body and the outdoor air are in good harmony today. You are free to run errands, play, and go on walks."),
    (0.00, 1, "#10b981", "Prime Conditions", "🌟 Clear Sailing", "Air is crisp and your personal baseline is calm. Enjoy the outdoors, open up the windows, and stay active!"),
]

# AQI-band guardrails: the EPA AQI category sets the RANGE of alert levels that is allowed.
# The model's score still decides where a person lands inside that range, but it can no longer make clean air
# look alarming (e.g., Level 4 at AQI 20) or dangerous air look fine (e.g., Level 2 at AQI 250).
# These ranges are a design rule anchored to the EPA AQI category cutoffs; they are not clinically validated.
AQI_LEVEL_RANGES = [
    # (highest AQI in the band, lowest allowed level, highest allowed level)
    (50, 1, 2),             # Good
    (100, 1, 3),            # Moderate
    (150, 2, 4),            # Unhealthy for Sensitive Groups
    (200, 3, 5),            # Unhealthy
    (float("inf"), 4, 5),   # Very Unhealthy and Hazardous
]

# the care-circle popup only appears when the air is at least this bad (AQI 101 = Unhealthy for Sensitive Groups)
CARE_CIRCLE_MIN_AQI = 101

# EPA sensitive groups for particle pollution include people with heart or lung disease, older adults, and children
# (AirNow). The app only asks about age and heart disease, so those are the two it can use. The EPA gives no age for
# "older adult"; 65 is a common convention and is an assumption here. Hypertension is not on the EPA list, so it is not used.
SENSITIVE_MIN_AGE = 65
# at or above this AQI, EPA guidance tells sensitive groups to act (Unhealthy for Sensitive Groups starts at 101)
SENSITIVE_RULE_MIN_AQI = 101

# lookup by level number: (lower probability cutoff, color, label, headline, advisory)
LEVEL_INFO = {lvl: (thr, color, label, headline, advisory) for thr, lvl, color, label, headline, advisory in LEVEL_THRESHOLDS}


def is_sensitive_group(age: float, heart: bool) -> bool:
    """EPA-listed sensitive group that the app can identify: older adults or people with heart disease."""
    return bool(heart) or age >= SENSITIVE_MIN_AGE


def apply_aqi_guardrails(model_level: int, aqi: float, sensitive: bool = False) -> int:
    """Clamp the model's level into the range allowed for today's AQI band.

    Sensitive groups also get one extra step at AQI 101 and above: their lowest allowed level is the band's lowest
    level plus one (capped at 5), because EPA guidance tells them to take action earlier than everyone else.
    """
    aqi_whole = round(aqi)  # AQI is reported as a whole number
    for highest_aqi, lowest_level, highest_level in AQI_LEVEL_RANGES:
        if aqi_whole <= highest_aqi:
            level = max(lowest_level, min(highest_level, model_level))
            if sensitive and aqi_whole >= SENSITIVE_RULE_MIN_AQI:
                level = max(level, min(5, lowest_level + 1))
            return level
    return model_level


def meter_probability(prob: float, level: int) -> float:
    """Probability used only to place the thermometer pin, kept inside the displayed level's own band
    so the pin always agrees with the label and color after a guardrail changes the level."""
    low = LEVEL_INFO[level][0]
    if level >= 5:
        return max(low, prob)
    high = LEVEL_INFO[level + 1][0]
    return max(low, min(prob, high - 1e-9))


def should_alert_care_circle(level: int, aqi: float, aqi_estimated: bool = False) -> bool:
    """Care-circle popup: Level 4 or 5 AND AQI at least 101, and never for a neighborhood-average estimate."""
    return level >= 4 and round(aqi) >= CARE_CIRCLE_MIN_AQI and not aqi_estimated


# ------------------------------------------------------------------------------
# 2. PURPLEAIR SENSOR INTEGRATION & EPA CALIBRATION
# ------------------------------------------------------------------------------
 # Convert PM2.5 concentration (µg/m³) to EPA AQI using the standard breakpoints
 # pm2.5 values are converted to AQI values based on the EPA's defined breakpoints for air quality categories
def pm25_to_aqi(pm: float) -> float:
    breakpoints = [
        (0.0, 9.0, 0.0, 50.0),
        (9.0, 35.4, 51.0, 100.0),
        (35.4, 55.4, 101.0, 150.0),
        (55.4, 125.4, 151.0, 200.0),
        (125.4, 225.4, 201.0, 300.0),
        (225.4, 500.4, 301.0, 500.0),
    ]
    if pm <= 0:
        return 0.0
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm < c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low)
    return 500.0

# Approximate inverse of pm25_to_aqi: used only when a PM2.5 value is not supplied (for example by tests)
def aqi_to_pm25(aqi: float) -> float:
    breakpoints = [
        (0.0, 9.0, 0.0, 50.0),
        (9.0, 35.4, 51.0, 100.0),
        (35.4, 55.4, 101.0, 150.0),
        (55.4, 125.4, 151.0, 200.0),
        (125.4, 225.4, 201.0, 300.0),
        (225.4, 500.4, 301.0, 500.0),
    ]
    if aqi <= 0:
        return 0.0
    for c_low, c_high, i_low, i_high in breakpoints:
        if aqi <= i_high:
            return max(c_low, c_low + (aqi - i_low) * (c_high - c_low) / (i_high - i_low))
    return 500.4


# Map EPA AQI values to descriptive categories and colors
# based on the EPA's Air Quality Index (AQI) scale, which categorizes air quality into six levels ranging from "Good" to "Hazardous"
def get_epa_aqi_category(aqi: float) -> dict:
    if aqi <= 50:
        return {"category": "Fresh & Clean", "color": "#10b981", "bg": "#ecfdf5", "simple_desc": "Clean and fresh air. Perfect for outdoor fun."}
    elif aqi <= 100:
        return {"category": "Fair / Moderate", "color": "#f59e0b", "bg": "#fffbeb", "simple_desc": "Generally fine, though sensitive lungs might feel a tickle."}
    elif aqi <= 150:
        return {"category": "Smoky / Hazy", "color": "#f97316", "bg": "#fff7ed", "simple_desc": "Fine dust particles are elevated. Vulnerable groups should limit time outside."}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#ef4444", "bg": "#fef2f2", "simple_desc": "Dust and smoke particles are elevated today. Staying inside is recommended."}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "color": "#8b5cf6", "bg": "#f5f3ff", "simple_desc": "Heavy smoke or smog pollution. Seal windows and avoid all outdoor trips."}
    else:
        return {"category": "Hazardous", "color": "#991b1b", "bg": "#fef2f2", "simple_desc": "Very heavy smoke conditions. Run indoor filtration continuously."}

# get user's location coordinates (latitude, longitude) and display name using OpenStreetMap's Nominatim API
def geocode_address(query: str) -> tuple[float, float, str]:
    encoded = urllib.parse.quote(query.strip())
    url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1&countrycodes=us"
    req = urllib.request.Request(url, headers={"User-Agent": "PlayfulHealthAir/9.0 (community-check)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if not data:
        fallback_url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
        req_fallback = urllib.request.Request(fallback_url, headers={"User-Agent": "PlayfulHealthAir/9.0 (community-check)"})
        with urllib.request.urlopen(req_fallback, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            raise ValueError(f"Could not find '{query}'. Try typing a 5-digit zip code or city name.")

    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]

# get the nearest PurpleAir sensor's AQI, name, pollutant details, and humidity based on the user's coordinates
def fetch_purpleair_aqi(lat: float, lon: float) -> tuple[float, str, str, float, float]:
    if not PURPLEAIR_API_READ_KEY:
        raise ValueError("PurpleAir API key is missing. Set the PURPLEAIR_API_KEY environment variable.")

    # define a search radius of 0.05 degrees (~5.5 km) around the user's location to find nearby PurpleAir sensors
    delta = 0.05
    params = {
        "fields": "name,latitude,longitude,pm2.5_cf_1,humidity,last_seen,location_type",
        "location_type": 0,
        "nwlat": lat + delta,
        "nwlng": lon - delta,
        "selat": lat - delta,
        "selng": lon + delta,
    }
    url = f"https://api.purpleair.com/v1/sensors?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": PURPLEAIR_API_READ_KEY})

    # url request to PurpleAir API to fetch sensor data within the defined bounding box
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    # get the list of fields and data rows from the API response
    fields = payload.get("fields", [])
    data_rows = payload.get("data", [])
    if not data_rows:
        raise RuntimeError("No working neighborhood air sensors found close to this spot.")

    # create a mapping of field names to their indices for easier access to sensor data
    idx = {f: i for i, f in enumerate(fields)}
    closest_row = None
    min_dist = float("inf")
    now = time.time()

    # latitude and longitude of the user's location are compared to each sensor's coordinates to find the closest sensor
    for row in data_rows:
        # skip sensors with no PM2.5 reading or no recent update, so missing data is never treated as clean air (0.0)
        if row[idx["pm2.5_cf_1"]] is None or row[idx["last_seen"]] is None:
            continue
        if now - row[idx["last_seen"]] > MAX_SENSOR_AGE_SECONDS:
            continue
        s_lat = row[idx["latitude"]]
        s_lon = row[idx["longitude"]]
        # shrink the longitude difference by cos(latitude) so degrees of lat/lon are comparable
        dist = math.hypot(s_lat - lat, (s_lon - lon) * math.cos(math.radians(lat)))
        if dist < min_dist:
            min_dist = dist
            closest_row = row

    if closest_row is None:
        raise RuntimeError("No neighborhood air sensors with a recent reading were found close to this spot.")

    # get raw values
    sensor_name = closest_row[idx["name"]]
    raw_pm25 = float(closest_row[idx["pm2.5_cf_1"]])
    humidity = float(closest_row[idx["humidity"]] or 50.0)

    # EPA calibration formula for PM2.5 based on PurpleAir's raw readings and humidity, which adjusts the raw PM2.5 value to better match EPA's reference monitors
    epa_pm25 = max(0.5, 0.524 * raw_pm25 - 0.0862 * humidity + 5.75)
    aqi = pm25_to_aqi(epa_pm25)
    pollutant_detail = f"{epa_pm25:.1f} µg/m³"

    return aqi, sensor_name, pollutant_detail, epa_pm25, humidity

# get the nearest OpenWeatherMap sensor's pollutant data (PM2.5, O3, NO2, PM10, SO2, CO) and AQI index based on the user's coordinates
def fetch_openweather_pollutants(lat: float, lon: float) -> dict:
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap API key is missing. Set the OPENWEATHER_API_KEY environment variable.")

    # construct the API request URL with the user's coordinates and the OpenWeatherMap API key to fetch air pollution data
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)

    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if not payload.get("list"):
        # No usable data; return all fields as None so ozone and NO2 show as N/A on the page.
        return {"pm2_5": None, "o3": None, "no2": None, "pm10": None, "so2": None, "co": None, "owm_aqi_index": None}

    # extract the pollutant components and AQI index from the API response, which includes concentrations of various pollutants and an overall AQI index based on the US EPA scale
    components = payload["list"][0]["components"]
    return {
        "pm2_5": components.get("pm2_5"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "pm10": components.get("pm10"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "owm_aqi_index": payload["list"][0]["main"]["aqi"],
    }


# ------------------------------------------------------------------------------
# 3. INTERACTIVE LOGISTIC REGRESSION SCORING (Calculates with AQI only)
# ------------------------------------------------------------------------------
# compute the detailed stroke risk score based on user inputs and local air quality data using the logistic regression model coefficients
def compute_detailed_risk(
    age: int, hypertension: bool, heart: bool, smoke_status: str, fam_smoke: bool, fam_stroke: bool,
    aqi: float, pm25: float | None = None
) -> tuple[float, int, str, str, str, str, int, float]:
    # returns: probability, final level, color, label, headline, advisory, the model's own level (before guardrails), meter probability
    age_over_40 = max(0.0, (age - 40) / 10.0)
    if pm25 is None:
        pm25 = aqi_to_pm25(aqi)  # the air term is based on PM2.5 in ug/m3, not on the AQI number
    h_val = 1.0 if heart else 0.0
    ht_val = 1.0 if hypertension else 0.0
    f_stroke_val = 1.0 if fam_stroke else 0.0
    f_smoke_val = 1.0 if fam_smoke else 0.0
    # Pick the smoking coefficient for this person's status ("never"/"former"/"current").
    # Anything unrecognised (including None or missing) is treated as "never" (no contribution).
    if smoke_status == "current":
        smoke_coef = COEFS["smoke_current"]
    elif smoke_status == "former":
        smoke_coef = COEFS["smoke_former"]
    else:
        smoke_coef = 0.0
    # convert user inputs into numerical values for the logistic regression calculation, where binary features are represented as 1.0 (true) or 0.0 (false), and age is scaled in decades over 40

    logit = (
        INTERCEPT
        + COEFS["age_over_40"] * age_over_40
        + COEFS["heart"] * h_val
        + COEFS["hypertension"] * ht_val
        + smoke_coef
        + COEFS["fam_stroke"] * f_stroke_val
        + COEFS["fam_smoke"] * f_smoke_val
        + AIR["pm25_log_rr_per_10ug"] * max(0.0, pm25 - AIR["reference_pm25"]) / 10.0
    )
    # logit = linear combination of features and their weights (from trained_model.json, or the built-in defaults),
    # plus the literature air term: ln(RR) per 10 ug/m3 of PM2.5 above the reference level (zero at or below it)

    prob = 1.0 / (1.0 + math.exp(-logit))
    # convert the logit to a probability using the logistic function, which maps any real-valued number into the (0, 1) interval

    # find the model's own level from the predicted probability using the thresholds
    model_level = next(level for threshold, level, *_ in LEVEL_THRESHOLDS if prob >= threshold)

    # then keep that level inside the range allowed for today's AQI band (sensitive groups get one extra step at AQI 101+)
    level = apply_aqi_guardrails(model_level, aqi, is_sensitive_group(age, heart))
    _, color, label, headline, advisory = LEVEL_INFO[level]

    return prob, level, color, label, headline, advisory, model_level, meter_probability(prob, level)


# ------------------------------------------------------------------------------
# 4. DASHBOARD FRONTEND TEMPLATE
# ------------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pionair - Health & Air Checker</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --brand: #6366f1;
      --brand-hover: #4f46e5;
      --border: #e2e8f0;
      --shadow: 0 12px 30px -4px rgba(99, 102, 241, 0.08), 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      max-width: 660px;
      margin: 32px auto;
      padding: 0 16px 40px;
      color: var(--text);
      background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
      line-height: 1.6;
    }
    .card {
      background: var(--card-bg);
      padding: 34px 28px;
      border-radius: 28px;
      box-shadow: var(--shadow);
      border: 2px solid rgba(255, 255, 255, 0.8);
    }
    .header-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: #e0e7ff;
      color: #4338ca;
      font-size: 13px;
      font-weight: 800;
      border-radius: 999px;
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    h1 {
      margin: 0 0 6px 0;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }
    .subtitle {
      font-size: 16px;
      color: var(--muted);
      margin-bottom: 26px;
    }

    .form-group { margin-bottom: 20px; }
    label.field-label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 15px;
    }
    input[type="number"], input[type="text"] {
      width: 100%;
      padding: 15px 18px;
      border: 2px solid var(--border);
      border-radius: 16px;
      font-size: 16px;
      font-family: inherit;
      color: var(--text);
      background: #ffffff;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    input[type="number"]:focus, input[type="text"]:focus {
      border-color: var(--brand);
      box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
    }

    .checkbox-group {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 22px;
    }
    .check-button {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 15px 18px;
      border: 2px solid var(--border);
      border-radius: 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      background: #f8fafc;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .check-button:hover {
      background: #ffffff;
      border-color: #cbd5e1;
      transform: translateY(-1px);
    }
    .check-button input {
      width: 22px;
      height: 22px;
      cursor: pointer;
      accent-color: var(--brand);
    }

    .toggle-row {
      display: flex;
      gap: 16px;
      margin-bottom: 12px;
      font-size: 14px;
      font-weight: 700;
      flex-wrap: wrap;
    }
    .toggle-row label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }
    .toggle-row input {
      width: 18px;
      height: 18px;
      accent-color: var(--brand);
    }

    .ac-choice-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .ac-choice-card {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 14px 18px;
      border: 2px solid var(--border);
      border-radius: 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      background: #f8fafc;
      transition: all 0.2s ease;
    }
    .ac-choice-card:hover {
      background: #ffffff;
      border-color: #cbd5e1;
    }
    .ac-choice-card input {
      width: 20px;
      height: 20px;
      accent-color: var(--brand);
      cursor: pointer;
    }

    .btn-submit {
      width: 100%;
      padding: 18px;
      background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
      color: #ffffff;
      border: none;
      border-radius: 18px;
      font-weight: 800;
      font-size: 18px;
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.4);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 24px -4px rgba(99, 102, 241, 0.5);
    }
    .btn-submit:disabled {
      background: #94a3b8;
      box-shadow: none;
      transform: none;
      cursor: not-allowed;
    }

    #status {
      margin-top: 14px;
      font-size: 14.5px;
      font-weight: 700;
      color: var(--muted);
      text-align: center;
    }

    /* Result Dashboard */
    #result {
      margin-top: 32px;
      display: none;
      animation: bounceIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes bounceIn {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }

    .monitor-tag {
      background: #f1f5f9;
      border-radius: 14px;
      padding: 10px 16px;
      font-size: 13px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .monitor-tag strong { color: var(--text); }

    .hero-stat-card {
      padding: 26px;
      border-radius: 22px;
      border: 3px solid;
      text-align: center;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
    }
    .hero-stat-title {
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
      opacity: 0.85;
    }
    .hero-stat-main {
      font-size: 34px;
      font-weight: 900;
      letter-spacing: -0.02em;
      line-height: 1.15;
      margin-bottom: 8px;
    }
    .hero-stat-sub {
      font-size: 16px;
      font-weight: 700;
    }

    .pair-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    @media (max-width: 560px) { .pair-grid { grid-template-columns: repeat(2, 1fr); } }
    .mini-card {
      background: #ffffff;
      border: 2px solid var(--border);
      border-radius: 18px;
      padding: 14px 10px;
      text-align: center;
    }
    .mini-card-label {
      font-size: 10.5px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .mini-card-num {
      font-size: 24px;
      font-weight: 900;
      line-height: 1.1;
    }
    .mini-card-unit {
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      margin-top: 1px;
    }
    .mini-card-desc {
      font-size: 12px;
      font-weight: 700;
      margin-top: 4px;
    }
    /* Disclaimers and source notes: neutral slate with a darker left bar so they stand out without looking like an alert */
    .estimated-note, .disclaimer {
      font-size: 13px;
      font-weight: 600;
      line-height: 1.45;
      color: #1e293b;
      background: #e2e8f0;
      border: 1px solid #cbd5e1;
      border-left: 5px solid #64748b;
      border-radius: 10px;
      padding: 10px 14px;
      margin-bottom: 16px;
    }
    .estimated-note { display: none; }
    /* Friendly "PionAir is not working right now" box, shown only when no live air data is available */
    .unavailable-box {
      text-align: left;
      background: #ffffff;
      border: 1.5px solid #cbd5e1;
      border-radius: 16px;
      padding: 16px 18px;
      margin: 12px 0;
      color: #1e293b;
      font-size: 14px;
      line-height: 1.5;
    }
    .unavailable-title { font-size: 17px; font-weight: 800; margin-bottom: 6px; }
    .unavailable-box p { margin: 0; }

    /* Pure Green -> Lime -> Yellow -> Orange -> Red Gradient Track */
    .meter-box {
      background: #ffffff;
      border: 2px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .meter-label-row {
      display: flex;
      justify-content: space-between;
      font-weight: 800;
      font-size: 14px;
      margin-bottom: 12px;
    }
    .meter-track {
      height: 20px;
      border-radius: 10px;
      background: linear-gradient(to right, #10b981 0%, #84cc16 25%, #eab308 50%, #f97316 75%, #ef4444 100%);
      position: relative;
    }
    .meter-pin {
      position: absolute;
      top: -6px;
      width: 32px;
      height: 32px;
      background: #ffffff;
      border: 4px solid var(--text);
      border-radius: 50%;
      transform: translateX(-50%);
      box-shadow: 0 4px 10px rgba(0,0,0,0.15);
      transition: left 0.4s ease;
    }
    .meter-steps {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
      margin-top: 10px;
    }

    .advisory-bubble {
      border-radius: 20px;
      padding: 20px 22px;
      border: 2px solid;
      margin-bottom: 24px;
    }
    .advisory-headline {
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 6px;
    }
    .advisory-body {
      font-size: 15px;
      font-weight: 600;
      line-height: 1.55;
    }

    .explainer-card {
      background: #f8fafc;
      border: 2px dashed #cbd5e1;
      border-radius: 20px;
      padding: 22px;
    }
    .explainer-title {
      font-size: 16px;
      font-weight: 800;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .explainer-p {
      font-size: 14px;
      color: var(--muted);
      margin: 0 0 12px 0;
      line-height: 1.5;
    }
    .explainer-p:last-child { margin-bottom: 0; }

    /* Friendly, Calming Modal (Strictly No Red) */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(5px);
      display: none;
      justify-content: center;
      align-items: center;
      z-index: 9999;
      animation: fadeIn 0.25s ease-out;
    }
    .modal-dialog {
      background: #ffffff;
      padding: 32px 28px;
      border-radius: 26px;
      max-width: 460px;
      width: 90%;
      text-align: center;
      box-shadow: 0 20px 45px -10px rgba(99, 102, 241, 0.2);
      border: 2px solid #e0e7ff;
      animation: popUp 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .modal-icon {
      font-size: 46px;
      margin-bottom: 12px;
      line-height: 1;
    }
    .modal-title {
      font-size: 21px;
      font-weight: 800;
      color: #0f172a;
      margin-bottom: 8px;
    }
    .modal-body {
      font-size: 15.5px;
      font-weight: 600;
      color: #475569;
      line-height: 1.55;
      margin-bottom: 22px;
    }
    .duration-options {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }
    .duration-btn {
      padding: 13px 10px;
      background: #f8fafc;
      border: 2px solid #e2e8f0;
      border-radius: 14px;
      font-size: 14px;
      font-weight: 700;
      color: #1e293b;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .duration-btn:hover {
      background: #eef2ff;
      border-color: #6366f1;
      color: #4338ca;
    }
    .modal-primary-btn {
      width: 100%;
      padding: 15px;
      background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
      color: #ffffff;
      border: none;
      border-radius: 16px;
      font-weight: 800;
      font-size: 16px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
      transition: transform 0.15s;
    }
    .modal-primary-btn:hover {
      transform: translateY(-1px);
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes popUp {
      from { opacity: 0; transform: scale(0.94); }
      to { opacity: 1; transform: scale(1); }
    }
  </style>
</head>
<body>
<div class="card">
  <div class="header-pill">🌤️ Local Air + Body Shield</div>
  <h1>Pionair Health Checker</h1>
  <div class="subtitle">See how clean your neighborhood air is and get simple tips tailored to you.</div>

  <div class="form-group">
    <label class="field-label" for="age">1. What is your age?</label>
    <input type="number" id="age" value="50" min="18" max="120">
  </div>

  <div class="form-group">
    <label class="field-label">2. Tap any health conditions you have:</label>
    <div class="checkbox-group">
      <label class="check-button">
        <input type="checkbox" id="hypertension">
        <span>🩺 High Blood Pressure</span>
      </label>
      <label class="check-button">
        <input type="checkbox" id="heart">
        <span>❤️ Heart Conditions (chest pain or prior heart issues)</span>
      </label>
      <label class="check-button">
        <input type="checkbox" id="fam_stroke">
        <span>🧬 Family History of Stroke</span>
      </label>
      <label class="check-button">
        <input type="checkbox" id="fam_smoke">
        <span>💨 Secondhand Smoke (Someone smokes at home)</span>
      </label>
    </div>
  </div>

  <div class="form-group">
    <label class="field-label">3. What is your smoking status?</label>
    <div class="checkbox-group">
      <label class="check-button">
        <input type="radio" name="smoke_status" value="never" checked>
        <span>🚭 Never smoked tobacco</span>
      </label>
      <label class="check-button">
        <input type="radio" name="smoke_status" value="former">
        <span>🍃 Used to smoke (quit 5+ years ago)</span>
      </label>
      <label class="check-button">
        <input type="radio" name="smoke_status" value="current">
        <span>🚬 I currently smoke tobacco</span>
      </label>
    </div>
  </div>

  <div class="form-group">
    <label class="field-label">4. Do you have air conditioning indoors?</label>
    <div class="ac-choice-grid">
      <label class="ac-choice-card">
        <input type="radio" name="has_ac" value="yes" checked>
        <span>❄️ Yes, I have A/C</span>
      </label>
      <label class="ac-choice-card">
        <input type="radio" name="has_ac" value="no">
        <span>🪟 No A/C</span>
      </label>
    </div>
  </div>

  <div class="form-group">
    <label class="field-label">5. Where are you?</label>
    <div class="toggle-row">
      <label><input type="radio" name="loc_mode" value="address" checked onclick="toggleMode()"> Enter City or Zip</label>
      <label><input type="radio" name="loc_mode" value="device" onclick="toggleMode()"> Phone GPS</label>
      <label><input type="radio" name="loc_mode" value="manual" onclick="toggleMode()"> Manual AQI (Testing)</label>
    </div>
    <div id="address-box">
      <input type="text" id="address_input" placeholder="e.g. 10044 or Roosevelt Island, NY" value="10044">
    </div>
    <div id="manual-aqi-box" style="display: none;">
      <input type="number" id="manual_aqi_input" placeholder="Type AQI value (e.g. 200, 300)" value="200" min="0" max="500">
    </div>
  </div>

  <button class="btn-submit" id="submit-btn" onclick="evaluateRisk()">Check My Air Safety Score</button>
  <div id="status"></div>

  <!-- Results View -->
  <div id="result">
    <div class="monitor-tag">
      <span>📍 <strong>Location:</strong> <span id="res-location"></span></span>
      <span>📡 <strong>Sensor:</strong> <span id="res-station"></span></span>
    </div>

    <!-- Main Hero Stat -->
    <div class="hero-stat-card" id="hero-card">
      <div class="hero-stat-title">Today's Safety Verdict</div>
      <div class="hero-stat-main" id="hero-label">--</div>
    </div>

    <!-- Details Grid: Values displayed, calculation unaffected -->
    <div class="estimated-note" id="estimated-note">ℹ️ Ozone and NO₂ data is not available right now, so they show N/A. They never change your score.</div>
    <div class="estimated-note" id="sensitive-note">ℹ️ You are in a group the EPA lists as sensitive to particle pollution (age 65 or older, or heart disease), so your alert level is one step higher when the AQI is 101 or above.</div>
    <div class="estimated-note" id="aqi-estimated-note">ℹ️ No live sensor found nearby — this score uses the typical level for your neighborhood, not today's air.</div>
    <div class="pair-grid">
      <div class="mini-card" id="air-card">
        <div class="mini-card-label">AQI</div>
        <div class="mini-card-num" id="air-num">--</div>
        <div class="mini-card-desc" id="air-desc">--</div>
      </div>
      <div class="mini-card" id="pm25-card">
        <div class="mini-card-label">PM2.5</div>
        <div class="mini-card-num" id="pm25-num" style="color: #6366f1;">--</div>
        <div class="mini-card-unit">µg/m³</div>
        <div class="mini-card-desc" id="pm25-src" style="color:#64748b;"></div>
      </div>
      <div class="mini-card" id="o3-card">
        <div class="mini-card-label">Ozone (O₃)</div>
        <div class="mini-card-num" id="o3-num" style="color: #0d9488;">--</div>
        <div class="mini-card-unit">µg/m³</div>
      </div>
      <div class="mini-card" id="no2-card">
        <div class="mini-card-label">NO₂</div>
        <div class="mini-card-num" id="no2-num" style="color: #0ea5e9;">--</div>
        <div class="mini-card-unit">µg/m³</div>
      </div>
    </div>
    <p class="disclaimer">
      Sources: the AQI and PM2.5 come from the nearest PurpleAir sensor and drive your score. In NYC, if no sensor is available, both use the neighborhood's yearly average instead. Ozone and NO₂ are modeled estimates from OpenWeatherMap, shown for context only, and show N/A when unavailable.
    </p>

    <!-- NYC neighborhood context (display only; not used in the score) -->
    <div class="explainer-card" id="nyc-card" style="display:none; margin-bottom:20px;">
      <div class="explainer-title">🏙️ Your neighborhood, long term</div>
      <p class="explainer-p" id="nyc-pm"></p>
      <p class="explainer-p" id="nyc-cv"></p>
      <p class="disclaimer" style="margin-top:12px; margin-bottom:0;">Neighborhood PM2.5 is a yearly average from the NYC Health Department's air quality data (NYC Open Data, "Air Quality"), and the hospitalization figure is a modeled estimate from the same source. These are long-term background figures, not today's air, and they do not change your score.</p>
    </div>

    <!-- Restored Classic Pure Green -> Lime -> Yellow -> Orange -> Red Thermometer Meter -->
    <div class="meter-box">
      <div class="meter-label-row">
        <span>Overall Safety Thermometer</span>
        <span id="meter-text-val">Checking...</span>
      </div>
      <div class="meter-track">
        <div class="meter-pin" id="meter-pin" style="left: 10%;"></div>
      </div>
      <div class="meter-steps">
        <span>Low</span>
        <span>Mild</span>
        <span>Moderate</span>
        <span>Elevated</span>
        <span>High</span>
      </div>
    </div>

    <!-- Advisory -->
    <div class="advisory-bubble" id="advisory-box">
      <div class="advisory-headline" id="advisory-title">What this means for you:</div>
      <div class="advisory-body" id="advisory-text">--</div>
    </div>

    <!-- Explainer -->
    <div class="explainer-card">
      <div class="explainer-title">💡 How does this work?</div>
      <p class="explainer-p">
        <strong>1. Real-time Air Quality:</strong> We track live ground-level AQI alongside ambient readings for fine particles (PM2.5), Ozone (O₃), and Nitrogen Dioxide (NO₂).
      </p>
      <p class="explainer-p">
        <strong>2. Your Clinical Profile:</strong> Your overall caution score combines your age and health history with today's outdoor AQI. Today's air sets the overall range of caution (clean air never triggers a high alert, and unhealthy air never reads as fine), and your health history moves you within that range. People the EPA lists as sensitive to particle pollution (age 65 or older, or heart disease) get one extra step of caution when the AQI is 101 or higher.
      </p>
      <p class="explainer-p">
        <strong>3. What your score means:</strong> A lower score means you can enjoy the outdoors without hesitation. A higher score is a gentle heads-up to avoid strenuous exercise until the air clears up.
      </p>
    </div>
  </div>
</div>

<!-- Friendly Safety Notification Popup Modal (Gentle Violet/Blue, Strictly No Red) -->
<div class="modal-overlay" id="safety-modal">
  <div class="modal-dialog">
    <div class="modal-icon" id="modal-icon">💙</div>
    <div class="modal-title" id="modal-title">Checking In On You</div>
    <div class="modal-body" id="modal-body">
      <!-- Injected dynamically based on AC and exposure time -->
    </div>
    <div id="modal-action-area">
      <!-- Buttons injected dynamically -->
    </div>
  </div>
</div>

<script>
let currentEvaluatedAQI = 50;

function toggleMode() {
  const mode = document.querySelector('input[name="loc_mode"]:checked').value;
  document.getElementById('address-box').style.display = mode === 'address' ? 'block' : 'none';
  document.getElementById('manual-aqi-box').style.display = mode === 'manual' ? 'block' : 'none';
}

function closeSafetyModal() {
  document.getElementById('safety-modal').style.display = 'none';
}

/* Research-supported threshold check:
   - AQI > 200: Healthy limit for sensitive/high-strain is <= 30 mins (0.5 hrs).
   - AQI 151-200: Limit is <= 60 mins (1 hr).
   - AQI <= 150: Limit is <= 120 mins (2 hrs).
*/
function checkTimeOutside(minutes) {
  let maxSafeMinutes = 60;
  if (currentEvaluatedAQI > 200) {
    maxSafeMinutes = 30;
  } else if (currentEvaluatedAQI > 150) {
    maxSafeMinutes = 60;
  } else {
    maxSafeMinutes = 120;
  }

  const titleEl = document.getElementById('modal-title');
  const iconEl = document.getElementById('modal-icon');
  const bodyEl = document.getElementById('modal-body');
  const actionArea = document.getElementById('modal-action-area');

  if (minutes > maxSafeMinutes) {
    iconEl.textContent = '🕊️';
    titleEl.textContent = 'Care Update Sent';
    bodyEl.textContent = "You may have been exposed to unhealthy air today - We sent a message to your loved one to check-in on you.";
  } else {
    iconEl.textContent = '🏡';
    titleEl.textContent = 'Safe & Sound Inside';
    bodyEl.textContent = "The weather isn't ideal outside. We're glad you stayed inside today!";
  }

  actionArea.innerHTML = '<button class="modal-primary-btn" onclick="closeSafetyModal()">Got It, Thanks!</button>';
}

function handleHighRiskAlert(hasAC, aqi) {
  currentEvaluatedAQI = aqi;
  const modal = document.getElementById('safety-modal');
  const iconEl = document.getElementById('modal-icon');
  const titleEl = document.getElementById('modal-title');
  const bodyEl = document.getElementById('modal-body');
  const actionArea = document.getElementById('modal-action-area');

  if (hasAC) {
    iconEl.textContent = '🌤️';
    titleEl.textContent = 'Quick Outdoor Check-In';
    bodyEl.textContent = "How long did you spend outside?";
    actionArea.innerHTML = `
      <div class="duration-options">
        <button class="duration-btn" onclick="checkTimeOutside(15)">Under 30 mins</button>
        <button class="duration-btn" onclick="checkTimeOutside(45)">30 – 60 mins</button>
        <button class="duration-btn" onclick="checkTimeOutside(90)">1 – 2 hours</button>
        <button class="duration-btn" onclick="checkTimeOutside(150)">Over 2 hours</button>
      </div>
    `;
  } else {
    iconEl.textContent = '🕊️';
    titleEl.textContent = 'Safety Check-In';
    bodyEl.textContent = "Take it easy today, you may be at risk of exposure to unhealthy air today.";
    actionArea.innerHTML = '<button class="modal-primary-btn" onclick="closeSafetyModal()">I Understand</button>';
  }

  modal.style.display = 'flex';
}

function displayResults(data) {
  document.getElementById("result").style.display = "block";
  document.getElementById("res-location").textContent = data.location_label;
  document.getElementById("res-station").textContent = data.station;

  // Hero Card
  const heroCard = document.getElementById("hero-card");
  heroCard.style.borderColor = data.color;
  heroCard.style.background = data.epa_category.bg;
  document.getElementById("hero-label").textContent = data.label;
  document.getElementById("hero-label").style.color = data.color;

  // Air Card
  const airCard = document.getElementById("air-card");
  airCard.style.borderColor = data.epa_category.color;
  document.getElementById("air-num").textContent = Math.round(data.aqi);
  document.getElementById("air-num").style.color = data.epa_category.color;
  document.getElementById("air-desc").textContent = data.epa_category.category;

  // Pollutant cards
  const p = data.pollutants || {};
  document.getElementById("pm25-num").textContent = p.pm2_5 != null ? p.pm2_5.toFixed(1) : "N/A";
  document.getElementById("o3-num").textContent = p.o3 != null ? p.o3.toFixed(1) : "N/A";
  document.getElementById("no2-num").textContent = p.no2 != null ? p.no2.toFixed(1) : "N/A";

  // Label the PM2.5 tile with where its value came from
  const pm25SrcEl = document.getElementById("pm25-src");
  if (data.pm25_source === "manual") pm25SrcEl.textContent = "approx from AQI";
  else if (data.pm25_source === "neighborhood") pm25SrcEl.textContent = "neighborhood avg";
  else if (data.pm25_source === "live") pm25SrcEl.textContent = "live sensor";
  else pm25SrcEl.textContent = "";

  const note = document.getElementById("estimated-note");
  note.style.display = data.pollutants_estimated ? "block" : "none";
  document.getElementById("aqi-estimated-note").style.display = data.aqi_estimated ? "block" : "none";
  document.getElementById("sensitive-note").style.display = data.sensitive_rule_active ? "block" : "none";

  // NYC neighborhood context card (display only)
  const nyc = data.nyc_context;
  const nycCard = document.getElementById("nyc-card");
  if (nyc) {
    document.getElementById("nyc-pm").textContent =
      `${nyc.name}: average PM2.5 of ${nyc.pm25_annual_mean} µg/m³ (${nyc.pm25_period}). NYC overall: ${nyc.citywide_pm25} µg/m³.`;
    const cvEl = document.getElementById("nyc-cv");
    if (nyc.cv_burden_per_100k != null) {
      cvEl.textContent = `Modeled PM2.5-related heart and blood vessel hospitalizations (age 40+): about ${nyc.cv_burden_per_100k} per 100,000 adults per year (${nyc.cv_period} estimate).`;
      cvEl.style.display = "block";
    } else {
      cvEl.style.display = "none";
    }
    nycCard.style.display = "block";
  } else {
    nycCard.style.display = "none";
  }

  // Thermometer Pin mapped to 0.0 - 0.12 probability scale
  const pin = document.getElementById("meter-pin");
  const pinPos = Math.min(96, Math.max(4, ((data.meter_probability ?? data.probability) / 0.12) * 100));
  pin.style.left = pinPos + "%";
  pin.style.borderColor = data.color;
  document.getElementById("meter-text-val").textContent = data.label;
  document.getElementById("meter-text-val").style.color = data.color;

  // Advisory
  const advBox = document.getElementById("advisory-box");
  advBox.style.background = data.epa_category.bg;
  advBox.style.borderColor = data.color;
  document.getElementById("advisory-title").textContent = data.headline;
  document.getElementById("advisory-title").style.color = data.color;
  document.getElementById("advisory-text").textContent = data.advisory;

  // Trigger friendly modal only for Level 4 or 5 with AQI 101 or higher (decided by the server)
  if (data.care_circle_alert) {
    const hasAC = document.querySelector('input[name="has_ac"]:checked').value === 'yes';
    handleHighRiskAlert(hasAC, data.aqi);
  }
}

async function evaluateRisk() {
  const btn = document.getElementById("submit-btn");
  const status = document.getElementById("status");
  const mode = document.querySelector('input[name="loc_mode"]:checked').value;

  btn.disabled = true;
  status.textContent = "Connecting with nearby neighborhood sensors...";

  const basePayload = {
    age: parseInt(document.getElementById("age").value, 10),
    hypertension: document.getElementById("hypertension").checked,
    heart: document.getElementById("heart").checked,
    smoke_status: document.querySelector('input[name="smoke_status"]:checked').value,
    fam_smoke: document.getElementById("fam_smoke").checked,
    fam_stroke: document.getElementById("fam_stroke").checked,
    has_ac: document.querySelector('input[name="has_ac"]:checked').value === 'yes'
  };

  try {
    let payload = { ...basePayload };
    if (mode === "address") {
      const q = document.getElementById("address_input").value.trim();
      if (!q) throw new Error("Please enter a zip code or city name.");
      payload.query = q;
    } else if (mode === "device") {
      status.textContent = "Getting phone GPS coordinates...";
      const pos = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: true, timeout: 10000 });
      });
      payload.lat = pos.coords.latitude;
      payload.lon = pos.coords.longitude;
    } else if (mode === "manual") {
      const manualVal = parseFloat(document.getElementById("manual_aqi_input").value);
      if (isNaN(manualVal) || manualVal < 0) throw new Error("Please enter a valid AQI number (e.g. 0 to 500).");
      payload.manual_aqi = manualVal;
    }

    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (data.kind === "unavailable") {
      status.innerHTML = `
        <div class="unavailable-box">
          <div class="unavailable-title">😷 PionAir isn't working right now</div>
          <p>We couldn't load live air quality data for this location, so we can't give you a result. Please try again later.</p>
          <p class="disclaimer" style="margin:12px 0 0;">Note: this prototype uses PurpleAir for live air data, which needs an API key (PURPLEAIR_API_KEY). If the key is missing, expired, or over its limit, or there is no sensor near you, this message appears. Inside NYC, the app falls back to the neighborhood average instead.</p>
        </div>`;
      return;
    }
    if (data.error) throw new Error(data.error);

    status.textContent = "";
    displayResults(data);
  } catch (err) {
    status.textContent = "Notice: " + err.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# 5. SERVER DISPATCH
# ------------------------------------------------------------------------------
class DataUnavailable(Exception):
    """Live air data is unavailable and there is no safe fallback (e.g. outside NYC without a working PurpleAir key).

    The server converts this into a friendly 503 message rather than the raw traceback.
    """


# HTTP server handler that serves the HTML page and handles API requests for evaluating risk based on user input and air quality data
class AppHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/evaluate":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            try:
                age = payload.get("age")
                if isinstance(age, bool) or not isinstance(age, (int, float)) or not (18 <= age <= 120):
                    raise ValueError("Please enter an age between 18 and 120.")
                ctx = None              # NYC neighborhood context (stays None for manual entry / outside NYC)
                aqi_estimated = False   # True only when AQI comes from the NYC neighborhood average, not a live sensor
                live_pm25 = None        # live sensor PM2.5, used only to compare with the neighborhood average
                model_pm25 = None       # PM2.5 (ug/m3) that drives the air term in the score
                pm25_source = "live"    # where the PM2.5 tile value came from: "live", "manual", "neighborhood"
                # Check whether user provided a manual AQI value for testing
                if "manual_aqi" in payload:
                    aqi = float(payload["manual_aqi"])
                    label = f"Manual Input (AQI {aqi:.0f})"
                    station = "User Simulated Entry"
                    # Approximate PM2.5 from entered AQI for display
                    if aqi <= 50:
                        approx_pm = (aqi / 50.0) * 9.0
                    elif aqi <= 100:
                        approx_pm = 9.0 + ((aqi - 50.0) / 50.0) * 26.4
                    elif aqi <= 150:
                        approx_pm = 35.4 + ((aqi - 100.0) / 50.0) * 20.0
                    elif aqi <= 200:
                        approx_pm = 55.4 + ((aqi - 150.0) / 50.0) * 70.0
                    else:
                        approx_pm = 125.4 + ((aqi - 200.0) / 100.0) * 100.0
                    pollutant = f"{approx_pm:.1f} µg/m³"
                    pollutants = {"pm2_5": approx_pm, "o3": None, "no2": None, "pm10": None, "so2": None, "co": None, "owm_aqi_index": None}
                    model_pm25 = approx_pm
                    used_fallback_pollutants = True
                    pm25_source = "manual"
                else:
                    if "query" in payload:
                        lat, lon, label = geocode_address(payload["query"])
                    else:
                        lat = payload["lat"]
                        lon = payload["lon"]
                        label = f"Your Location ({lat:.3f}, {lon:.3f})"

                    # NYC neighborhood context (None outside NYC or if the lookup fails)
                    ctx = nyc_context(lat, lon, zip_hint=payload.get("query"))  # typed ZIPs resolve without a network call

                    try:
                        aqi, station, pollutant, epa_pm25, humidity = fetch_purpleair_aqi(lat, lon)
                        live_pm25 = epa_pm25
                    except Exception as sensor_exc:
                        # No usable live sensor. In NYC, fall back to the neighborhood's long-term PM2.5 average,
                        # clearly flagged as an estimate (this includes a missing PurpleAir key; the reason is printed below).
                        # Outside NYC there is no neighborhood average to fall back on, so keep the original error.
                        if not ctx:
                            raise DataUnavailable(str(sensor_exc)) from sensor_exc
                        print(f"[PurpleAir] {sensor_exc} -> using NYC neighborhood average for {ctx['name']}")
                        epa_pm25 = ctx["pm25_annual_mean"]
                        aqi = pm25_to_aqi(epa_pm25)
                        station = f"{ctx['name']} neighborhood average (no live sensor)"
                        pollutant = f"{epa_pm25:.1f} µg/m³"
                        humidity = None
                        aqi_estimated = True
                        pm25_source = "neighborhood"

                    model_pm25 = epa_pm25  # live sensor value, or the neighborhood average in the NYC fallback

                    try:
                        pollutants = fetch_openweather_pollutants(lat, lon)
                        used_fallback_pollutants = False
                    except Exception:
                        # OWM key not set yet, or request failed. Leave o3/no2 as None (shown as
                        # N/A on the page) so we never imply a measured value we do not have.
                        pollutants = {"pm2_5": epa_pm25, "o3": None, "no2": None, "pm10": None, "so2": None, "co": None, "owm_aqi_index": None}
                        used_fallback_pollutants = True

                    # The PM2.5 tile always shows the value that drives the score (live sensor, or the
                    # NYC neighborhood average in the fallback), not OpenWeatherMap's modeled PM2.5.
                    pollutants["pm2_5"] = epa_pm25

                # NO2, PM2.5, and O3 are passed for display only; risk is driven by AQI + clinical profile
                prob, level, color, label_text, headline, advisory, model_level, meter_prob = compute_detailed_risk(
                    payload["age"],
                    payload["hypertension"],
                    payload["heart"],
                    payload.get("smoke_status", "never"),
                    payload["fam_smoke"],
                    payload["fam_stroke"],
                    aqi,
                    model_pm25,
                )

                resp = {
                    "location_label": label,
                    "aqi": aqi,
                    "station": station,
                    "pollutant": pollutant,
                    "epa_category": get_epa_aqi_category(aqi),
                    "pollutants": pollutants,
                    "pollutants_estimated": used_fallback_pollutants,
                    "probability": prob,
                    "meter_probability": meter_prob,
                    "model_level": model_level,
                    "guardrail_applied": level != model_level,
                    "care_circle_alert": should_alert_care_circle(level, aqi, aqi_estimated),
                    "sensitive_group": is_sensitive_group(payload["age"], payload["heart"]),
                    "sensitive_rule_active": is_sensitive_group(payload["age"], payload["heart"]) and round(aqi) >= SENSITIVE_RULE_MIN_AQI,
                    "pm25_used": model_pm25,
                    "pm25_source": pm25_source,
                    "level": level,
                    "color": color,
                    "label": label_text,
                    "headline": headline,
                    "advisory": advisory,
                    "nyc_context": ctx,
                    "aqi_estimated": aqi_estimated,
                    "live_pm25": live_pm25,
                }
                self.send_response(200)
            except DataUnavailable as exc:
                # Expected: no live air data and no neighborhood-average fallback. Not a crash.
                print(f"[Air data unavailable] {exc}")
                resp = {"error": "PionAir is not working right now.", "kind": "unavailable"}
                self.send_response(503)
            except Exception as exc:
                print("\n" + "=" * 60)
                print("FULL ERROR DETAILS (this is what actually happened):")
                traceback.print_exc()
                print("=" * 60 + "\n")
                resp = {"error": str(exc)}
                self.send_response(400)

            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    _print_startup_diagnostics()
    port = int(os.environ.get("PORT", 8000))
    server = http.server.HTTPServer(("0.0.0.0", port), AppHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Pionair dashboard running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nClosing Pionair.")
        server.server_close()


if __name__ == "__main__":
    main()