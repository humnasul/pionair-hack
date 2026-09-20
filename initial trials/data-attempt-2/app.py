import http.server
import json
import math
import os
import urllib.parse
import urllib.request
import webbrowser

MODEL_JSON_PATH = "trained_model.json"
PURPLEAIR_API_READ_KEY = os.getenv("PURPLEAIR_API_KEY", "2A871D1F-B466-11F1-9E30-4201AC1DC129")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "6e73f0faa5cbfbef42b0b8789b8c52a1")

# ------------------------------------------------------------------------------
# 1. MODEL COEFFICIENT LOADER
# ------------------------------------------------------------------------------
if os.path.exists(MODEL_JSON_PATH):
    with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
        _loaded = json.load(f)
        INTERCEPT = _loaded.get("intercept", -3.85)
        COEFS = _loaded.get("coefficients", {})
else:
    # Fallback used only if trained_model.json is missing. These are the same
    # literature-derived / learned values as trained_model.json, already
    # passed through the same 0.65 shrinkage factor used in train_datasets.py,
    # so behavior stays consistent whether or not the JSON file is present.
    INTERCEPT = -3.85
    COEFS = {
        "age_over_40": 0.45,
        "heart": 0.39,
        "hypertension": 0.21,
        "smoke": 0.20,
        "fam_stroke": 0.22,
        "fam_smoke": 0.13,
        "pm2_5_scaled": 0.08,
        "o3_scaled": 0.13,
        "no2_scaled": 0.07,
    }

LEVEL_THRESHOLDS = [
    (0.50, 5, "#ef4444", "Take Immediate Shelter", "🚨 Red Flag Alert", "Your body is under major strain from the air quality outside. Close all windows, turn on an air filter if you have one, and rest indoors. If you feel dizzy, short of breath, or notice sudden numbness, contact a doctor or dial 911 right away."),
    (0.35, 4, "#f97316", "Stay Indoors & Rest", "⚠️ High Strain Zone", "Skip outdoor exercise or heavy chores today. Keep inside with clean air, drink extra water, and make sure your regular prescriptions or inhalers are within reach."),
    (0.20, 3, "#f59e0b", "Pace Yourself", "👀 Yellow Caution", "If you have heart or blood pressure concerns, take things slow outside. Avoid long walks along busy roads while the air is hazy or dusty."),
    (0.10, 2, "#10b981", "Looking Safe", "👍 Green Light", "Your body and the outdoor air are in good harmony today. You are free to run errands, play, and go on walks."),
    (0.00, 1, "#06b6d4", "Prime Conditions", "🌟 Clear Sailing", "Air is crisp and your personal baseline is calm. Enjoy the outdoors, open up the windows, and stay active!"),
]


# ------------------------------------------------------------------------------
# 2. PURPLEAIR SENSOR INTEGRATION & EPA CALIBRATION
# ------------------------------------------------------------------------------
def pm25_to_aqi(pm: float) -> float:
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.4, 301, 500),
    ]
    if pm <= 0:
        return 0.0
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low)
    return 500.0


def get_epa_aqi_category(aqi: float) -> dict:
    if aqi <= 50:
        return {"category": "Fresh & Clean", "color": "#10b981", "bg": "#ecfdf5", "simple_desc": "Clean and fresh air. Perfect for outdoor fun."}
    elif aqi <= 100:
        return {"category": "Fair / Moderate", "color": "#f59e0b", "bg": "#fffbeb", "simple_desc": "Generally fine, though sensitive lungs might feel a tickle."}
    elif aqi <= 150:
        return {"category": "Smoky / Hazy", "color": "#f97316", "bg": "#fff7ed", "simple_desc": "Fine dust particles are elevated. Vulnerable groups should limit time outside."}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#ef4444", "bg": "#fef2f2", "simple_desc": "Dirty air that can cause coughing or wheezing. Stay inside."}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "color": "#8b5cf6", "bg": "#f5f3ff", "simple_desc": "Heavy smoke or smog pollution. Seal windows and avoid all outdoor trips."}
    else:
        return {"category": "Hazardous", "color": "#991b1b", "bg": "#fef2f2", "simple_desc": "Emergency smoke conditions. Run indoor filtration continuously."}


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


def fetch_purpleair_aqi(lat: float, lon: float) -> tuple[float, str, str, float, float]:
    if not PURPLEAIR_API_READ_KEY:
        raise ValueError("PurpleAir API key is missing. Set the PURPLEAIR_API_KEY environment variable.")

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

    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    fields = payload.get("fields", [])
    data_rows = payload.get("data", [])
    if not data_rows:
        raise RuntimeError("No working neighborhood air sensors found close to this spot.")

    idx = {f: i for i, f in enumerate(fields)}
    closest_row = None
    min_dist = float("inf")

    for row in data_rows:
        s_lat = row[idx["latitude"]]
        s_lon = row[idx["longitude"]]
        dist = math.hypot(s_lat - lat, s_lon - lon)
        if dist < min_dist:
            min_dist = dist
            closest_row = row

    sensor_name = closest_row[idx["name"]]
    raw_pm25 = float(closest_row[idx["pm2.5_cf_1"]] or 0.0)
    humidity = float(closest_row[idx["humidity"]] or 50.0)

    epa_pm25 = max(0.0, 0.524 * raw_pm25 - 0.0862 * humidity + 5.75)
    aqi = pm25_to_aqi(epa_pm25)
    pollutant_detail = f"{epa_pm25:.1f} µg/m³"

    return aqi, sensor_name, pollutant_detail, epa_pm25, humidity


def fetch_openweather_pollutants(lat: float, lon: float) -> dict:
    """
    Live PM2.5, ozone (O3), and NO2 from OpenWeatherMap's Air Pollution API.
    Free tier: https://openweathermap.org/api/air-pollution
    Units are µg/m3 for all pollutants. Note: OWM's own 1-5 "aqi" index uses
    a different scale than the US EPA 0-500 AQI used elsewhere in this file
    (pm25_to_aqi / get_epa_aqi_category) - don't mix them without conversion.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap API key is missing. Set the OPENWEATHER_API_KEY environment variable.")

    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)

    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if not payload.get("list"):
        raise RuntimeError("No air pollution data returned for this location.")

    components = payload["list"][0]["components"]
    return {
        "pm2_5": components.get("pm2_5"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "pm10": components.get("pm10"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "owm_aqi_index": payload["list"][0]["main"]["aqi"],  # 1 (Good) - 5 (Very Poor), OWM's own scale
    }


# ------------------------------------------------------------------------------
# 3. INTERACTIVE LOGISTIC REGRESSION SCORING
# ------------------------------------------------------------------------------
def compute_detailed_risk(
    age: int, hypertension: bool, heart: bool, smoke: bool, fam_smoke: bool, fam_stroke: bool,
    pm2_5: float, o3: float, no2: float
) -> tuple[float, int, str, str, str, str]:
    age_over_40 = max(0.0, (age - 40) / 10.0)
    pm2_5_scaled = pm2_5 / 10.0
    o3_scaled = o3 / 10.0
    no2_scaled = no2 / 10.0
    h_val = 1.0 if heart else 0.0
    ht_val = 1.0 if hypertension else 0.0
    s_val = 1.0 if smoke else 0.0
    f_stroke_val = 1.0 if fam_stroke else 0.0
    f_smoke_val = 1.0 if fam_smoke else 0.0

    logit = (
        INTERCEPT
        + COEFS.get("age_over_40", 0.45) * age_over_40
        + COEFS.get("heart", 0.39) * h_val
        + COEFS.get("hypertension", 0.21) * ht_val
        + COEFS.get("smoke", 0.20) * s_val
        + COEFS.get("fam_stroke", 0.22) * f_stroke_val
        + COEFS.get("fam_smoke", 0.13) * f_smoke_val
        + COEFS.get("pm2_5_scaled", 0.08) * pm2_5_scaled
        + COEFS.get("o3_scaled", 0.13) * o3_scaled
        + COEFS.get("no2_scaled", 0.07) * no2_scaled
    )

    prob = 1.0 / (1.0 + math.exp(-logit))

    for threshold, level, color, label, headline, advisory in LEVEL_THRESHOLDS:
        if prob >= threshold:
            return prob, level, color, label, headline, advisory

    return prob, 1, "#06b6d4", "Prime Conditions", LEVEL_THRESHOLDS[-1][4], LEVEL_THRESHOLDS[-1][5]


# ------------------------------------------------------------------------------
# 4. DASHBOARD FRONTEND TEMPLATE (Fun, Colorful, Ultra-Clear)
# ------------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AirBuddy • Neighborhood Health & Air Checker</title>
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
      font-size: 38px;
      font-weight: 900;
      letter-spacing: -0.03em;
      line-height: 1.1;
      margin-bottom: 8px;
    }
    .hero-stat-sub {
      font-size: 16px;
      font-weight: 700;
    }

    .pair-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 20px;
    }
    @media (max-width: 500px) { .pair-grid { grid-template-columns: 1fr; } }
    .mini-card {
      background: #ffffff;
      border: 2px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      text-align: center;
    }
    .mini-card-label {
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .mini-card-num {
      font-size: 30px;
      font-weight: 900;
    }
    .mini-card-desc {
      font-size: 13.5px;
      font-weight: 700;
      margin-top: 2px;
    }

    /* Visual Thermometer Bar */
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
      background: linear-gradient(to right, #06b6d4 0%, #10b981 25%, #f59e0b 50%, #f97316 75%, #ef4444 100%);
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
      box-shadow: 0 4px 10px rgba(0,0,0,0.25);
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

    /* Advice Box */
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

    /* Educational Card */
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
  </style>
</head>
<body>
<div class="card">
  <div class="header-pill">🌤️ Local Air + Body Shield</div>
  <h1>AirBuddy Health Checker</h1>
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
        <input type="checkbox" id="smoke">
        <span>🚬 I Smoke Tobacco Regularly</span>
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
    <label class="field-label">3. Where are you?</label>
    <div class="toggle-row">
      <label><input type="radio" name="loc_mode" value="address" checked onclick="toggleMode()"> Enter City or Zip</label>
      <label><input type="radio" name="loc_mode" value="device" onclick="toggleMode()"> Phone GPS</label>
    </div>
    <div id="address-box">
      <input type="text" id="address_input" placeholder="e.g. 10044 or Roosevelt Island, NY" value="10044">
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
      <div class="hero-stat-sub" id="hero-prob">--</div>
    </div>

    <!-- Details Grid -->
    <div class="pair-grid">
      <div class="mini-card" id="air-card">
        <div class="mini-card-label">Outdoor Air Score</div>
        <div class="mini-card-num" id="air-num">--</div>
        <div class="mini-card-desc" id="air-desc">--</div>
      </div>
      <div class="mini-card">
        <div class="mini-card-label">Air Particulates (PM2.5)</div>
        <div class="mini-card-num" id="pm-num" style="color: #6366f1;">--</div>
        <div class="mini-card-desc">Fine Smoke & Dust</div>
      </div>
    </div>

    <!-- Fun Thermometer Meter -->
    <div class="meter-box">
      <div class="meter-label-row">
        <span>Overall Safety Thermometer</span>
        <span id="meter-text-val">Checking...</span>
      </div>
      <div class="meter-track">
        <div class="meter-pin" id="meter-pin" style="left: 10%;"></div>
      </div>
      <div class="meter-steps">
        <span>Chill</span>
        <span>Mild</span>
        <span>Moderate</span>
        <span>Elevated</span>
        <span>High</span>
      </div>
    </div>

    <!-- Plain Language Advice -->
    <div class="advisory-bubble" id="advisory-box">
      <div class="advisory-headline" id="advisory-title">What this means for you:</div>
      <div class="advisory-body" id="advisory-text">--</div>
    </div>

    <!-- How This Works & What It Means -->
    <div class="explainer-card">
      <div class="explainer-title">💡 How does this work?</div>
      <p class="explainer-p">
        <strong>1. We check outdoor sensors live:</strong> We grab the latest readings from real, neighborhood PurpleAir monitors outside your window right now.
      </p>
      <p class="explainer-p">
        <strong>2. We connect your personal picture:</strong> When air is smoky, tiny particles enter your lungs and bloodstream. If your heart or blood vessels are already working extra hard, dirty air can trigger shortness of breath, blood pressure spikes, or chest tightness.
      </p>
      <p class="explainer-p">
        <strong>3. What your score means:</strong> A lower score means you can enjoy the outdoors with zero worries! A higher score is a gentle heads-up to take it easy, close the windows, and avoid strenuous exercise until the air clears up.
      </p>
    </div>
  </div>
</div>

<script>
function toggleMode() {
  const isAddress = document.querySelector('input[name="loc_mode"]:checked').value === 'address';
  document.getElementById('address-box').style.display = isAddress ? 'block' : 'none';
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
  document.getElementById("hero-prob").textContent = `${(data.probability * 100).toFixed(0)}% Environmental Strain Index`;
  document.getElementById("hero-prob").style.color = data.color;

  // Air Card
  const airCard = document.getElementById("air-card");
  airCard.style.borderColor = data.epa_category.color;
  document.getElementById("air-num").textContent = Math.round(data.aqi);
  document.getElementById("air-num").style.color = data.epa_category.color;
  document.getElementById("air-desc").textContent = data.epa_category.category;

  // PM2.5 Card
  document.getElementById("pm-num").textContent = data.pollutant;

  // Thermometer Pin (clamped 4% to 96%)
  const pin = document.getElementById("meter-pin");
  const pinPos = Math.min(96, Math.max(4, (data.probability / 0.55) * 100));
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
    smoke: document.getElementById("smoke").checked,
    fam_smoke: document.getElementById("fam_smoke").checked,
    fam_stroke: document.getElementById("fam_stroke").checked
  };

  try {
    let payload = { ...basePayload };
    if (mode === "address") {
      const q = document.getElementById("address_input").value.trim();
      if (!q) throw new Error("Please enter a zip code or city name.");
      payload.query = q;
    } else {
      status.textContent = "Getting phone GPS coordinates...";
      const pos = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: true, timeout: 10000 });
      });
      payload.lat = pos.coords.latitude;
      payload.lon = pos.coords.longitude;
    }

    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
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
                if "query" in payload:
                    lat, lon, label = geocode_address(payload["query"])
                else:
                    lat = payload["lat"]
                    lon = payload["lon"]
                    label = f"Your Location ({lat:.3f}, {lon:.3f})"

                aqi, station, pollutant, epa_pm25, humidity = fetch_purpleair_aqi(lat, lon)

                try:
                    pollutants = fetch_openweather_pollutants(lat, lon)
                    used_fallback_pollutants = False
                except Exception:
                    # OWM key not set yet, or request failed. Fall back to the
                    # PurpleAir PM2.5 estimate for pm2_5 and treat o3/no2 as
                    # 0 (i.e. "unknown, don't let them affect the score") so
                    # the model still runs rather than crashing the request.
                    pollutants = {"pm2_5": epa_pm25, "o3": 0.0, "no2": 0.0, "pm10": None, "so2": None, "co": None, "owm_aqi_index": None}
                    used_fallback_pollutants = True

                prob, level, color, label_text, headline, advisory = compute_detailed_risk(
                    payload["age"],
                    payload["hypertension"],
                    payload["heart"],
                    payload["smoke"],
                    payload["fam_smoke"],
                    payload["fam_stroke"],
                    pollutants["pm2_5"] or 0.0,
                    pollutants["o3"] or 0.0,
                    pollutants["no2"] or 0.0,
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
                    "level": level,
                    "color": color,
                    "label": label_text,
                    "headline": headline,
                    "advisory": advisory,
                }
                self.send_response(200)
            except Exception as exc:
                resp = {"error": str(exc)}
                self.send_response(400)

            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = int(os.environ.get("PORT", 8000))
    server = http.server.HTTPServer(("0.0.0.0", port), AppHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"AirBuddy dashboard running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nClosing AirBuddy.")
        server.server_close()


if __name__ == "__main__":
    main()