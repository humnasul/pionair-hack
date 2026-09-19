import http.server
import json
import math
import os
import urllib.parse
import urllib.request
import webbrowser

PURPLEAIR_API_READ_KEY = os.getenv("PURPLEAIR_API_KEY", "2A871D1F-B466-11F1-9E30-4201AC1DC129")

MODEL_PARAMS = {
    "intercept": -3.80,          # Baseline log-odds (~2.2% baseline event risk)
    "age_per_decade": 0.65,      # OR ~1.92 per decade above 40
    "lung": 0.74,                # OR ~2.10 for chronic respiratory disease
    "heart": 1.39,               # OR ~4.02 for diagnosed cardiovascular condition
    "smoke": 0.79,               # OR ~2.20 for active smoking status
    "fam_smoke": 0.26,           # OR ~1.30 for family tobacco exposure
    "fam_stroke": 0.35,          # OR ~1.42 for family stroke history
    "aqi_per_10": 0.15,          # OR ~1.16 per 10 AQI units (~1.5% event increase per point)
}

LEVEL_THRESHOLDS = [
    (0.50, 5, "#dc2626", "Severe Risk", "Remain indoors, keep windows tightly closed, and operate a certified HEPA air purifier. Seek immediate medical attention if experiencing chest tightness or shortness of breath."),
    (0.35, 4, "#ea580c", "High Risk", "Avoid all strenuous outdoor exertion. Individuals with cardiopulmonary conditions should stay in clean indoor air spaces and monitor respiratory symptoms."),
    (0.20, 3, "#ca8a04", "Moderate Risk", "Sensitive individuals (asthma, COPD, prior cardiac history) should limit prolonged outdoor exertion and keep quick-relief medication accessible."),
    (0.10, 2, "#16a34a", "Low Risk", "Air quality and baseline profile are generally safe. Sensitive individuals should continue observing normal preventive routines."),
    (0.00, 1, "#059669", "Minimal Risk", "Ideal environmental and physiological baseline. Normal indoor and outdoor activities are fully encouraged."),
]


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
        return {"category": "Good", "color": "#16a34a", "bg": "#dcfce7", "text": "Air quality is satisfactory and poses little to no health risk."}
    elif aqi <= 100:
        return {"category": "Moderate", "color": "#ca8a04", "bg": "#fef9c3", "text": "Acceptable air quality; unusually sensitive people may experience minor irritation."}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "color": "#ea580c", "bg": "#ffedd5", "text": "Individuals with heart or lung disease, older adults, and children are at greater risk."}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#dc2626", "bg": "#fee2e2", "text": "Some members of the general public may experience health effects; sensitive groups more serious effects."}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "color": "#9333ea", "bg": "#f3e8ff", "text": "Health alert: risk of health effects increased for everyone in the area."}
    else:
        return {"category": "Hazardous", "color": "#7f1d1d", "bg": "#fecaca", "text": "Health warning of emergency conditions: everyone is more likely to be affected."}


def geocode_address(query: str) -> tuple[float, float, str]:
    encoded = urllib.parse.quote(query.strip())
    url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1&countrycodes=us"
    req = urllib.request.Request(url, headers={"User-Agent": "HealthRiskAQI/5.0 (cardio-eval)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if not data:
        fallback_url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
        req_fallback = urllib.request.Request(fallback_url, headers={"User-Agent": "HealthRiskAQI/5.0 (cardio-eval)"})
        with urllib.request.urlopen(req_fallback, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            raise ValueError(f"Could not locate '{query}'. Try including city or state (e.g. '10044, NY').")

    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


def fetch_purpleair_aqi(lat: float, lon: float) -> tuple[float, str, str, float, float]:
    if PURPLEAIR_API_READ_KEY == "YOUR_PURPLEAIR_READ_KEY_HERE":
        raise ValueError("PurpleAir API Read Key is not set. Get one at develop.purpleair.com.")

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
        raise RuntimeError("No active outdoor PurpleAir monitors found within ~5km of this point.")

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


def compute_detailed_risk(
    age: int, lung: bool, heart: bool, smoke: bool, fam_smoke: bool, fam_stroke: bool, aqi: float
) -> tuple[float, int, str, str, str, list]:
    age_contribution = max(0.0, (age - 40) / 10.0) * MODEL_PARAMS["age_per_decade"]
    lung_contribution = MODEL_PARAMS["lung"] if lung else 0.0
    heart_contribution = MODEL_PARAMS["heart"] if heart else 0.0
    smoke_contribution = MODEL_PARAMS["smoke"] if smoke else 0.0
    fam_smoke_contribution = MODEL_PARAMS["fam_smoke"] if fam_smoke else 0.0
    fam_stroke_contribution = MODEL_PARAMS["fam_stroke"] if fam_stroke else 0.0
    aqi_contribution = (aqi / 10.0) * MODEL_PARAMS["aqi_per_10"]

    logit = (
        MODEL_PARAMS["intercept"]
        + age_contribution
        + lung_contribution
        + heart_contribution
        + smoke_contribution
        + fam_smoke_contribution
        + fam_stroke_contribution
        + aqi_contribution
    )

    prob = 1.0 / (1.0 + math.exp(-logit))

    breakdown = [
        {"factor": "Baseline Calibrated Risk", "score": round(MODEL_PARAMS["intercept"], 2), "type": "baseline"},
        {"factor": f"Age Exposure ({age} yrs)", "score": round(age_contribution, 2), "active": age > 40},
        {"factor": "Cardiovascular Diagnosis", "score": round(heart_contribution, 2), "active": heart},
        {"factor": "Chronic Lung Condition", "score": round(lung_contribution, 2), "active": lung},
        {"factor": "Active Smoker Status", "score": round(smoke_contribution, 2), "active": smoke},
        {"factor": "Family Stroke / Heart Disease", "score": round(fam_stroke_contribution, 2), "active": fam_stroke},
        {"factor": "Family Smoking Exposure", "score": round(fam_smoke_contribution, 2), "active": fam_smoke},
        {"factor": f"Environmental AQI ({aqi:.0f})", "score": round(aqi_contribution, 2), "active": True},
    ]

    for threshold, level, color, label, advisory in LEVEL_THRESHOLDS:
        if prob >= threshold:
            return prob, level, color, label, advisory, breakdown

    return prob, 1, "#059669", "Minimal Risk", LEVEL_THRESHOLDS[-1][4], breakdown


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cardiopulmonary & Environmental Risk Evaluator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --slate-50: #f8fafc;
      --slate-100: #f1f5f9;
      --slate-200: #e2e8f0;
      --slate-600: #475569;
      --slate-800: #1e293b;
      --blue-600: #2563eb;
      --blue-700: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      max-width: 680px;
      margin: 32px auto;
      padding: 0 16px;
      color: var(--slate-800);
      background: var(--slate-50);
      line-height: 1.5;
    }
    .card {
      background: #ffffff;
      padding: 28px 32px;
      border-radius: 16px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
      border: 1px solid var(--slate-200);
    }
    h2 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; }
    .subtitle { color: var(--slate-600); font-size: 14px; margin-bottom: 22px; }
    
    .form-group { margin-bottom: 16px; }
    label.field-label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 13.5px; }
    input[type="number"], input[type="text"] {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      font-size: 14px;
      transition: border-color 0.15s;
    }
    input[type="number"]:focus, input[type="text"]:focus {
      outline: none;
      border-color: var(--blue-600);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    .checkbox-tile-group {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }
    @media (max-width: 540px) { .checkbox-tile-group { grid-template-columns: 1fr; } }
    .checkbox-tile {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
      background: #fafafa;
    }
    .checkbox-tile:hover { background: #f4f6f8; }
    .checkbox-tile input { margin-top: 2px; }

    .toggle-row { display: flex; gap: 14px; margin-bottom: 10px; font-size: 13px; }
    .toggle-row label { display: flex; align-items: center; gap: 5px; cursor: pointer; }

    button {
      width: 100%;
      padding: 13px;
      background: var(--blue-600);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 15px;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover { background: var(--blue-700); }
    button:disabled { background: #94a3b8; cursor: not-allowed; }

    #status { margin-top: 14px; font-size: 13px; color: var(--slate-600); text-align: center; }

    /* Results Dashboard */
    #result {
      margin-top: 28px;
      display: none;
      animation: fadeIn 0.25s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .dashboard-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--slate-200);
      margin-bottom: 18px;
    }
    .station-info { font-size: 12.5px; color: var(--slate-600); }
    .station-info strong { color: var(--slate-800); }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 18px;
    }
    .metric-card {
      padding: 16px;
      border-radius: 10px;
      border: 1px solid var(--slate-200);
      background: #ffffff;
    }
    .metric-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--slate-600); margin-bottom: 4px; }
    .metric-val { font-size: 32px; font-weight: 800; line-height: 1; }
    .metric-sub { font-size: 12px; margin-top: 6px; }

    /* Gauge / Thermometer Bar */
    .gauge-wrapper { margin: 18px 0; }
    .gauge-bar {
      height: 12px;
      border-radius: 6px;
      background: linear-gradient(to right, #059669 0%, #16a34a 20%, #ca8a04 40%, #ea580c 70%, #dc2626 100%);
      position: relative;
    }
    .gauge-pointer {
      position: absolute;
      top: -4px;
      width: 20px;
      height: 20px;
      background: #ffffff;
      border: 3px solid var(--slate-800);
      border-radius: 50%;
      transform: translateX(-50%);
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-ticks {
      display: flex;
      justify-content: space-between;
      font-size: 10.5px;
      color: var(--slate-600);
      margin-top: 6px;
    }

    /* Advisory Box */
    .advisory-box {
      padding: 14px 16px;
      border-radius: 10px;
      margin-bottom: 20px;
      font-size: 13.5px;
      border-left: 4px solid;
    }

    /* Risk Breakdown Bars */
    .breakdown-section { margin-top: 20px; }
    .breakdown-title { font-size: 13px; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.03em; color: var(--slate-600); }
    .breakdown-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12.5px;
      padding: 6px 0;
      border-bottom: 1px dotted var(--slate-200);
    }
    .breakdown-bar-wrap {
      flex: 1;
      max-width: 140px;
      height: 6px;
      background: var(--slate-100);
      border-radius: 3px;
      margin: 0 12px;
      overflow: hidden;
    }
    .breakdown-fill { height: 100%; border-radius: 3px; }
  </style>
</head>
<body>
<div class="card">
  <h2>Environmental Risk Assessment</h2>
  <div class="subtitle">Logistic regression event forecasting linked to live PurpleAir ground sensors</div>

  <div class="form-group">
    <label class="field-label" for="age">Patient Age</label>
    <input type="number" id="age" value="48" min="1" max="120">
  </div>

  <label class="field-label">Cardiopulmonary & Clinical Profile</label>
  <div class="checkbox-tile-group">
    <label class="checkbox-tile">
      <input type="checkbox" id="heart">
      <span>Diagnosed Cardiovascular Condition</span>
    </label>
    <label class="checkbox-tile">
      <input type="checkbox" id="lung">
      <span>Chronic Respiratory / Asthma / COPD</span>
    </label>
    <label class="checkbox-tile">
      <input type="checkbox" id="smoke">
      <span>Active Smoker</span>
    </label>
    <label class="checkbox-tile">
      <input type="checkbox" id="fam_stroke">
      <span>Family History of Stroke / Cardiac Event</span>
    </label>
    <label class="checkbox-tile" style="grid-column: span 2;">
      <input type="checkbox" id="fam_smoke">
      <span>Household / Familial Tobacco Smoke Exposure</span>
    </label>
  </div>

  <div class="form-group">
    <label class="field-label">Sensor Location Query</label>
    <div class="toggle-row">
      <label><input type="radio" name="loc_mode" value="address" checked onclick="toggleMode()"> US Zip / Neighborhood</label>
      <label><input type="radio" name="loc_mode" value="device" onclick="toggleMode()"> Precise Device GPS</label>
    </div>
    <div id="address-box">
      <input type="text" id="address_input" placeholder="e.g. 10044 or Roosevelt Island, NY" value="10044">
    </div>
  </div>

  <button id="submit-btn" onclick="evaluateRisk()">Compute Acute Risk Analysis</button>
  <div id="status"></div>

  <div id="result">
    <div class="dashboard-header">
      <div class="station-info">
        <div><strong>Location:</strong> <span id="res-location"></span></div>
        <div><strong>Station:</strong> <span id="res-station"></span></div>
      </div>
      <div id="res-badge" style="padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 12px;"></div>
    </div>

    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Calculated Event Probability</div>
        <div class="metric-val" id="res-prob">--%</div>
        <div class="metric-sub" id="res-tier-name">Level 1</div>
      </div>
      <div class="metric-card" id="res-aqi-card">
        <div class="metric-label">PurpleAir Ground AQI</div>
        <div class="metric-val" id="res-aqi">--</div>
        <div class="metric-sub" id="res-pollutant">PM2.5: --</div>
      </div>
    </div>

    <div class="gauge-wrapper">
      <div class="gauge-bar">
        <div class="gauge-pointer" id="gauge-marker" style="left: 0%;"></div>
      </div>
      <div class="gauge-ticks">
        <span>0% (Minimal)</span>
        <span>20% (Mod)</span>
        <span>35% (High)</span>
        <span>50%+ (Severe)</span>
      </div>
    </div>

    <div class="advisory-box" id="res-advisory-box">
      <strong id="res-advisory-title">Clinical & Behavioral Advisory:</strong>
      <div id="res-advisory-text" style="margin-top: 4px;"></div>
    </div>

    <div class="breakdown-section">
      <div class="breakdown-title">Odds Contribution By Factor (Log-Odds)</div>
      <div id="breakdown-container"></div>
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

  // Probability display
  const pct = (data.probability * 100).toFixed(1);
  const probEl = document.getElementById("res-prob");
  probEl.textContent = pct + "%";
  probEl.style.color = data.color;

  const tierEl = document.getElementById("res-tier-name");
  tierEl.textContent = `${data.tier_label} (Level ${data.level})`;
  tierEl.style.color = data.color;

  // AQI display
  const aqiCard = document.getElementById("res-aqi-card");
  document.getElementById("res-aqi").textContent = Math.round(data.aqi);
  document.getElementById("res-aqi").style.color = data.epa_category.color;
  document.getElementById("res-pollutant").textContent = `${data.pollutant} • ${data.epa_category.category}`;

  // Badge
  const badge = document.getElementById("res-badge");
  badge.textContent = `Level ${data.level} - ${data.tier_label}`;
  badge.style.background = data.epa_category.bg;
  badge.style.color = data.epa_category.color;

  // Gauge pointer (clamp between 2% and 98%)
  const gaugeMarker = document.getElementById("gauge-marker");
  const markerPos = Math.min(98, Math.max(2, (data.probability / 0.60) * 100));
  gaugeMarker.style.left = markerPos + "%";
  gaugeMarker.style.borderColor = data.color;

  // Advisory Box
  const advBox = document.getElementById("res-advisory-box");
  advBox.style.background = data.epa_category.bg;
  advBox.style.borderColor = data.color;
  document.getElementById("res-advisory-title").textContent = `${data.tier_label} Advisory:`;
  document.getElementById("res-advisory-text").textContent = data.advisory;

  // Breakdown Render
  const bContainer = document.getElementById("breakdown-container");
  bContainer.innerHTML = "";
  data.breakdown.forEach(item => {
    const row = document.createElement("div");
    row.className = "breakdown-row";

    const label = document.createElement("span");
    label.textContent = item.factor;
    if (item.active === false) label.style.color = "#94a3b8";

    const barWrap = document.createElement("div");
    barWrap.className = "breakdown-bar-wrap";

    const fill = document.createElement("div");
    fill.className = "breakdown-fill";
    const mag = Math.min(100, Math.abs(item.score) * 45);
    fill.style.width = mag + "%";
    fill.style.background = item.score >= 0 ? data.color : "#64748b";
    barWrap.appendChild(fill);

    const val = document.createElement("span");
    val.style.fontWeight = "600";
    val.style.minWidth = "40px";
    val.style.textAlign = "right";
    val.textContent = (item.score >= 0 ? "+" : "") + item.score.toFixed(2);
    if (item.active === false) val.style.color = "#94a3b8";

    row.appendChild(label);
    row.appendChild(barWrap);
    row.appendChild(val);
    bContainer.appendChild(row);
  });
}

async function evaluateRisk() {
  const btn = document.getElementById("submit-btn");
  const status = document.getElementById("status");
  const mode = document.querySelector('input[name="loc_mode"]:checked').value;

  btn.disabled = true;
  status.textContent = "Querying live PurpleAir sensor matrix...";

  const basePayload = {
    age: parseInt(document.getElementById("age").value, 10),
    lung: document.getElementById("lung").checked,
    heart: document.getElementById("heart").checked,
    smoke: document.getElementById("smoke").checked,
    fam_smoke: document.getElementById("fam_smoke").checked,
    fam_stroke: document.getElementById("fam_stroke").checked
  };

  try {
    let payload = { ...basePayload };
    if (mode === "address") {
      const q = document.getElementById("address_input").value.trim();
      if (!q) throw new Error("Please enter a zip code or address.");
      payload.query = q;
    } else {
      status.textContent = "Acquiring high-precision device coordinates...";
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
    status.textContent = "Evaluation Error: " + err.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


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
                    label = f"Coordinates ({lat:.4f}, {lon:.4f})"

                aqi, station, pollutant, epa_pm25, humidity = fetch_purpleair_aqi(lat, lon)
                prob, level, color, tier_label, advisory, breakdown = compute_detailed_risk(
                    payload["age"],
                    payload["lung"],
                    payload["heart"],
                    payload["smoke"],
                    payload["fam_smoke"],
                    payload["fam_stroke"],
                    aqi,
                )

                resp = {
                    "location_label": label,
                    "aqi": aqi,
                    "station": station,
                    "pollutant": f"PM2.5: {pollutant} (RH: {humidity:.0f}%)",
                    "epa_category": get_epa_aqi_category(aqi),
                    "probability": prob,
                    "level": level,
                    "color": color,
                    "tier_label": tier_label,
                    "advisory": advisory,
                    "breakdown": breakdown,
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
    port = 8000
    server = http.server.HTTPServer(("127.0.0.1", port), AppHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Visual dashboard running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()