import json
import math
import urllib.request

# Literature-derived log-odds weights: beta = ln(Odds Ratio)
MODEL_PARAMS = {
    "intercept": -3.80,          # Baseline log-odds (~2.2% baseline event risk)
    "age_per_decade": 0.65,      # OR ~1.92 per decade above 40 (Framingham / SCORE2)
    "lung": 0.74,                # OR ~2.10 for diagnosed chronic respiratory disease
    "heart": 1.39,               # OR ~4.02 for diagnosed cardiovascular disease
    "smoke": 0.79,               # OR ~2.20 for active smoking status
    "fam_smoke": 0.26,           # OR ~1.30 for household/familial smoking history
    "fam_stroke": 0.35,          # OR ~1.42 for familial cerebrovascular history
    "aqi_per_10": 0.15,          # OR ~1.16 per 10 AQI units (~1.5% event increase per point)
}

LEVEL_THRESHOLDS = [
    (0.50, 5, "red", "Severe Risk: Remain indoors, keep windows closed, and run an air purifier."),
    (0.35, 4, "orange", "High Risk: Avoid outdoor exertion and monitor for shortness of breath."),
    (0.20, 3, "yellow", "Moderate Risk: Sensitive individuals should reduce prolonged exertion."),
    (0.10, 2, "light green", "Low Risk: Generally safe; follow regular preventive care."),
    (0.00, 1, "dark green", "Minimal Risk: Excellent air quality and baseline clinical risk."),
]


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HealthRiskAQI/2.0 (cardiopulmonary-assessment)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_live_local_aqi() -> tuple[float, str, str, str]:
    """Auto-detects location via IP and fetches real-time AQI from WAQI."""
    # 1. Resolve geographic coordinates automatically
    ip_data = http_get_json("https://ipapi.co/json/")
    lat = float(ip_data["latitude"])
    lon = float(ip_data["longitude"])
    city = ip_data.get("city", "Unknown City")
    region = ip_data.get("region", "")
    location_label = f"{city}, {region}"

    # 2. Query WAQI using coordinates to find the nearest physical sensor station
    waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token=9305a2962136964338e3ccf505dd1d7f5872dec9"
    waqi_data = http_get_json(waqi_url)

    if waqi_data.get("status") != "ok":
        raise RuntimeError("Could not retrieve station data from AQI provider.")

    station_payload = waqi_data["data"]
    aqi_val = float(station_payload["aqi"])
    station_name = station_payload.get("city", {}).get("name", "Nearest Sensor Station")
    dominant_pollutant = station_payload.get("dominentpol", "pm25").upper()

    return aqi_val, location_label, station_name, dominant_pollutant


def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def ask_int(prompt: str, min_val: int = 1, max_val: int = 120) -> int:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"Please enter an integer between {min_val} and {max_val}.")


def compute_logistic_risk(
    age: int,
    lung: bool,
    heart: bool,
    smoke: bool,
    fam_smoke: bool,
    fam_stroke: bool,
    aqi: float,
) -> tuple[float, int, str, str]:
    # Calculate logit: z = beta_0 + sum(beta_i * x_i)
    logit = MODEL_PARAMS["intercept"]
    logit += max(0.0, (age - 40) / 10.0) * MODEL_PARAMS["age_per_decade"]
    if lung:
        logit += MODEL_PARAMS["lung"]
    if heart:
        logit += MODEL_PARAMS["heart"]
    if smoke:
        logit += MODEL_PARAMS["smoke"]
    if fam_smoke:
        logit += MODEL_PARAMS["fam_smoke"]
    if fam_stroke:
        logit += MODEL_PARAMS["fam_stroke"]
    logit += (aqi / 10.0) * MODEL_PARAMS["aqi_per_10"]

    # Logistic sigmoid function
    prob = 1.0 / (1.0 + math.exp(-logit))

    for threshold, level, color, advisory in LEVEL_THRESHOLDS:
        if prob >= threshold:
            return prob, level, color, advisory

    return prob, 1, "dark green", "Minimal Risk: Ideal conditions."


def main():
    print("=" * 60)
    print("   AUTOMATED HEALTH & ENVIRONMENTAL RISK EVALUATION   ")
    print("=" * 60)

    age = ask_int("Patient Age", min_val=1, max_val=120)
    lung = ask_yes_no("Diagnosed chronic lung condition (Asthma/COPD)?")
    heart = ask_yes_no("Diagnosed cardiovascular condition?")
    smoke = ask_yes_no("Active cigarette or tobacco smoker?")
    fam_smoke = ask_yes_no("Significant family history of smoking?")
    fam_stroke = ask_yes_no("Family history of stroke / early heart disease?")

    print("\nDetecting local environment...")
    try:
        aqi, loc_label, station, pollutant = get_live_local_aqi()
        print(f"-> Detected Location: {loc_label}")
        print(f"-> Monitoring Station: {station}")
        print(f"-> Live Ground-Level AQI: {aqi:.0f} (Dominant pollutant: {pollutant})")
    except Exception as err:
        print(f"-> Location detection fallback: using baseline ambient standard (AQI = 50.0). Reason: {err}")
        aqi = 50.0

    prob, level, color, advisory = compute_logistic_risk(
        age, lung, heart, smoke, fam_smoke, fam_stroke, aqi
    )

    print("\n" + "=" * 60)
    print(f"Estimated Event Probability: {prob * 100:.1f}%")
    print(f"Stratified Risk Category: Level {level} [{color.upper()}]")
    print(f"Action Advisory: {advisory}")
    print("=" * 60)


if __name__ == "__main__":
    main()