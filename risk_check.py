import math

# Literature-derived log-odds weights (see conversation for sources)
WEIGHTS = {
    "intercept": -3.2,
    "age_per_decade": 0.69,
    "lung": 0.18,
    "heart": 1.61,
    "smoke": 0.55,
    "fam_smoke": 0.30,
    "fam_stroke": 0.35,
    "aqi_per_10": 0.12,
}

COLORS = {
    1: "dark green",
    2: "light green",
    3: "yellow",
    4: "orange",
    5: "red",
}


def ask_yes_no(prompt):
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def ask_int(prompt, min_val=0, max_val=200):
    while True:
        raw = input(prompt + ": ").strip()
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"Please enter a whole number between {min_val} and {max_val}.")


def compute_level(age, lung, heart, smoke, fam_smoke, fam_stroke, aqi):
    logit = WEIGHTS["intercept"]
    logit += max(0, (age - 50) / 10) * WEIGHTS["age_per_decade"]
    if lung:
        logit += WEIGHTS["lung"]
    if heart:
        logit += WEIGHTS["heart"]
    if smoke:
        logit += WEIGHTS["smoke"]
    if fam_smoke:
        logit += WEIGHTS["fam_smoke"]
    if fam_stroke:
        logit += WEIGHTS["fam_stroke"]
    logit += (aqi / 10) * WEIGHTS["aqi_per_10"]

    p = 1 / (1 + math.exp(-logit))

    if p >= 0.50:
        return 5
    elif p >= 0.35:
        return 4
    elif p >= 0.20:
        return 3
    elif p >= 0.10:
        return 2
    else:
        return 1


def main():
    age = ask_int("Age", min_val=1, max_val=120)
    lung = ask_yes_no("Do you have a diagnosed lung condition?")
    heart = ask_yes_no("Do you have a diagnosed heart condition?")
    smoke = ask_yes_no("Do you smoke?")
    fam_smoke = ask_yes_no("Does your family have a background of smoking?")
    fam_stroke = ask_yes_no("Does your family have a background of stroke?")
    aqi = ask_int("Current AQI for your location (check airnow.gov or aqicn.org if unsure)", min_val=0, max_val=500)

    level = compute_level(age, lung, heart, smoke, fam_smoke, fam_stroke, aqi)
    print(COLORS[level])


if __name__ == "__main__":
    main()
