import os
import joblib
import json
from pathlib import Path

PROJECT_ROOT = r"C:\Users\balaj\Pictures\hackathon now"

# ----------------------
# Load models & encoders
# ----------------------
yield_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "yield_model.pkl"))
yield_encoders = joblib.load(os.path.join(PROJECT_ROOT, "encoders", "yield_encoders.pkl"))

crop_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "crop_model.pkl"))
crop_label_encoder = joblib.load(os.path.join(PROJECT_ROOT, "encoders", "crop_label_encoder.pkl"))

fert_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "fertilizer_model.pkl"))
fert_encoders = joblib.load(os.path.join(PROJECT_ROOT, "encoders", "fertilizer_encoders.pkl"))

# ----------------------
# Fertilizer gain table
# ----------------------
fert_gain_path = Path(os.path.join(PROJECT_ROOT, "encoders", "fert_gain_table.json"))
if fert_gain_path.exists():
    fert_gain_table = json.load(open(fert_gain_path))
else:
    fert_gain_table = {}

DEFAULT_FERT_GAIN = 0.10
YIELD_SCALE_FACTOR = 100.0


# ----------------------
# Helpers
# ----------------------
def safe_encode(encoder, value):
    try:
        return encoder.transform([str(value).lower()])[0]
    except Exception:
        return -1


def encode_yield_input(state, district, crop_year, season, crop_str, area):
    s = safe_encode(yield_encoders['state_name'], state)
    d = safe_encode(yield_encoders['district_name'], district)
    se = safe_encode(yield_encoders['season'], season)
    crop_enc = safe_encode(yield_encoders['crop'], crop_str)
    return [s, d, int(crop_year), se, crop_enc, float(area)]


def compute_fert_gain(crop_str, fert_name):
    crop_str = str(crop_str).lower()
    fert_name = str(fert_name).lower()
    if crop_str in fert_gain_table and fert_name in fert_gain_table[crop_str]:
        return float(fert_gain_table[crop_str][fert_name])
    return DEFAULT_FERT_GAIN


# ----------------------
# Main prediction function
# ----------------------
def predict_crop(farm_meta, soil_weather, selected_crop):
    # normalize
    farm_meta = {k: (v.lower() if isinstance(v, str) else v) for k, v in farm_meta.items()}
    soil_weather = {k: (v.lower() if isinstance(v, str) else v) for k, v in soil_weather.items()}
    selected_crop = selected_crop.lower()

    # -----------------
    # 1) Farmer's selected crop
    # -----------------
    X_sel = encode_yield_input(
        farm_meta['state'], farm_meta['district'],
        farm_meta['crop_year'], farm_meta['season'],
        selected_crop, farm_meta['area']
    )
    sel_yield = yield_model.predict([X_sel])[0] * YIELD_SCALE_FACTOR

    # Fertilizer for selected crop
    st_enc = safe_encode(fert_encoders['Soil Type'], soil_weather['Soil Type'])
    ct_enc = safe_encode(fert_encoders['Crop Type'], selected_crop)

    fert_features = [soil_weather['Temperature'], soil_weather['Humidity'],
                     soil_weather['Soil Moisture'], st_enc, ct_enc,
                     soil_weather['N'], soil_weather['K'], soil_weather['P']]

    fert_idx = fert_model.predict([fert_features])[0]
    fert_name_sel = fert_encoders['Fertilizer Name'].inverse_transform([fert_idx])[0]

    fert_gain_sel = compute_fert_gain(selected_crop, fert_name_sel)
    final_yield_sel = sel_yield * (1 + fert_gain_sel)

    # -----------------
    # 2) Alternative crop suggestion
    # -----------------
    crop_features = [soil_weather['N'], soil_weather['P'], soil_weather['K'],
                     soil_weather['temperature'], soil_weather['humidity'],
                     soil_weather['ph'], soil_weather['rainfall']]
    alt_idx = crop_model.predict([crop_features])[0]
    alt_crop = crop_label_encoder.inverse_transform([alt_idx])[0].lower()

    X_alt = encode_yield_input(
        farm_meta['state'], farm_meta['district'],
        farm_meta['crop_year'], farm_meta['season'],
        alt_crop, farm_meta['area']
    )
    alt_yield = yield_model.predict([X_alt])[0] * YIELD_SCALE_FACTOR

    # Fertilizer for alt crop
    ct_alt_enc = safe_encode(fert_encoders['Crop Type'], alt_crop)
    fert_features_alt = [soil_weather['Temperature'], soil_weather['Humidity'],
                         soil_weather['Soil Moisture'], st_enc, ct_alt_enc,
                         soil_weather['N'], soil_weather['K'], soil_weather['P']]

    fert_idx_alt = fert_model.predict([fert_features_alt])[0]
    fert_name_alt = fert_encoders['Fertilizer Name'].inverse_transform([fert_idx_alt])[0]

    fert_gain_alt = compute_fert_gain(alt_crop, fert_name_alt)
    final_yield_alt = alt_yield * (1 + fert_gain_alt)

    diff_pct_alt = ((final_yield_alt - final_yield_sel) / final_yield_sel) * 100 if final_yield_sel > 0 else 0

    return {
        "selected_crop": selected_crop.capitalize(),
        "sel_yield": round(float(sel_yield), 2),
        "fert_name_sel": fert_name_sel,
        "fert_gain_sel": round(float(fert_gain_sel * 100), 2),
        "final_yield_sel": round(float(final_yield_sel), 2),
        "alt_crop": alt_crop.capitalize(),
        "alt_yield": round(float(alt_yield), 2),
        "fert_name_alt": fert_name_alt,
        "fert_gain_alt": round(float(fert_gain_alt * 100), 2),
        "final_yield_alt": round(float(final_yield_alt), 2),
        "diff_pct_alt": round(float(diff_pct_alt), 2)
    }


# ----------------------
# Console Input
# ----------------------
if __name__ == "__main__":
    print("🌾 Enter Farmer Details")
    state = input("Enter State: ")
    district = input("Enter District: ")
    crop_year = int(input("Enter Crop Year (e.g. 2024): "))
    season = input("Enter Season (e.g. Kharif, Rabi): ")
    area = float(input("Enter Area (ha): "))

    print("\n🌱 Enter Soil & Weather Details")
    N = int(input("Enter Nitrogen (N): "))
    P = int(input("Enter Phosphorus (P): "))
    K = int(input("Enter Potassium (K): "))
    temperature = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    ph = float(input("Enter Soil pH: "))
    rainfall = float(input("Enter Rainfall (mm): "))
    soil_type = input("Enter Soil Type (e.g. Sandy, Clay): ")
    soil_moisture = float(input("Enter Soil Moisture (%): "))

    selected_crop = input("\n🌾 Enter the crop you want to grow: ")

    farm = {
        "state": state, "district": district,
        "crop_year": crop_year, "season": season,
        "area": area
    }

    soil_weather = {
        "N": N, "P": P, "K": K,
        "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall,
        "Temperature": temperature, "Humidity": humidity, "Soil Moisture": soil_moisture,
        "Soil Type": soil_type, "Crop Type": selected_crop
    }

    result = predict_crop(farm, soil_weather, selected_crop)

    print("\n🌾 Crop Yield Prediction Report\n")
    print(f"👉 Selected Crop: {result['selected_crop']}")
    print(f"Expected Yield: {result['sel_yield']} kg/ha")
    print(f"Recommended Fertilizer: {result['fert_name_sel']} (+{result['fert_gain_sel']}%)")
    print(f"✅ Final Adjusted Yield: {result['final_yield_sel']} kg/ha\n")
    print(f"👉 Alternative Crop Suggestion: {result['alt_crop']}")
    print(f"Expected Yield: {result['alt_yield']} kg/ha")
    print(f"Recommended Fertilizer: {result['fert_name_alt']} (+{result['fert_gain_alt']}%)")
    print(f"✅ Final Adjusted Yield: {result['final_yield_alt']} kg/ha ({result['diff_pct_alt']}% vs Selected)\n")

    print("📢 Advice:")
    if result['final_yield_alt'] > result['final_yield_sel']:
        print(f"- Consider switching to {result['alt_crop']} + {result['fert_name_alt']} for higher productivity")
    else:
        print(f"- Continue with {result['selected_crop']} + {result['fert_name_sel']} for stable productivity")
    print(f"- Irrigate twice per week (based on weather)")
    print(f"- Watch out for pest alerts in {season} season")
