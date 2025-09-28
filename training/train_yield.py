import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

PROJECT_ROOT = r"C:\Users\balaj\Pictures\hackathon now"
yield_csv = os.path.join(PROJECT_ROOT, "data", "processed_production_clean.csv")

df = pd.read_csv(yield_csv)
df = df.dropna()

# ✅ create encoder dictionary
encoders = {}

# encode categorical columns
for col in ['state_name', 'district_name', 'season', 'crop']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le   # store encoder for later use

# Features and target
X = df[['state_name', 'district_name', 'crop_year', 'season', 'crop', 'area']]
y = df['yield_kg_per_ha']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Yield Model R²:", r2_score(y_test, y_pred))
print("Yield Model MAE:", mean_absolute_error(y_test, y_pred))

# Make sure output folders exist
models_dir = os.path.join(PROJECT_ROOT, "models")
encoders_dir = os.path.join(PROJECT_ROOT, "encoders")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(encoders_dir, exist_ok=True)

# Save model and encoders
model_path = os.path.join(models_dir, "yield_model.pkl")
encoder_path = os.path.join(encoders_dir, "yield_encoders.pkl")

joblib.dump(model, model_path)
joblib.dump(encoders, encoder_path)

print(f"✅ Yield model saved at {model_path}")
print(f"✅ Yield encoders saved at {encoder_path}")