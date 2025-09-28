import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# ----------------------
# Load dataset
# ----------------------
PROJECT_ROOT = r"C:\Users\balaj\Pictures\hackathon now"
fertilizer_csv = os.path.join(PROJECT_ROOT, "data", "processed_fertilizer_recommendation.csv")

df = pd.read_csv(fertilizer_csv)
df = df.dropna()

# ----------------------
# Encode categorical columns
# ----------------------
encoders = {}

le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'].astype(str))
encoders['Soil Type'] = le_soil

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'].astype(str))
encoders['Crop Type'] = le_crop

le_fert = LabelEncoder()
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'].astype(str))
encoders['Fertilizer Name'] = le_fert

y = df['Fertilizer Name']

# Features
X = df[['Temparature', 'Humidity', 'Soil Moisture', 
        'Soil Type', 'Crop Type', 'Nitrogen', 
        'Potassium', 'Phosphorous']]

# ----------------------
# Train-test split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# Train model
# ----------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ----------------------
# Evaluate model
# ----------------------
y_pred = model.predict(X_test)
print("Fertilizer Model Accuracy:", accuracy_score(y_test, y_pred))

# ----------------------
# Save model + encoders
# ----------------------
models_dir = os.path.join(PROJECT_ROOT, "models")
encoders_dir = os.path.join(PROJECT_ROOT, "encoders")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(encoders_dir, exist_ok=True)

model_path = os.path.join(models_dir, "fertilizer_model.pkl")
encoders_path = os.path.join(encoders_dir, "fertilizer_encoders.pkl")

joblib.dump(model, model_path)
joblib.dump(encoders, encoders_path)

print(f"✅ Fertilizer model saved at {model_path}")
print(f"✅ Fertilizer encoders saved at {encoders_path}")