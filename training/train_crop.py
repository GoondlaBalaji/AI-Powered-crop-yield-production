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
crop_csv = os.path.join(PROJECT_ROOT, "data", "processed_crop_recommendation.csv")

df = pd.read_csv(crop_csv)
df = df.dropna()

# ----------------------
# Features and target
# ----------------------
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# ✅ Keep encoder so we can save it
le_label = LabelEncoder()
y = le_label.fit_transform(df['label'].astype(str))

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
print("Crop Model Accuracy:", accuracy_score(y_test, y_pred))

# ----------------------
# Save model + encoder
# ----------------------
models_dir = os.path.join(PROJECT_ROOT, "models")
encoders_dir = os.path.join(PROJECT_ROOT, "encoders")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(encoders_dir, exist_ok=True)

model_path = os.path.join(models_dir, "crop_model.pkl")
encoder_path = os.path.join(encoders_dir, "crop_label_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(le_label, encoder_path)

print(f"✅ Crop model saved at {model_path}")
print(f"✅ Crop encoder saved at {encoder_path}")