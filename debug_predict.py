"""Debug prediction pipeline locally."""
import sys
sys.path.insert(0, "src")

from features.extract_features import extract_features
import joblib
import pandas as pd

url = "https://www.google.com/search?q=test"

print("=" * 60)
print("STEP 1: Extract features")
print("=" * 60)
features = extract_features(url)
print(f"Features dict: {features}")

print("\n" + "=" * 60)
print("STEP 2: Load model")
print("=" * 60)
model = joblib.load("models/robust_xgboost.pkl")
print(f"Model type: {type(model)}")

print("\n" + "=" * 60)
print("STEP 3: Get feature names from model")
print("=" * 60)
try:
    feat_names = model.get_booster().feature_names
    print(f"Feature names: {feat_names}")
    print(f"Count: {len(feat_names) if feat_names else 'None'}")
except Exception as e:
    print(f"ERROR: {e}")
    feat_names = None

print("\n" + "=" * 60)
print("STEP 4: Build DataFrame")
print("=" * 60)
try:
    if feat_names:
        df = pd.DataFrame([features])[feat_names]
    else:
        df = pd.DataFrame([features])
    print(f"DataFrame shape: {df.shape}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("STEP 5: Predict")
print("=" * 60)
try:
    proba = model.predict_proba(df.values)[0]
    print(f"Proba: {proba}")
except Exception as e:
    import traceback
    traceback.print_exc()