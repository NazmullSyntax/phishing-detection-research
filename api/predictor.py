"""
Core prediction logic. Loads model once, extracts features, predicts.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.extract_features import extract_features

MODEL_PATH = Path("models/robust_xgboost.pkl")
MODEL_VERSION = "robust_xgboost_v1"
METADATA_PATH = Path("experiments/baseline/best_model_metadata.json")

# Load feature list from metadata (source of truth — preserves training order)
with open(METADATA_PATH) as _f:
    FEATURE_COLS = json.load(_f)["features"]

# Loaded lazily on first use (avoid blocking startup)
_model = None


def load_model():
    """Load model from disk. Cached after first call."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(url: str) -> dict:
    """Predict phishing/legitimate for a URL."""
    url = str(url).strip()
    if not url or len(url) < 4:
        raise ValueError("URL is empty or too short")
    if "." not in url:
        raise ValueError("URL must contain a dot (e.g., 'example.com')")

    start = time.perf_counter()

    # 1. Extract features
    features_dict = extract_features(url)

    # 2. Convert to DataFrame with EXACT column order from training metadata
    model = load_model()
    features_df = pd.DataFrame([features_dict])[FEATURE_COLS]

    # 3. Predict
    proba = model.predict_proba(features_df.values)[0]
    phishing_prob = float(proba[1])
    prediction = "phishing" if phishing_prob >= 0.5 else "legitimate"
    confidence = float(max(proba))

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "url": url,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_prob, 4),
        "latency_ms": round(latency_ms, 3),
        "model_version": MODEL_VERSION,
    }


def get_model_info() -> dict:
    """Return model health/info."""
    try:
        load_model()
        return {"model_loaded": True, "model_version": MODEL_VERSION}
    except Exception:
        return {"model_loaded": False, "model_version": MODEL_VERSION}