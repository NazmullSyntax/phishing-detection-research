"""
Core prediction logic. Loads model once, extracts features, predicts.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.extract_features import extract_features

MODEL_PATH = Path("models/robust_xgboost.pkl")
MODEL_VERSION = "robust_xgboost_v1"

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
    """
    Predict phishing/legitimate for a URL.
    Returns dict with prediction, confidence, latency, etc.
    """
    # Validate URL syntax minimally
    url = str(url).strip()
    if not url or len(url) < 4:
        raise ValueError("URL is empty or too short")
    if "." not in url:
        raise ValueError("URL must contain a dot (e.g., 'example.com')")

    # Time the whole prediction
    start = time.perf_counter()

    # 1. Extract features
    features_dict = extract_features(url)

    # 2. Convert to DataFrame with same column order as training
    model = load_model()
    feature_cols = model.get_booster().feature_names
    features_df = pd.DataFrame([features_dict])[feature_cols]

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