"""
Core prediction logic with:
- 26 features (v2)
- Loaded threshold from experiments/baseline/threshold.json
- Uncertain zone for low-confidence predictions
- Trusted domain whitelist (policy layer, not ML)
- Tracking param stripping
"""
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.extract_features import extract_features

MODEL_PATH = Path("models/robust_xgboost.pkl")
MODEL_VERSION = "robust_xgboost_v2"
METADATA_PATH = Path("experiments/baseline/best_model_metadata.json")
THRESHOLD_PATH = Path("experiments/baseline/threshold.json")

# Load feature list from metadata
with open(METADATA_PATH) as _f:
    FEATURE_COLS = json.load(_f)["features"]

# Load optimal threshold (tuned on validation set)
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH) as _t:
        THRESHOLD_DATA = json.load(_t)
    PHISHING_THRESHOLD = THRESHOLD_DATA["optimal_threshold"]
else:
        PHISHING_THRESHOLD = 0.92  # fallback # fallback

# Uncertain zone: probabilities in this range are "uncertain"
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = PHISHING_THRESHOLD  # e.g., 0.85

# ---------- TRUSTED DOMAIN WHITELIST ----------
# This is a POLICY LAYER, not learned from Kaggle.
# Tested separately from ML model evaluation.
TRUSTED_DOMAINS = {
    # Tech giants
    "google.com", "youtube.com", "gmail.com", "github.com", "gitlab.com",
    "microsoft.com", "apple.com", "amazon.com", "facebook.com", "instagram.com",
    "twitter.com", "linkedin.com", "wikipedia.org", "stackoverflow.com",
    "python.org", "pypi.org", "mozilla.org", "cloudflare.com",
    "openai.com", "anthropic.com", "chatgpt.com", "claude.ai",
    # Education
    "coursera.org", "edx.org", "khanacademy.org", "udemy.com",
    "phitron.io", "programming-hero.com", "freecodecamp.org", "w3schools.com",
    # Banking (global)
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "hsbc.com", "citibank.com",
    # Bangladesh
    "bkash.com", "nagad.com.bd", "rocket.com.bd", "dutchbanglabank.com",
    "bracbank.com", "citybank.com.bd", "dbbl.com",
    # Trading
    "exness.com", "binance.com", "coinbase.com", "kraken.com",
    "metatrader.com", "etrade.com", "fidelity.com", "schwab.com",
    # Government (Bangladesh)
    "gov.bd", "bangladesh.gov.bd",
    # CDN / Infra
    "cloudfront.net", "akamai.com", "fastly.com",
}

# ---------- TRACKING PARAMS ----------
TRACKING_PARAMS = {
    "_gl", "_ga", "_gid", "_gcl_au", "_gcl_aw", "_fplc", "FPAU",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "yclid",
}


def is_trusted_domain(url: str) -> bool:
    """Check if URL's domain is in whitelist (proper subdomain matching)."""
    try:
        netloc = urlparse(url).netloc.lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        for trusted in TRUSTED_DOMAINS:
            if netloc == trusted or netloc.endswith("." + trusted):
                return True
        return False
    except Exception:
        return False


def strip_tracking_params(url: str) -> str:
    """Remove tracking params from URL query string."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url

        params = parsed.query.split("&")
        kept = []
        for param in params:
            key = param.split("=")[0].lower()
            if key not in TRACKING_PARAMS:
                kept.append(param)

        new_query = "&".join(kept)
        if new_query:
            return url.replace(parsed.query, new_query, 1)
        else:
            return url.replace("?" + parsed.query, "", 1)
    except Exception:
        return url


_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(url: str) -> dict:
    """
    Predict phishing/legitimate/uncertain for a URL.

    Strategy:
    1. Validate URL syntax
    2. Whitelist check (policy layer) → trust directly
    3. Strip tracking params
    4. Extract 26 features
    5. Predict with XGBoost
    6. Apply threshold + uncertain zone
    """
    url = str(url).strip()
    if not url or len(url) < 4:
        raise ValueError("URL is empty or too short")
    if "." not in url:
        raise ValueError("URL must contain a dot (e.g., 'example.com')")

    start = time.perf_counter()

    # Whitelist check
    if is_trusted_domain(url):
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "url": url,
            "prediction": "legitimate",
            "confidence": 0.99,
            "phishing_probability": 0.01,
            "risk_level": "safe",
            "latency_ms": round(latency_ms, 3),
            "model_version": MODEL_VERSION + "+whitelist",
            "threshold_used": None,
        }

    # Strip tracking params
    url_clean = strip_tracking_params(url)

    # Extract features
    features_dict = extract_features(url_clean)

    # Predict
    model = load_model()
    features_df = pd.DataFrame([features_dict])[FEATURE_COLS]
    proba = model.predict_proba(features_df.values)[0]
    phishing_prob = float(proba[1])

    # Apply threshold + uncertain zone
    if phishing_prob >= PHISHING_THRESHOLD:
        prediction = "phishing"
        confidence = phishing_prob
        risk = "high"
    elif phishing_prob < UNCERTAIN_LOW:
        prediction = "legitimate"
        confidence = 1 - phishing_prob
        risk = "safe"
    else:
        prediction = "uncertain"
        confidence = 1 - abs(phishing_prob - 0.5) * 2  # peaks at 0.5
        risk = "medium"

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "url": url,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_prob, 4),
        "risk_level": risk,
        "latency_ms": round(latency_ms, 3),
        "model_version": MODEL_VERSION,
        "threshold_used": PHISHING_THRESHOLD,
    }


def get_model_info() -> dict:
    try:
        load_model()
        return {
            "model_loaded": True,
            "model_version": MODEL_VERSION,
            "threshold": PHISHING_THRESHOLD,
        }
    except Exception:
        return {"model_loaded": False, "model_version": MODEL_VERSION, "threshold": None}