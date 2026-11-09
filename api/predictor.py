"""
Core prediction logic with domain whitelist + tracking param normalization.
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
MODEL_VERSION = "robust_xgboost_v1"
METADATA_PATH = Path("experiments/baseline/best_model_metadata.json")

# Load feature list from metadata
with open(METADATA_PATH) as _f:
    FEATURE_COLS = json.load(_f)["features"]

# ---------- DOMAIN WHITELIST ----------
# Known legitimate domains. Subdomains of these are also trusted.
TRUSTED_DOMAINS = {
    # Tech
    "google.com", "youtube.com", "gmail.com", "github.com", "gitlab.com",
    "microsoft.com", "apple.com", "amazon.com", "facebook.com", "instagram.com",
    "twitter.com", "linkedin.com", "wikipedia.org", "stackoverflow.com",
    "python.org", "pypi.org", "mozilla.org", "cloudflare.com",
    # Banking (global)
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "hsbc.com", "citibank.com",
    # Bangladesh
    "bkash.com", "nagad.com.bd", "rocket.com.bd", "dutchbanglabank.com",
    "bracbank.com", "citybank.com.bd", "dbbl.com",
    # Trading / Broker
    "exness.com", "binance.com", "coinbase.com", "kraken.com", "metatrader.com",
    "etrade.com", "fidelity.com", "schwab.com",
    # Education
    "coursera.org", "edx.org", "khanacademy.org", "udemy.com",
    # Government (Bangladesh)
    "gov.bd", "bangladesh.gov.bd",
    # CDN / Infra
    "cloudfront.net", "akamai.com", "fastly.com",
}


def is_trusted_domain(url: str) -> bool:
    """Check if URL's domain is in whitelist (including subdomains)."""
    try:
        netloc = urlparse(url).netloc.lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Check exact match or subdomain match
        for trusted in TRUSTED_DOMAINS:
            if netloc == trusted or netloc.endswith("." + trusted):
                return True
        return False
    except Exception:
        return False


# ---------- TRACKING PARAM NORMALIZATION ----------
TRACKING_PARAMS = {
    "_gl", "_ga", "_gid", "_gcl_au", "_gcl_aw", "_fplc", "FPAU",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "yclid",
    "ref", "ref_src", "ref_url", "referrer",
}


def strip_tracking_params(url: str) -> str:
    """
    Remove tracking params from URL query string.
    Example: "google.com?utm_source=x&q=test" → "google.com?q=test"
    """
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url

        # Split query by &
        params = parsed.query.split("&")
        kept_params = []
        for param in params:
            key = param.split("=")[0].lower()
            if key not in TRACKING_PARAMS:
                kept_params.append(param)

        new_query = "&".join(kept_params)
        if new_query:
            new_url = url.replace(parsed.query, new_query, 1)
        else:
            # No params left → remove the "?"
            new_url = url.replace("?" + parsed.query, "", 1)

        return new_url
    except Exception:
        return url


# ---------- MODEL LOADING ----------
_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------- MAIN PREDICTION ----------
def predict(url: str) -> dict:
    """
    Predict phishing/legitimate for a URL.

    Strategy:
    1. Validate URL syntax
    2. Check domain whitelist → trust directly (no ML)
    3. Strip tracking params
    4. Extract features
    5. Predict with XGBoost
    """
    url = str(url).strip()
    if not url or len(url) < 4:
        raise ValueError("URL is empty or too short")
    if "." not in url:
        raise ValueError("URL must contain a dot (e.g., 'example.com')")

    start = time.perf_counter()

    # STEP 1: Whitelist check
    if is_trusted_domain(url):
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "url": url,
            "prediction": "legitimate",
            "confidence": 0.99,
            "phishing_probability": 0.01,
            "latency_ms": round(latency_ms, 3),
            "model_version": MODEL_VERSION + "+whitelist",
        }

    # STEP 2: Strip tracking params (reduce false positives)
    url_clean = strip_tracking_params(url)

    # STEP 3: Extract features from cleaned URL
    features_dict = extract_features(url_clean)

    # STEP 4: Predict
    model = load_model()
    features_df = pd.DataFrame([features_dict])[FEATURE_COLS]
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
    try:
        load_model()
        return {"model_loaded": True, "model_version": MODEL_VERSION}
    except Exception:
        return {"model_loaded": False, "model_version": MODEL_VERSION}