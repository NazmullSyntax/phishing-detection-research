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

with open(METADATA_PATH) as _f:
    FEATURE_COLS = json.load(_f)["features"]

# সেফটি নেট: ৯৫% এর নিচে কনফিডেন্স হলে ফিশিং বলবে না
PHISHING_THRESHOLD = 0.95
UNCERTAIN_LOW = 0.35

# Whitelist (বিশ্বস্ত সাইট)
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "gmail.com", "github.com", "gitlab.com",
    "microsoft.com", "apple.com", "amazon.com", "facebook.com", "instagram.com",
    "twitter.com", "linkedin.com", "wikipedia.org", "stackoverflow.com",
    "python.org", "pypi.org", "mozilla.org", "cloudflare.com",
    "openai.com", "anthropic.com", "chatgpt.com", "claude.ai",
    "coursera.org", "edx.org", "khanacademy.org", "udemy.com",
    "phitron.io", "programming-hero.com", "freecodecamp.org", "w3schools.com",
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "hsbc.com", "citibank.com", "bkash.com", "nagad.com.bd", "rocket.com.bd",
    "dutchbanglabank.com", "bracbank.com", "citybank.com.bd", "dbbl.com",
    "exness.com", "binance.com", "coinbase.com", "kraken.com",
    "metatrader.com", "etrade.com", "fidelity.com", "schwab.com",
    "gov.bd", "bangladesh.gov.bd", "cloudfront.net", "akamai.com", "fastly.com",
}

TRACKING_PARAMS = {
    "_gl", "_ga", "_gid", "_gcl_au", "_gcl_aw", "_fplc", "FPAU",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "yclid",
}


def is_trusted_domain(url: str) -> bool:
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
    url = str(url).strip()
    if not url or len(url) < 4:
        raise ValueError("URL is empty or too short")
    if "." not in url:
        raise ValueError("URL must contain a dot (e.g., 'example.com')")

    start = time.perf_counter()

    # Whitelist চেক
    if is_trusted_domain(url):
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "url": url, "prediction": "legitimate", "confidence": 0.99,
            "phishing_probability": 0.01, "risk_level": "safe",
            "latency_ms": round(latency_ms, 3), "model_version": MODEL_VERSION + "+whitelist",
            "threshold_used": None,
        }

    url_clean = strip_tracking_params(url)
    features_dict = extract_features(url_clean)

    model = load_model()
    features_df = pd.DataFrame([features_dict])[FEATURE_COLS]
    proba = model.predict_proba(features_df.values)[0]
    phishing_prob = float(proba[1])

    # Threshold 95% (0.95) ব্যবহার করা হচ্ছে
    if phishing_prob >= PHISHING_THRESHOLD:
        prediction, confidence, risk = "phishing", phishing_prob, "high"
    elif phishing_prob < UNCERTAIN_LOW:
        prediction, confidence, risk = "legitimate", 1 - phishing_prob, "safe"
    else:
        prediction, confidence, risk = "uncertain", 1 - abs(phishing_prob - 0.5) * 2, "medium"

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "url": url, "prediction": prediction, "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_prob, 4), "risk_level": risk,
        "latency_ms": round(latency_ms, 3), "model_version": MODEL_VERSION,
        "threshold_used": PHISHING_THRESHOLD,
    }


def get_model_info() -> dict:
    try:
        load_model()
        return {"model_loaded": True, "model_version": MODEL_VERSION, "threshold": PHISHING_THRESHOLD}
    except Exception:
        return {"model_loaded": False, "model_version": MODEL_VERSION, "threshold": None}