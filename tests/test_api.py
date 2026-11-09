"""
Pytest tests for FastAPI endpoints.
Run: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.main import app

client = TestClient(app)


def test_root():
    """Root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Phishing URL Detection API"
    assert "predict" in data


def test_health():
    """Health endpoint reports model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_legitimate():
    """Legit URL classified as legitimate."""
    response = client.post("/predict", json={"url": "https://www.google.com"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] > 0


def test_predict_phishing():
    """Subdomain attack URL classified as phishing."""
    response = client.post(
        "/predict",
        json={"url": "http://paypal.com.verify-login.xyz/account/update"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "phishing"
    assert data["confidence"] >= 0.5


def test_predict_invalid_url():
    """URL without dot is rejected."""
    response = client.post("/predict", json={"url": "no-dot-here"})
    assert response.status_code == 400


def test_predict_empty_url():
    """Empty URL is rejected by Pydantic."""
    response = client.post("/predict", json={"url": ""})
    assert response.status_code == 422


def test_predict_missing_field():
    """Missing URL field returns 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_too_long_url():
    """URL over 2048 chars returns 422."""
    response = client.post("/predict", json={"url": "a" * 3000})
    assert response.status_code == 422


# ---- Whitelist + tracking param tests ----


def test_whitelist_exness():
    """Exness URL should be classified as legitimate (whitelist)."""
    response = client.post("/predict", json={
        "url": "https://my.exness.com/accounts/sign-in?lng=en&_gl=1*8vnco7"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"
    assert "whitelist" in data["model_version"]


def test_whitelist_google():
    """Google URL with tracking should be whitelisted."""
    response = client.post("/predict", json={
        "url": "https://www.google.com/search?q=test&_ga=123&_gl=456"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"


def test_whitelist_python_org():
    """Python.org with tracking should be whitelisted."""
    response = client.post("/predict", json={
        "url": "https://www.python.org/downloads/?utm_source=google&_gl=1*abc"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "legitimate"


def test_phishing_still_works():
    """Phishing detection should still work after whitelist addition."""
    response = client.post("/predict", json={
        "url": "http://paypal.com.verify-login.xyz/account/update"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "phishing"
    assert data["confidence"] >= 0.5