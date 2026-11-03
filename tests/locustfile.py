"""
Day 10: Locust load test for FastAPI phishing detection service.

Usage:
    locust -f tests/locustfile.py --host=http://127.0.0.1:8000

Then open http://localhost:8089 and set:
- Number of users: 10, 100, 1000
- Spawn rate: 10
"""
from locust import HttpUser, task, between
import random


SAMPLE_URLS = [
    "https://www.google.com/search?q=test",
    "https://github.com/NazmullSyntax/phishing-detection-research",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org",
    "https://www.python.org",
    "http://paypal.com.verify-login.xyz/account/update",
    "http://amazon.com.secure-signin.xyz/login",
    "http://microsoft.account-verify.xyz/signin",
    "http://apple-id-locked.secure-auth.xyz",
    "http://bank-secure-update.verify.xyz/account",
    "https://bit.ly/test123",
    "https://t.ly/abc456",
]


class PhishingDetectorUser(HttpUser):
    """Simulates a user submitting URLs for phishing detection."""
    wait_time = between(0.1, 0.5)  # 100-500ms between requests

    @task(3)
    def predict_phishing(self):
        """Predict a phishing URL (simulated attack) — 3x weight."""
        url = random.choice(SAMPLE_URLS[5:10])  # phishing ones
        self.client.post("/predict", json={"url": url})

    @task(2)
    def predict_legitimate(self):
        """Predict a legitimate URL — 2x weight."""
        url = random.choice(SAMPLE_URLS[:5])  # legit ones
        self.client.post("/predict", json={"url": url})

    @task(1)
    def predict_short(self):
        """Predict a short URL (known limitation) — 1x weight."""
        url = random.choice(SAMPLE_URLS[10:])
        self.client.post("/predict", json={"url": url})

    @task(1)
    def health_check(self):
        """Health check."""
        self.client.get("/health")