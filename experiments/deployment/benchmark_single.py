"""
Day 10: Single-URL benchmark + sequential test.
Measures latency distribution across many predictions.
"""
import time
from pathlib import Path

import numpy as np
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

TEST_URLS = [
    "https://www.google.com",
    "https://github.com",
    "https://www.wikipedia.org",
    "https://www.python.org",
    "http://paypal.com.verify-login.xyz/account/update",
    "http://amazon.com.secure-signin.xyz/login",
    "http://microsoft.account-verify.xyz/signin",
    "https://bit.ly/test123",
]

OUT_DIR = Path("experiments/deployment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("DAY 10: SINGLE URL BENCHMARK")
    print("=" * 60)

    # Warm-up
    print("\n[Warm-up] 3 requests...")
    for _ in range(3):
        requests.post(f"{API_URL}/predict", json={"url": "https://www.google.com"})

    # Single-URL test
    print("\n[Single] 100 sequential requests...")
    latencies = []
    start_total = time.perf_counter()
    for i in range(100):
        url = TEST_URLS[i % len(TEST_URLS)]
        t0 = time.perf_counter()
        r = requests.post(f"{API_URL}/predict", json={"url": url})
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        latencies.append(elapsed)
    total_time = time.perf_counter() - start_total

    latencies = np.array(latencies)
    stats = {
        "n_requests": 100,
        "total_time_s": total_time,
        "throughput_rps": 100 / total_time,
        "latency_mean_ms": float(latencies.mean()),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_min_ms": float(latencies.min()),
        "latency_max_ms": float(latencies.max()),
    }

    print(f"\n[Results] Sequential test (n=100):")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    df = pd.DataFrame([stats])
    df.to_csv(OUT_DIR / "benchmark_single.csv", index=False)
    df.to_csv(TABLE_DIR / "day10_benchmark_single.csv", index=False)
    print(f"\n[OK] Saved {OUT_DIR / 'benchmark_single.csv'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()