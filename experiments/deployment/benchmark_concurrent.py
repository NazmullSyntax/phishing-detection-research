"""
Day 10: Concurrent load test.
Tests multiple concurrency levels: 1, 5, 10, 50, 100 users.
"""
import concurrent.futures
import time
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import psutil
import os

API_URL = "http://127.0.0.1:8000"

TEST_URLS = [
    "https://www.google.com",
    "https://github.com",
    "https://www.wikipedia.org",
    "http://paypal.com.verify-login.xyz/account/update",
    "http://amazon.com.secure-signin.xyz/login",
]

OUT_DIR = Path("experiments/deployment")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def one_request(url):
    """Single request, returns latency in ms."""
    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/predict", json={"url": url})
    elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return None
    return elapsed


def get_process_stats():
    """Aggregate CPU % and RAM (MB) of Python processes."""
    total_cpu = 0.0
    total_mem_mb = 0.0
    for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in (proc.info['name'] or '').lower():
                total_cpu += proc.info['cpu_percent'] or 0.0
                if proc.info['memory_info']:
                    total_mem_mb += proc.info['memory_info'].rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total_cpu, total_mem_mb


def test_concurrency(concurrency: int, total_requests: int):
    """Fire total_requests with `concurrency` parallel workers."""
    print(f"\n[Test] concurrency={concurrency}, total={total_requests}")

    # Reset psutil CPU counters
    psutil.cpu_percent(interval=None)

    start = time.perf_counter()
    latencies = []
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(one_request, TEST_URLS[i % len(TEST_URLS)])
            for i in range(total_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result is None:
                errors += 1
            else:
                latencies.append(result)

    total_time = time.perf_counter() - start
    cpu, mem = get_process_stats()

    if not latencies:
        return None

    latencies = np.array(latencies)
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_time_s": total_time,
        "throughput_rps": len(latencies) / total_time,
        "latency_mean_ms": float(latencies.mean()),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "errors": errors,
        "error_rate": errors / total_requests,
        "cpu_percent": cpu,
        "ram_mb": mem,
    }


def main():
    print("=" * 60)
    print("DAY 10: CONCURRENT LOAD TEST")
    print("=" * 60)

    # Warm-up
    print("[Warm-up]")
    for _ in range(5):
        requests.get(f"{API_URL}/health")

    results = []
    scenarios = [
        (1, 100),
        (5, 200),
        (10, 300),
        (50, 500),
        (100, 1000),
    ]

    for concurrency, total in scenarios:
        r = test_concurrency(concurrency, total)
        if r:
            results.append(r)
            print(f"  Throughput: {r['throughput_rps']:.1f} req/s")
            print(f"  Latency p50: {r['latency_median_ms']:.2f}ms | p95: {r['latency_p95_ms']:.2f}ms")
            print(f"  CPU: {r['cpu_percent']:.1f}% | RAM: {r['ram_mb']:.0f}MB")
            print(f"  Errors: {r['errors']} ({r['error_rate']*100:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "benchmark_concurrent.csv", index=False)
    df.to_csv(TABLE_DIR / "day10_benchmark_concurrent.csv", index=False)
    print(f"\n[OK] Saved {OUT_DIR / 'benchmark_concurrent.csv'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df[["concurrency", "throughput_rps", "latency_median_ms", "latency_p95_ms", "error_rate"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()