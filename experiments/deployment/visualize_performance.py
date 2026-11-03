"""
Day 10: Visualize deployment performance.
- Latency distribution (sequential)
- Latency vs concurrency
- Throughput vs concurrency
- CPU/RAM vs concurrency
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

SINGLE_CSV = "experiments/deployment/benchmark_single.csv"
CONCURRENT_CSV = "experiments/deployment/benchmark_concurrent.csv"

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def main():
    print("=" * 60)
    print("DAY 10: VISUALIZE PERFORMANCE")
    print("=" * 60)

    # Load data
    single = pd.read_csv(SINGLE_CSV).iloc[0]
    conc = pd.read_csv(CONCURRENT_CSV)
    print(f"Single: n={single['n_requests']}, p50={single['latency_median_ms']:.2f}ms")
    print(f"Concurrent: {len(conc)} scenarios")

    # ---------- Plot 1: Latency percentiles (single) ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["latency_min_ms", "latency_median_ms", "latency_mean_ms",
               "latency_p95_ms", "latency_p99_ms", "latency_max_ms"]
    labels = ["Min", "Median\n(p50)", "Mean", "p95", "p99", "Max"]
    values = [single[m] for m in metrics]
    colors = ["green", "green", "steelblue", "orange", "crimson", "darkred"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.2f}ms", ha="center", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single-URL Latency Distribution (n=100 sequential requests)")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_latency_percentiles.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '22_latency_percentiles.png'}")

    # ---------- Plot 2: Latency vs concurrency ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conc["concurrency"], conc["latency_median_ms"], "o-",
            label="Median (p50)", linewidth=2, markersize=8, color="steelblue")
    ax.plot(conc["concurrency"], conc["latency_p95_ms"], "s--",
            label="p95", linewidth=2, markersize=8, color="orange")
    ax.plot(conc["concurrency"], conc["latency_p99_ms"], "^:",
            label="p99", linewidth=2, markersize=8, color="crimson")
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Concurrency")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "23_latency_vs_concurrency.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '23_latency_vs_concurrency.png'}")

    # ---------- Plot 3: Throughput vs concurrency ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conc["concurrency"].astype(str), conc["throughput_rps"],
                  color="steelblue")
    for bar, v in zip(bars, conc["throughput_rps"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", fontweight="bold")
    # Mark peak
    peak_idx = conc["throughput_rps"].idxmax()
    bars[peak_idx].set_color("gold")
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Throughput (requests/sec)")
    ax.set_title("Throughput vs Concurrency (Peak highlighted)")
    ax.set_ylim(0, conc["throughput_rps"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "24_throughput_vs_concurrency.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '24_throughput_vs_concurrency.png'}")

    # ---------- Plot 4: CPU + RAM vs concurrency ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(conc["concurrency"], conc["cpu_percent"], "o-",
             linewidth=2, markersize=8, color="crimson")
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5,
                label="Single core")
    ax1.set_xlabel("Concurrent users")
    ax1.set_ylabel("CPU usage (%)")
    ax1.set_title("CPU Usage vs Concurrency")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(conc["concurrency"], conc["ram_mb"], "s-",
             linewidth=2, markersize=8, color="steelblue")
    ax2.set_xlabel("Concurrent users")
    ax2.set_ylabel("RAM usage (MB)")
    ax2.set_title("RAM Usage vs Concurrency")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, conc["ram_mb"].max() * 1.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "25_resource_usage.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '25_resource_usage.png'}")

    # ---------- Plot 5: Error rate (should be 0) ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(conc["concurrency"].astype(str),
                  conc["error_rate"] * 100, color="green")
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Error Rate vs Concurrency (0% = perfect reliability)")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, conc["error_rate"] * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "26_error_rate.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '26_error_rate.png'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()