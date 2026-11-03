"""
Day 3: Exploratory Data Analysis on processed splits.
Generates 5 plots saved to results/figures/ and 1 stats CSV to results/tables/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from pathlib import Path

TRAIN = "data/processed/train.csv"
VAL = "data/processed/val.csv"
TEST = "data/processed/test.csv"

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def main():
    print("=" * 60)
    print("DAY 3: EDA")
    print("=" * 60)

    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    test = pd.read_csv(TEST)
    full = pd.concat([train, val, test], ignore_index=True)
    print(f"Loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    # Feature helpers
    full["url_length"] = full["url"].str.len()
    full["num_dots"] = full["url"].str.count(r"\.")
    full["num_hyphens"] = full["url"].str.count("-")
    full["num_digits"] = full["url"].str.count(r"\d")
    full["has_https"] = full["url"].str.startswith("https").astype(int)
    full["has_ip"] = full["url"].apply(
        lambda u: bool(
            urlparse(u).netloc
            and urlparse(u).netloc.split(":")[0].replace(".", "").isdigit()
        )
    )

    # ---- Plot 1: Class distribution by split ----
    fig, ax = plt.subplots(figsize=(8, 5))
    split_data = pd.DataFrame({
        "Split": ["Train"] * 2 + ["Val"] * 2 + ["Test"] * 2,
        "Label": ["legitimate", "phishing"] * 3,
        "Count": [
            (train["label"] == "legitimate").sum(), (train["label"] == "phishing").sum(),
            (val["label"] == "legitimate").sum(), (val["label"] == "phishing").sum(),
            (test["label"] == "legitimate").sum(), (test["label"] == "phishing").sum(),
        ]
    })
    sns.barplot(data=split_data, x="Split", y="Count", hue="Label", ax=ax)
    ax.set_title("Class Distribution per Split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '01_class_distribution.png'}")

    # ---- Plot 2: URL length distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("legitimate", "green"), ("phishing", "red")]:
        subset = full[full["label"] == label]["url_length"]
        ax.hist(subset, bins=50, alpha=0.5, label=label, color=color)
    ax.set_xlabel("URL length (chars)")
    ax.set_ylabel("Frequency")
    ax.set_title("URL Length Distribution by Class")
    ax.set_xlim(0, 200)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_url_length.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '02_url_length.png'}")

    # ---- Plot 3: HTTPS usage by class ----
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(
        data=full.groupby(["label", "has_https"]).size().reset_index(name="count"),
        x="label", y="count", hue="has_https", ax=ax
    )
    ax.set_title("HTTPS Usage by Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_https_usage.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '03_https_usage.png'}")

    # ---- Plot 4: Number of dots distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("legitimate", "green"), ("phishing", "red")]:
        subset = full[full["label"] == label]["num_dots"]
        ax.hist(subset, bins=20, alpha=0.5, label=label, color=color)
    ax.set_xlabel("Number of dots in URL")
    ax.set_ylabel("Frequency")
    ax.set_title("Dot Count Distribution")
    ax.set_xlim(0, 15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_dot_count.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '04_dot_count.png'}")

    # ---- Plot 5: IP-in-URL rate ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ip_rate = full.groupby("label")["has_ip"].mean() * 100
    ip_rate.plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_ylabel("% of URLs with IP in netloc")
    ax.set_title("IP-in-URL Rate by Class")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_ip_rate.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '05_ip_rate.png'}")

    # ---- Summary statistics ----
    stats = full.groupby("label").agg({
        "url_length": ["mean", "median", "max"],
        "num_dots": "mean",
        "num_hyphens": "mean",
        "num_digits": "mean",
    }).round(2)
    print("\n[Summary stats by class]")
    print(stats)
    stats.to_csv(TABLE_DIR / "day3_summary_stats.csv")
    print(f"[OK] Saved {TABLE_DIR / 'day3_summary_stats.csv'}")

    print("\n" + "=" * 60)
    print("EDA DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()