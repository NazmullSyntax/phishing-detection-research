"""
Day 6: Visualize attack results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS = "experiments/robustness/attack_results.csv"
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def main():
    print("=" * 60)
    print("DAY 6: VISUALIZE ATTACKS")
    print("=" * 60)

    df = pd.read_csv(RESULTS)
    print(df[["attack", "accuracy", "f1", "fpr"]].to_string(index=False))

    label_map = {
        "none": "Baseline",
        "character_substitution": "Char Sub",
        "subdomain_manipulation": "Subdomain",
        "unicode_homograph": "Unicode",
        "url_encoding": "Encoding",
        "url_shortening": "Shortening",
    }
    df["attack_label"] = df["attack"].map(label_map)

    # Plot 1: Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["gray" if a == "none" else "crimson" if acc < 0.6 else "steelblue"
              for a, acc in zip(df["attack"], df["accuracy"])]
    bars = ax.bar(df["attack_label"], df["accuracy"], color=colors)
    for bar, acc in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy under Adversarial Attacks")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.6, label="Random (50%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_attack_accuracy_comparison.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '11_attack_accuracy_comparison.png'}")

    # Plot 2: FPR
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["gray" if a == "none" else "crimson" if fpr > 0.5 else "steelblue"
              for a, fpr in zip(df["attack"], df["fpr"])]
    bars = ax.bar(df["attack_label"], df["fpr"], color=colors)
    for bar, fpr in zip(bars, df["fpr"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{fpr:.2%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate under Adversarial Attacks")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_attack_fpr_comparison.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '12_attack_fpr_comparison.png'}")

    # Plot 3: Degradation
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_acc = df[df["attack"] == "none"]["accuracy"].values[0]
    attack_df = df[df["attack"] != "none"].copy()
    attack_df["degradation"] = baseline_acc - attack_df["accuracy"]
    colors = ["crimson" if d > 0.3 else "orange" if d > 0.05 else "steelblue"
              for d in attack_df["degradation"]]
    bars = ax.barh(attack_df["attack_label"], attack_df["degradation"], color=colors)
    for bar, deg in zip(bars, attack_df["degradation"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{deg:.2%}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Accuracy Degradation from Baseline")
    ax.set_title("How Much Each Attack Breaks the Model")
    ax.set_xlim(0, 0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_attack_degradation.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '13_attack_degradation.png'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()