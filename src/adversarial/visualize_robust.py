"""
Day 7: Visualize baseline vs robust comparison.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = "experiments/robustness/baseline_vs_robust.csv"
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def main():
    print("=" * 60)
    print("DAY 7: VISUALIZE ROBUSTNESS")
    print("=" * 60)

    df = pd.read_csv(DATA)
    print(df[["attack", "baseline_accuracy", "robust_accuracy"]].to_string(index=False))

    # Rename for display
    label_map = {
        "clean": "Clean",
        "character_substitution": "Char Sub",
        "subdomain_manipulation": "Subdomain",
        "unicode_homograph": "Unicode",
        "url_encoding": "Encoding",
        "url_shortening": "Shortening",
    }
    df["attack_label"] = df["attack"].map(label_map)

    # ---- Plot 1: Side-by-side accuracy comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["baseline_accuracy"], width,
           label="Baseline (XGBoost)", color="steelblue")
    ax.bar([i + width / 2 for i in x], df["robust_accuracy"], width,
           label="Robust (Adversarial Training)", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(df["attack_label"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Robust Model under Adversarial Attacks")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.legend(loc="lower left")
    for i, (b, r) in enumerate(zip(df["baseline_accuracy"], df["robust_accuracy"])):
        ax.text(i - width / 2, b + 0.02, f"{b:.1%}", ha="center", fontsize=8)
        ax.text(i + width / 2, r + 0.02, f"{r:.1%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_baseline_vs_robust_accuracy.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '14_baseline_vs_robust_accuracy.png'}")

    # ---- Plot 2: Improvement ----
    fig, ax = plt.subplots(figsize=(10, 6))
    attack_df = df[df["attack"] != "clean"].copy()
    attack_df["improvement_pct"] = attack_df["accuracy_improvement"] * 100
    colors = ["green" if v > 0 else "crimson" for v in attack_df["improvement_pct"]]
    bars = ax.barh(attack_df["attack_label"], attack_df["improvement_pct"], color=colors)
    for bar, v in zip(bars, attack_df["improvement_pct"]):
        ax.text(v + (0.5 if v > 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}%", va="center", ha="left" if v > 0 else "right", fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Robust - Baseline (percentage points)")
    ax.set_title("Adversarial Training Improvement by Attack Type")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_robustness_improvement.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '15_robustness_improvement.png'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()