"""
Day 4: Feature EDA.
- Correlation heatmap
- Feature distributions by class (violin plots)
- Quick Random Forest importance preview
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FEATURES_TRAIN = "data/processed/train_features.csv"
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def main():
    print("=" * 60)
    print("DAY 4: FEATURE EDA")
    print("=" * 60)

    df = pd.read_csv(FEATURES_TRAIN)
    print(f"Loaded: {df.shape}")

    # Separate label + feature columns
    feature_cols = [c for c in df.columns if c not in ["label", "domain"]]
    print(f"Features: {len(feature_cols)}")

    # ---- Plot 1: Correlation heatmap ----
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap (21 features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_feature_correlation.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '06_feature_correlation.png'}")

    # ---- Plot 2: Feature distributions by class (violin) ----
    # Use top 8 most informative features based on a quick check
    plot_features = [
        "url_length", "num_digits", "entropy", "num_hyphens",
        "suspicious_keywords_count", "num_special_chars",
        "num_subdomains", "domain_length",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(plot_features):
        sns.violinplot(
            data=df, x="label", y=feat, ax=axes[i], cut=0, inner="quartile"
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel("")
    fig.suptitle("Feature Distributions by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_distributions.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '07_feature_distributions.png'}")

    # ---- Quick RF importance preview ----
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = df[feature_cols].values
    y = (df["label"] == "phishing").astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    score = rf.score(X_te, y_te)
    print(f"[Preview] Quick RF accuracy: {score:.4f}")

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importances.to_csv(TABLE_DIR / "day4_feature_importance_preview.csv", index=False)
    print(f"[OK] Saved {TABLE_DIR / 'day4_feature_importance_preview.csv'}")
    print("\nTop 10 features:")
    print(importances.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("FEATURE EDA DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()