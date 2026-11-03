"""
Day 5: Visualization of model results.
- Confusion matrices (all 5 models)
- ROC curves (all 5 models)
- PR curves (all 5 models)
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)

TEST = "data/processed/test_features.csv"
MODEL_DIR = Path("models")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def main():
    print("=" * 60)
    print("DAY 5: VISUALIZE MODEL RESULTS")
    print("=" * 60)

    test = pd.read_csv(TEST)
    with open("experiments/baseline/best_model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    X_test = test[feature_cols].values
    y_test = (test["label"] == "phishing").astype(int).values

    model_names = ["LogisticRegression", "DecisionTree", "RandomForest", "SVM", "XGBoost"]
    models = {}
    for name in model_names:
        path = MODEL_DIR / f"{name.lower()}.pkl"
        models[name] = joblib.load(path)
        print(f"  Loaded {name}")

    # ---- Plot 1: Confusion matrices ----
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, name in enumerate(model_names):
        y_pred = models[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["legit", "phish"], yticklabels=["legit", "phish"])
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    fig.suptitle("Confusion Matrices on Test Set", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_confusion_matrices.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '08_confusion_matrices.png'}")

    # ---- Plot 2: ROC curves ----
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in model_names:
        y_proba = models[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_roc_curves.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '09_roc_curves.png'}")

    # ---- Plot 3: PR curves ----
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in model_names:
        y_proba = models[name].predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_pr_curves.png")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR / '10_pr_curves.png'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()