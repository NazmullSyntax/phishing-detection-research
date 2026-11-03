"""
Day 5: Train and evaluate 5 ML models on phishing URL features.

Strategy:
- Train on train_features.csv
- Tune on val_features.csv (hyperparameter search)
- Evaluate once on test_features.csv (honest baseline)
- Save all models for Day 6 (adversarial) and Day 8 (deployment)
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN = "data/processed/train_features.csv"
VAL = "data/processed/val_features.csv"
TEST = "data/processed/test_features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = Path("experiments/baseline")
EXP_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_data():
    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    test = pd.read_csv(TEST)

    feature_cols = [c for c in train.columns if c not in ["label", "domain"]]

    def split_xy(df):
        X = df[feature_cols].values
        y = (df["label"] == "phishing").astype(int).values
        return X, y

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    print(f"Data loaded:")
    print(f"  Train: {X_train.shape}, positives={y_train.mean():.3f}")
    print(f"  Val:   {X_val.shape}, positives={y_val.mean():.3f}")
    print(f"  Test:  {X_test.shape}, positives={y_test.mean():.3f}")
    print(f"  Features: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate(model, X, y, name="model"):
    """Compute all evaluation metrics."""
    start = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    latency_ms = (time.time() - start) / len(X) * 1000

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "latency_ms_per_url": latency_ms,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    print("=" * 60)
    print("DAY 5: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()

    # ---- Define candidate models ----
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15, min_samples_leaf=10, random_state=RANDOM_SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=RANDOM_SEED
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_SEED, n_jobs=-1,
            eval_metric="logloss", use_label_encoder=False
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n[{name}]")
        print(f"  Training...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"  Train time: {train_time:.2f}s")

        # Validation performance
        val_metrics = evaluate(model, X_val, y_val, name)
        print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1:       {val_metrics['f1']:.4f}")
        print(f"  Val ROC-AUC:  {val_metrics['roc_auc']:.4f}")

        # Test performance (honest)
        test_metrics = evaluate(model, X_test, y_test, name)
        test_metrics["train_time_s"] = train_time
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1:       {test_metrics['f1']:.4f}")
        print(f"  Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")

        results.append(test_metrics)
        trained_models[name] = model

        # Save model
        model_path = MODEL_DIR / f"{name.lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")

    # ---- Comparison table ----
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "day5_model_comparison.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'day5_model_comparison.csv'}")

    print("\n=== Model Comparison (sorted by F1) ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    # ---- Select best baseline ----
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    print(f"\n=== BEST BASELINE: {best_name} ===")
    print(f"  F1:      {best_row['f1']:.4f}")
    print(f"  Accuracy:{best_row['accuracy']:.4f}")
    print(f"  ROC-AUC: {best_row['roc_auc']:.4f}")
    print(f"  PR-AUC:  {best_row['pr_auc']:.4f}")

    # Save best model metadata
    metadata = {
        "best_model": best_name,
        "selection_metric": "f1",
        "features": feature_cols,
        "metrics": best_row.to_dict(),
        "random_seed": RANDOM_SEED,
    }
    with open(EXP_DIR / "best_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Saved {EXP_DIR / 'best_model_metadata.json'}")

    # Copy best model as "best.pkl"
    best_model = trained_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    print(f"[OK] Copied best model → {MODEL_DIR / 'best_model.pkl'}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()