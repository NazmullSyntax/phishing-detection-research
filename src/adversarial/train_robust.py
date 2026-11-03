"""
Day 7: Train robust model with adversarial augmentation.

Strategy:
- Load original train set (35,801 URLs)
- Generate attacked versions using 5 attacks
- Combine: original + attacked = ~180K training samples
- Train XGBoost on combined data
- Evaluate on clean test AND attacked test
- Compare with baseline XGBoost
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import sys
sys.path.insert(0, "src")
from features.extract_features import extract_features
from adversarial.attacks import ATTACKS

RANDOM_SEED = 42

TRAIN = "data/processed/train.csv"
TEST = "data/processed/test.csv"

OUT_MODEL = "models/robust_xgboost.pkl"
OUT_DIR = Path("experiments/robustness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_TABLE_DIR = Path("results/tables")
FIG_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_batch(urls):
    """Extract features for a list of URLs."""
    return pd.DataFrame([extract_features(u) for u in urls])


def build_attacked_dataset(df, attack_name, attack_fn, use_seed=True):
    """Return list of attacked URLs for a given attack."""
    attacked = []
    for i, url in enumerate(df["url"]):
        if use_seed:
            try:
                attacked.append(attack_fn(url, seed=i))
            except TypeError:
                attacked.append(attack_fn(url))
        else:
            attacked.append(attack_fn(url))
    return attacked


def evaluate(model, X, y, name):
    """Compute classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "eval_set": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "fpr": float(fpr),
    }


def main():
    print("=" * 60)
    print("DAY 7: ADVERSARIAL TRAINING")
    print("=" * 60)

    # ---- Load original training data ----
    print("\n[1] Loading training data...")
    train_df = pd.read_csv(TRAIN)
    print(f"  Original train: {len(train_df):,} URLs")
    y_train_orig = (train_df["label"] == "phishing").astype(int).values

    # ---- Original features ----
    print("\n[2] Extracting features for original train...")
    X_train_orig = extract_batch(train_df["url"])
    print(f"  Shape: {X_train_orig.shape}")

    # ---- Generate attacked versions ----
    print("\n[3] Generating attacked versions...")
    all_X = [X_train_orig]
    all_y = [y_train_orig]
    feature_cols = list(X_train_orig.columns)

    for attack_name, attack_fn in ATTACKS.items():
        print(f"  Generating attack: {attack_name}...")
        attacked_urls = build_attacked_dataset(train_df, attack_name, attack_fn)
        X_attacked = extract_batch(attacked_urls)
        all_X.append(X_attacked)
        all_y.append(y_train_orig)  # labels preserved
        print(f"    Added: {len(X_attacked):,} rows")

    X_train_aug = pd.concat(all_X, ignore_index=True).values
    y_train_aug = np.concatenate(all_y)
    print(f"\n  Total augmented training: {X_train_aug.shape[0]:,} rows x {X_train_aug.shape[1]} features")
    print(f"  Class balance: phishing={y_train_aug.mean():.3f}")

    # ---- Train robust model ----
    print("\n[4] Training robust XGBoost...")
    start = time.time()
    robust_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    robust_model.fit(X_train_aug, y_train_aug)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")

    # Save model
    joblib.dump(robust_model, OUT_MODEL)
    print(f"  [OK] Saved: {OUT_MODEL}")

    # ---- Evaluate ----
    print("\n[5] Loading test set...")
    test_df = pd.read_csv(TEST)
    y_test = (test_df["label"] == "phishing").astype(int).values

    print("\n[6] Evaluating on CLEAN test set...")
    X_test_clean = extract_batch(test_df["url"])
    baseline_model = joblib.load("models/best_model.pkl")

    clean_baseline = evaluate(baseline_model, X_test_clean[feature_cols].values, y_test, "clean")
    clean_robust = evaluate(robust_model, X_test_clean[feature_cols].values, y_test, "clean")
    print(f"  Baseline clean accuracy: {clean_baseline['accuracy']:.4f}")
    print(f"  Robust   clean accuracy: {clean_robust['accuracy']:.4f}")
    print(f"  Trade-off: {(clean_robust['accuracy'] - clean_baseline['accuracy'])*100:+.2f}%")

    # ---- Evaluate on attacked test sets ----
    print("\n[7] Evaluating on ATTACKED test sets...")
    comparison = []

    # Add clean rows first
    comparison.append({
        "attack": "clean",
        "baseline_accuracy": clean_baseline["accuracy"],
        "robust_accuracy": clean_robust["accuracy"],
        "baseline_f1": clean_baseline["f1"],
        "robust_f1": clean_robust["f1"],
        "baseline_fpr": clean_baseline["fpr"],
        "robust_fpr": clean_robust["fpr"],
        "accuracy_improvement": clean_robust["accuracy"] - clean_baseline["accuracy"],
    })

    for attack_name, attack_fn in ATTACKS.items():
        print(f"\n  [{attack_name}]")
        attacked_urls = build_attacked_dataset(test_df, attack_name, attack_fn)
        X_test_attacked = extract_batch(attacked_urls)

        base_m = evaluate(baseline_model, X_test_attacked[feature_cols].values, y_test, attack_name)
        rob_m = evaluate(robust_model, X_test_attacked[feature_cols].values, y_test, attack_name)

        print(f"    Baseline: {base_m['accuracy']:.4f}")
        print(f"    Robust:   {rob_m['accuracy']:.4f}")
        print(f"    Improvement: {(rob_m['accuracy'] - base_m['accuracy'])*100:+.2f}%")

        comparison.append({
            "attack": attack_name,
            "baseline_accuracy": base_m["accuracy"],
            "robust_accuracy": rob_m["accuracy"],
            "baseline_f1": base_m["f1"],
            "robust_f1": rob_m["f1"],
            "baseline_fpr": base_m["fpr"],
            "robust_fpr": rob_m["fpr"],
            "accuracy_improvement": rob_m["accuracy"] - base_m["accuracy"],
        })

    # ---- Save comparison ----
    comp_df = pd.DataFrame(comparison)
    comp_path = OUT_DIR / "baseline_vs_robust.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[OK] Saved {comp_path}")

    # Also save to results/tables for figures
    comp_df.to_csv(FIG_TABLE_DIR / "day7_baseline_vs_robust.csv", index=False)

    # Save metadata
    metadata = {
        "robust_model": "XGBoost",
        "training_samples_original": len(train_df),
        "training_samples_augmented": int(X_train_aug.shape[0]),
        "training_time_s": float(train_time),
        "clean_accuracy_baseline": clean_baseline["accuracy"],
        "clean_accuracy_robust": clean_robust["accuracy"],
        "clean_tradeoff": clean_robust["accuracy"] - clean_baseline["accuracy"],
    }
    with open(OUT_DIR / "robust_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved {OUT_DIR / 'robust_model_metadata.json'}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: Baseline vs Robust")
    print("=" * 60)
    print(comp_df[["attack", "baseline_accuracy", "robust_accuracy", "accuracy_improvement"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("DAY 7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()