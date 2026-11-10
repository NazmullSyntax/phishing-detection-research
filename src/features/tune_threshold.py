"""
Day 10 (Fix): Tune classification threshold on validation set.

Goal: Minimize False Positive Rate while keeping Recall >= 92%.
Threshold selected on VAL set (not test) to avoid data leakage.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

VAL = "data/processed/val_features.csv"
MODEL = "models/robust_xgboost.pkl"
META = "experiments/baseline/best_model_metadata.json"

OUT_DIR = Path("experiments/baseline")
TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("THRESHOLD TUNING (on validation set)")
    print("=" * 60)

    # Load model + data
    model = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    val_df = pd.read_csv(VAL)
    X_val = val_df[feature_cols].values
    y_val = (val_df["label"] == "phishing").astype(int).values

    print(f"\nValidation set: {len(y_val):,} URLs")
    print(f"Class balance: {y_val.mean():.3f}")

    # Get probabilities
    proba = model.predict_proba(X_val)[:, 1]

    # Test multiple thresholds
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPR':<10}")
    print("-" * 70)

    results = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "threshold": t,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

        print(f"{t:<10.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {fpr:<10.4f}")

    # Selection criteria: Recall >= 92%, minimize FPR
    df = pd.DataFrame(results)
    candidates = df[df["recall"] >= 0.92].copy()

    if len(candidates) > 0:
        # Among those, pick the one with lowest FPR
        best = candidates.loc[candidates["fpr"].idxmin()]
    else:
        # Fallback: use 0.5
        best = df[df["threshold"] == 0.5].iloc[0]

    print(f"\n{'='*70}")
    print("BEST THRESHOLD (Recall >= 92%, min FPR):")
    print(f"{'='*70}")
    print(f"  Threshold:  {best['threshold']:.2f}")
    print(f"  Accuracy:   {best['accuracy']:.4f}")
    print(f"  Precision:  {best['precision']:.4f}")
    print(f"  Recall:     {best['recall']:.4f}")
    print(f"  F1:         {best['f1']:.4f}")
    print(f"  FPR:        {best['fpr']:.4f}  ← Lower is better")

    # Save
    save_data = {
        "optimal_threshold": float(best["threshold"]),
        "selection_criterion": "Recall >= 92%, minimize FPR",
        "validation_metrics": {
            "accuracy": float(best["accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "fpr": float(best["fpr"]),
        },
    }
    with open(OUT_DIR / "threshold.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[OK] Saved {OUT_DIR / 'threshold.json'}")

    df.to_csv(TABLE_DIR / "day10_threshold_tuning.csv", index=False)
    print(f"[OK] Saved {TABLE_DIR / 'day10_threshold_tuning.csv'}")

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()