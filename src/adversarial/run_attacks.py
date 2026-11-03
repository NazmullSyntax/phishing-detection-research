"""
Day 6: Run adversarial attacks on best baseline model.

For each attack type:
- Apply to test set URLs
- Re-extract features
- Predict with XGBoost
- Measure degradation (accuracy, precision, recall, F1, FPR)
- Save results

NO URL is visited — pure string manipulation + local model prediction.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from features.extract_features import extract_features
from adversarial.attacks import ATTACKS

TEST_RAW = "data/processed/test.csv"  # has url + label + domain
BEST_MODEL = "models/best_model.pkl"
BEST_META = "experiments/baseline/best_model_metadata.json"

RESULTS_DIR = Path("experiments/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_features_batch(urls, desc=""):
    """Extract features for a list of URLs (batched)."""
    print(f"  Extracting features for {len(urls):,} URLs {desc}...")
    start = time.time()
    rows = [extract_features(u) for u in urls]
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(urls)/elapsed:.0f} urls/sec)")
    return pd.DataFrame(rows)


def evaluate(y_true, y_pred, y_proba):
    """Return a dict of metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "fpr": float(fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    print("=" * 60)
    print("DAY 6: ADVERSARIAL ATTACK EXPERIMENTS")
    print("=" * 60)

    # ---- Load data + model ----
    print("\n[Setup] Loading data + model...")
    test_df = pd.read_csv(TEST_RAW)
    y_test = (test_df["label"] == "phishing").astype(int).values
    print(f"  Test set: {len(test_df):,} URLs")
    print(f"  Phishing: {y_test.sum():,}, Legitimate: {(y_test == 0).sum():,}")

    model = joblib.load(BEST_MODEL)
    with open(BEST_META) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    print(f"  Model: {meta['best_model']}")
    print(f"  Features: {len(feature_cols)}")

    # ---- Baseline (no attack) ----
    print("\n[Baseline] Evaluating original test set...")
    X_base = extract_features_batch(test_df["url"], "(baseline)")
    y_pred_base = model.predict(X_base[feature_cols].values)
    y_proba_base = model.predict_proba(X_base[feature_cols].values)[:, 1]
    baseline_metrics = evaluate(y_test, y_pred_base, y_proba_base)
    baseline_metrics["attack"] = "none"
    baseline_metrics["n_urls"] = len(test_df)
    print(f"  Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Baseline F1:       {baseline_metrics['f1']:.4f}")
    print(f"  Baseline FPR:      {baseline_metrics['fpr']:.4f}")

    all_results = [baseline_metrics]
    attack_details = {}

    # ---- Run each attack ----
    for attack_name, attack_fn in ATTACKS.items():
        print(f"\n[Attack: {attack_name}]")
        attacked_urls = []
        for i, url in enumerate(test_df["url"]):
            attacked = attack_fn(url, seed=i) if "seed" in attack_fn.__code__.co_varnames else attack_fn(url)
            attacked_urls.append(attacked)
        print(f"  Sample: {test_df['url'].iloc[0]}")
        print(f"       → {attacked_urls[0]}")

        X_attacked = extract_features_batch(attacked_urls, f"({attack_name})")
        y_pred = model.predict(X_attacked[feature_cols].values)
        y_proba = model.predict_proba(X_attacked[feature_cols].values)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["attack"] = attack_name
        metrics["n_urls"] = len(attacked_urls)

        print(f"  Accuracy: {metrics['accuracy']:.4f}  (Δ {metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"  F1:       {metrics['f1']:.4f}  (Δ {metrics['f1'] - baseline_metrics['f1']:+.4f})")
        print(f"  FPR:      {metrics['fpr']:.4f}  (Δ {metrics['fpr'] - baseline_metrics['fpr']:+.4f})")

        all_results.append(metrics)

        # Save attack details (sample URLs for analysis)
        attack_details[attack_name] = {
            "sample_original": list(test_df["url"].head(5)),
            "sample_attacked": attacked_urls[:5],
            "total_attacked": len(attacked_urls),
            "n_flipped": int((y_pred != y_pred_base).sum()),
        }

    # ---- Save CSV ----
    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "attack_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved {csv_path}")

    # Also save to results/tables for visualization
    results_df.to_csv(TABLE_DIR / "day6_attack_results.csv", index=False)

    # Save attack details
    details_path = RESULTS_DIR / "attack_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(attack_details, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {details_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results_df[["attack", "accuracy", "precision", "recall", "f1", "fpr"]].to_string(index=False))

    print("\n[Accuracy Degradation from Baseline]")
    base_acc = baseline_metrics["accuracy"]
    for _, row in results_df[results_df["attack"] != "none"].iterrows():
        degradation = base_acc - row["accuracy"]
        print(f"  {row['attack']:<28}: {degradation:+.4f} ({degradation*100:+.2f}%)")

    print("\n" + "=" * 60)
    print("ATTACKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()