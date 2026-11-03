# Processed Dataset Documentation

## Overview

The raw 253,098-URL dataset was cleaned, sampled, and split into train/val/test sets using **domain-grouped splitting** to prevent data leakage.

## Processing Pipeline

| Step | Action | Records Affected |
|---|---|---|
| 1 | Load raw | 253,098 |
| 2 | Drop invalid URLs (basic syntax check) | -19 |
| 3 | Drop URLs with empty domain | -0 |
| 4 | Drop domains with conflicting labels (100 domains) | -11,667 |
| 5 | Stratified sample 25,000 per class | → 50,000 |
| 6 | Domain-grouped split (70/15/15) | → 35,801 / 7,515 / 6,684 |
| 7 | Verify domain overlap across splits | 0 (all pairs) |

## Final Split Summary

| Split | Rows | Unique Domains |
|---|---|---|
| Train | 35,801 | 22,230 |
| Val | 7,515 | 4,764 |
| Test | 6,684 | 4,764 |
| **Total** | **50,000** | — |

Each split is approximately 50/50 phishing/legitimate (25,000 per class total).

## Files

| File | Purpose |
|---|---|
| `train.csv` | Model training |
| `val.csv` | Hyperparameter tuning, early stopping |
| `test.csv` | Final evaluation (untouched until Day 5) |

## Columns

| Column | Type | Description |
|---|---|---|
| `url` | string | Full URL string |
| `label` | string | `"phishing"` or `"legitimate"` |
| `domain` | string | Extracted registered domain (used for group splits) |

## Data Leakage Prevention

The **same domain is never present in more than one split**. Verified:
- Train ∩ Val domains = 0
- Train ∩ Test domains = 0
- Val ∩ Test domains = 0

**Rationale:** Prevents models from memorizing a domain's label in training and getting artificially inflated accuracy on test. This is a **contribution** of this project — most published phishing datasets do not report domain-grouped splits.

## Random Seed

All sampling and splitting use `random_state=42` for exact reproducibility.

## Regeneration

```bash
python src/data/clean_and_split.py