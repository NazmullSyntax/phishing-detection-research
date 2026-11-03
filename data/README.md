# Dataset Documentation

## Source

**Primary Dataset:** Phishing URL Dataset (Kaggle)
- **URL:** https://www.kaggle.com/datasets/dattatejaofficial/phishing-url-dataset
- **Author:** Datta Teja
- **Accessed:** 2026-11-04
- **Format:** CSV (`raw_data.csv`)
- **Size:** 253,098 unique URLs (~18 MB)
- **Original Sources:** PhishTank, IEEE data repositories
- **License:** Refer to Kaggle dataset page

## Dataset Description

The dataset contains labeled URLs categorized as either phishing or legitimate:
- **Phishing URLs:** Sourced from verified reports of malicious activity (PhishTank)
- **Legitimate URLs:** Collected from trusted benign web sources

## Statistics (Raw Dataset)

| Metric | Value |
|---|---|
| Total rows | 253,098 |
| Total columns | TBD (from inspection) |
| Phishing class | ~49% |
| Legitimate class | ~51% |
| Missing values | TBD |
| Duplicate rows | TBD |

*(Statistics will be updated after full inspection in Day 3)*

## Files

| File | Description |
|---|---|
| `raw/phishing_raw.csv` | Original downloaded dataset (not pushed to GitHub) |
| `raw/.gitkeep` | Ensures raw folder exists in repo |
| `processed/.gitkeep` | Ensures processed folder exists in repo |

## Download Instructions (for reproducibility)

1. Create a free Kaggle account: https://www.kaggle.com
2. Navigate to: https://www.kaggle.com/datasets/dattatejaofficial/phishing-url-dataset
3. Click "Download" → save `archive.zip`
4. Extract → locate `raw_data.csv`
5. Place at: `data/raw/phishing_raw.csv`

## Why This Dataset?

1. **Balanced:** ~49/51 phishing/legit ratio eliminates need for SMOTE before splitting
2. **Single file:** Simple, reproducible loading
3. **Verified sources:** Phishing URLs from PhishTank (community-verified)
4. **Size sufficient:** 253K allows us to sample 50K for experiments and later scale up

## Known Limitations

1. **Static dataset:** Cannot capture newly registered phishing domains after creation
2. **English/Latin-script bias:** May underrepresent regional phishing
3. **No temporal metadata:** Cannot perform temporal splitting
4. **No adversarial examples:** Contains only "natural" URLs, not manipulated ones (our contribution will address this)

## Data Safety Rules

- ✅ We analyze URLs as **strings only**
- ❌ We never visit or crawl any URL from this dataset
- ❌ We never execute any content from these URLs
- ✅ All analysis is offline

## Ethical Use

This dataset is used solely for academic research on defensive phishing detection. No offensive use is intended.

## References

- Kaggle dataset: https://www.kaggle.com/datasets/dattatejaofficial/phishing-url-dataset