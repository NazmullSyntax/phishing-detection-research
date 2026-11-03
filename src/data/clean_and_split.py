"""
Day 3: Clean, sample, and split the phishing URL dataset.

Steps:
1. Load raw dataset
2. URL validation (basic syntactic checks)
3. Domain extraction
4. Domain-label conflict detection
5. Stratified sample 25K phishing + 25K legitimate
6. Domain-grouped split: train/val/test (70/15/15)
7. Save processed files
"""
import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

RAW_PATH = "data/raw/phishing_raw.csv"
OUT_TRAIN = "data/processed/train.csv"
OUT_VAL = "data/processed/val.csv"
OUT_TEST = "data/processed/test.csv"

RANDOM_SEED = 42
SAMPLE_PER_CLASS = 25_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def extract_domain(url: str):
    """Return the registered domain (naive: netloc without www/port)."""
    try:
        parsed = urlparse(str(url).strip())
        netloc = parsed.netloc.lower()
        # Remove port
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # Remove www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_valid_url(url: str) -> bool:
    """Basic syntactic URL check (no network access)."""
    if not isinstance(url, str) or len(url) < 4:
        return False
    url = url.strip()
    # Must contain a dot
    if "." not in url:
        return False
    # Must have a plausible scheme or start with domain
    if url.startswith(("http://", "https://", "ftp://")):
        return len(url) > 10
    # Bare domain like "example.com"
    return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}", url))


def main():
    print("=" * 60)
    print("DAY 3: CLEANING & SPLITTING")
    print("=" * 60)

    # --- 1. Load raw ---
    df = pd.read_csv(RAW_PATH)
    print(f"\n[1] Raw: {len(df):,} rows, {df.columns.tolist()}")

    # --- 2. Validate URL syntax ---
    df["is_valid"] = df["url"].apply(is_valid_url)
    invalid = (~df["is_valid"]).sum()
    print(f"[2] Invalid URLs: {invalid:,} ({invalid/len(df)*100:.2f}%) — dropped")
    df = df[df["is_valid"]].reset_index(drop=True)
    print(f"    Remaining: {len(df):,}")

    # --- 3. Domain extraction ---
    df["domain"] = df["url"].apply(extract_domain)
    empty_domain = (df["domain"] == "").sum()
    print(f"[3] Empty domains: {empty_domain:,} — dropped")
    df = df[df["domain"] != ""].reset_index(drop=True)
    print(f"    Unique domains: {df['domain'].nunique():,}")

    # --- 4. Domain-label conflicts ---
    domain_labels = df.groupby("domain")["label"].nunique()
    conflicting_domains = set(domain_labels[domain_labels > 1].index)
    print(f"[4] Domains with BOTH labels: {len(conflicting_domains):,} — dropped")
    df = df[~df["domain"].isin(conflicting_domains)].reset_index(drop=True)
    print(f"    Remaining: {len(df):,}")

    # --- 5. Sample 25K + 25K ---
    print(f"\n[5] Sampling {SAMPLE_PER_CLASS:,} per class...")
    phishing = df[df["label"] == "phishing"]
    legit = df[df["label"] == "legitimate"]

    n_phishing = min(SAMPLE_PER_CLASS, len(phishing))
    n_legit = min(SAMPLE_PER_CLASS, len(legit))
    print(f"    Available — phishing: {len(phishing):,}, legit: {len(legit):,}")

    sample_phishing = phishing.sample(n=n_phishing, random_state=RANDOM_SEED)
    sample_legit = legit.sample(n=n_legit, random_state=RANDOM_SEED)

    df_sample = pd.concat([sample_phishing, sample_legit]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    print(f"    Sampled: {len(df_sample):,} rows")
    print(f"    Distribution: {df_sample['label'].value_counts().to_dict()}")

    # --- 6. Domain-grouped split ---
    print(f"\n[6] Domain-grouped split (70/15/15)...")
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    train_idx, temp_idx = next(gss1.split(df_sample, groups=df_sample["domain"]))
    train_df = df_sample.iloc[train_idx].reset_index(drop=True)
    temp_df = df_sample.iloc[temp_idx].reset_index(drop=True)

    # Split temp into val + test (equal halves)
    val_frac_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=(1 - val_frac_of_temp), random_state=RANDOM_SEED
    )
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["domain"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"    Train: {len(train_df):,} (domains: {train_df['domain'].nunique():,})")
    print(f"    Val:   {len(val_df):,} (domains: {val_df['domain'].nunique():,})")
    print(f"    Test:  {len(test_df):,} (domains: {test_df['domain'].nunique():,})")

    # --- 7. Verify NO domain overlap ---
    train_domains = set(train_df["domain"])
    val_domains = set(val_df["domain"])
    test_domains = set(test_df["domain"])
    overlap_tv = len(train_domains & val_domains)
    overlap_tt = len(train_domains & test_domains)
    overlap_vt = len(val_domains & test_domains)
    print(f"\n[7] Domain overlap check (should be 0):")
    print(f"    Train ∩ Val:  {overlap_tv}")
    print(f"    Train ∩ Test: {overlap_tt}")
    print(f"    Val ∩ Test:   {overlap_vt}")
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, \
        "DOMAIN OVERLAP DETECTED — data leakage!"

    # --- 8. Save ---
    # Drop helper columns except domain (we keep it for later use)
    keep_cols = ["url", "label", "domain"]
    train_df[keep_cols].to_csv(OUT_TRAIN, index=False)
    val_df[keep_cols].to_csv(OUT_VAL, index=False)
    test_df[keep_cols].to_csv(OUT_TEST, index=False)
    print(f"\n[8] Saved:")
    print(f"    {OUT_TRAIN}")
    print(f"    {OUT_VAL}")
    print(f"    {OUT_TEST}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()