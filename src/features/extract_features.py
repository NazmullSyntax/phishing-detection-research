"""
Day 4: URL Feature Extraction.

Extracts 21 lexical features from URL strings (no network access).
All features are derived purely from the URL string — safe, offline, fast.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from pathlib import Path

# Suspicious keywords commonly used in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "signin", "bank", "paypal", "amazon", "apple", "microsoft",
    "google", "password", "credential", "urgent", "alert",
    "suspended", "locked", "unlock", "recovery", "support",
]

# Top-level domains commonly associated with phishing
SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "website", "space",
    "live", "icu", "buzz", "monster", "quest", "cyou", "sbs",
}


def shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_tld(url: str) -> str:
    """Extract top-level domain (last dot segment, ignoring trailing slashes)."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(":")[0]
        # Strip www
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        return parts[-1] if len(parts) >= 2 else ""
    except Exception:
        return ""


def count_subdomains(url: str) -> int:
    """Number of subdomains (excluding root domain and TLD)."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        # e.g., "a.b.example.com" -> 2 subdomains (a, b)
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def has_ip_address(url: str) -> int:
    """1 if the netloc contains an IP address, else 0."""
    try:
        netloc = urlparse(url).netloc.split(":")[0]
        parts = netloc.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return 1
    except Exception:
        pass
    return 0


def count_suspicious_keywords(url: str) -> int:
    """Count suspicious keyword occurrences in the URL."""
    url_lower = url.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)


def extract_features(url: str) -> dict:
    """Extract 21 lexical features from a URL string."""
    url = str(url).strip()
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    netloc = parsed.netloc.lower().split(":")[0]
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment
    tld = extract_tld(url)

    features = {
        # A. Length & Size
        "url_length": len(url),
        "domain_length": len(netloc),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "tld_length": len(tld),

        # B. Character Counts
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_slashes": url.count("/"),
        "num_special_chars": sum(c in "@?=&%#;" for c in url),
        "num_uppercase": sum(c.isupper() for c in url),
        "num_percent_encoding": len(re.findall(r"%[0-9A-Fa-f]{2}", url)),

        # C. Structural
        "num_subdomains": count_subdomains(url),
        "has_https": int(url.lower().startswith("https://")),
        "has_ip_address": has_ip_address(url),
        "has_at_symbol": int("@" in url),
        "has_double_slash_in_path": int("//" in path),
        "path_depth": path.count("/"),

        # D. Content
        "suspicious_keywords_count": count_suspicious_keywords(url),
        "entropy": shannon_entropy(url),
    }
    return features


def extract_dataframe(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """Extract features for all rows; concatenate with label + domain."""
    feature_rows = [extract_features(u) for u in df[url_col]]
    features_df = pd.DataFrame(feature_rows)
    # Keep label + domain for later grouping/evaluation
    keep_cols = [c for c in ["label", "domain"] if c in df.columns]
    return pd.concat([df[keep_cols].reset_index(drop=True), features_df], axis=1)


def main():
    print("=" * 60)
    print("DAY 4: FEATURE EXTRACTION")
    print("=" * 60)

    splits = {
        "train": "data/processed/train.csv",
        "val": "data/processed/val.csv",
        "test": "data/processed/test.csv",
    }
    out_dir = Path("data/processed")

    for name, path in splits.items():
        print(f"\n[{name}] loading {path}")
        df = pd.read_csv(path)
        print(f"    rows: {len(df):,}")

        features_df = extract_dataframe(df)
        out_path = out_dir / f"{name}_features.csv"
        features_df.to_csv(out_path, index=False)
        print(f"    -> saved {out_path} ({features_df.shape[0]:,} rows x {features_df.shape[1]} cols)")
        print(f"    feature columns: {list(features_df.columns[2:])}")

    print("\n" + "=" * 60)
    print("DONE — feature extraction complete")
    print("=" * 60)


if __name__ == "__main__":
    main()