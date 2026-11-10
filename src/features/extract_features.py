"""
Day 4 + Fix v2: URL Feature Extraction.

Extracts 26 lexical features from URL strings (no network access).
Version 2: Reduced keyword aggressiveness, added TLD + port signals.
"""
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

# Suspicious keywords (kept, but used with normalization)
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "signin", "bank", "paypal", "amazon", "apple", "microsoft",
    "google", "password", "credential", "urgent", "alert",
    "suspended", "locked", "unlock", "recovery", "support",
]

# TLDs associated with phishing (cheap/free registrations)
SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "website", "space",
    "live", "icu", "buzz", "monster", "quest", "cyou", "sbs",
    "click", "link", "gq", "tk", "ml", "ga", "cf", "work", "party",
}

# Trusted TLDs
TRUSTED_TLDS = {
    "com", "org", "edu", "gov", "net", "io", "co", "us", "uk",
}

# Suspicious file extensions in path
SUSPICIOUS_EXTENSIONS = {".php", ".exe", ".bat", ".cmd", ".scr", ".vbs", ".js"}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_tld(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        return parts[-1] if len(parts) >= 2 else ""
    except Exception:
        return ""


def count_subdomains(url: str) -> int:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def has_ip_address(url: str) -> int:
    try:
        netloc = urlparse(url).netloc.split(":")[0]
        parts = netloc.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return 1
    except Exception:
        pass
    return 0


def has_custom_port(url: str) -> int:
    try:
        netloc = urlparse(url).netloc
        if ":" in netloc:
            port = netloc.split(":")[-1]
            if port.isdigit() and int(port) not in (80, 443, 8080):
                return 1
    except Exception:
        pass
    return 0


def count_suspicious_keywords(url: str) -> int:
    url_lower = url.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)


def path_has_suspicious_ext(url: str) -> int:
    try:
        path = urlparse(url).path.lower()
        return int(any(path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS))
    except Exception:
        return 0


def extract_features(url: str) -> dict:
    """Extract 26 lexical features from a URL string."""
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

    kw_count = count_suspicious_keywords(url)
    url_len = max(1, len(url))
    keyword_density = kw_count / (url_len / 10.0)

    features = {
        # A. Length & Size (6)
        "url_length": len(url),
        "domain_length": len(netloc),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "tld_length": len(tld),

        # B. Character Counts (7)
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_slashes": url.count("/"),
        "num_special_chars": sum(c in "@?=&%#;" for c in url),
        "num_uppercase": sum(c.isupper() for c in url),
        "num_percent_encoding": len(re.findall(r"%[0-9A-Fa-f]{2}", url)),

        # C. Structural (8)
        "num_subdomains": count_subdomains(url),
        "has_https": int(url.lower().startswith("https://")),
        "has_ip_address": has_ip_address(url),
        "has_at_symbol": int("@" in url),
        "has_double_slash_in_path": int("//" in path),
        "path_depth": path.count("/"),
        "has_custom_port": has_custom_port(url),
        "path_has_suspicious_ext": path_has_suspicious_ext(url),

        # D. Content & TLD (5)
        "keyword_density": round(keyword_density, 4),
        "entropy": shannon_entropy(url),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "trusted_tld": int(tld in TRUSTED_TLDS),
        "has_suspicious_keyword": int(kw_count > 0),
    }
    return features


def extract_dataframe(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    feature_rows = [extract_features(u) for u in df[url_col]]
    features_df = pd.DataFrame(feature_rows)
    keep_cols = [c for c in ["label", "domain"] if c in df.columns]
    return pd.concat([df[keep_cols].reset_index(drop=True), features_df], axis=1)


def main():
    """Re-extract features for all splits (train, val, test)."""
    print("=" * 60)
    print("DAY 4: FEATURE EXTRACTION (v2 - 26 features)")
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
        print(f"    feature count: {features_df.shape[1] - 2}")  # minus label + domain

    print("\n" + "=" * 60)
    print("DONE — feature extraction complete")
    print("=" * 60)


if __name__ == "__main__":
    main()