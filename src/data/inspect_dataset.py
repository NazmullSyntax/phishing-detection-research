"""
Inspect raw phishing dataset: shape, columns, class balance, duplicates, sample URLs.
"""
import pandas as pd

RAW_PATH = "data/raw/phishing_raw.csv"

def main():
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    # Load
    df = pd.read_csv(RAW_PATH)
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Columns
    print(f"\n📋 Columns: {list(df.columns)}")

    # First 5 rows
    print(f"\n🔍 First 5 rows:")
    print(df.head())

    # Missing values
    print(f"\n❓ Missing values per column:")
    print(df.isnull().sum())

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n🔁 Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")

    # Detect label column
    label_col = None
    for col in df.columns:
        if df[col].nunique() == 2 and df[col].dtype in ['int64', 'object', 'bool']:
            label_col = col
            break

    if label_col:
        print(f"\n🏷️  Label column detected: '{label_col}'")
        print(f"\n⚖️  Class distribution:")
        print(df[label_col].value_counts())
        print(f"\n   Percentage:")
        print(df[label_col].value_counts(normalize=True) * 100)

    # Sample URLs
    url_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if values look like URLs
            sample = str(df[col].iloc[0])
            if 'http' in sample or '.' in sample:
                url_col = col
                break

    if url_col:
        print(f"\n🔗 URL column detected: '{url_col}'")
        print(f"\n📝 Sample URLs (5 random):")
        for i, url in enumerate(df[url_col].sample(5, random_state=42).tolist(), 1):
            print(f"   {i}. {url[:100]}{'...' if len(str(url)) > 100 else ''}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()