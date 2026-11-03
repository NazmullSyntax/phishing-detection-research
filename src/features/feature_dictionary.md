# Feature Dictionary

All features are extracted **lexically from the URL string only**. No network requests, no DNS lookups, no content fetching. This keeps the pipeline fast, safe, and deployable.

## Group A — Length & Size Features

| Feature | Type | Formula / Rule | Rationale |
|---|---|---|---|
| `url_length` | int | Total characters in URL | Phishing URLs average ~19% longer |
| `domain_length` | int | Characters in netloc (excluding port) | Long domains are suspicious |
| `path_length` | int | Characters in path after domain | Fake admin panels use long paths |
| `query_length` | int | Characters after `?` | Query-based obfuscation |
| `fragment_length` | int | Characters after `#` | Rare in phishing, useful when present |
| `tld_length` | int | Length of TLD (`.com`, `.support`) | Long/new TLDs are suspicious |

## Group B — Character Count Features

| Feature | Type | Formula / Rule | Rationale |
|---|---|---|---|
| `num_dots` | int | Count of `.` | Subdomain chains |
| `num_hyphens` | int | Count of `-` | `paypal-login-secure.com` pattern |
| `num_digits` | int | Count of `0-9` | Phishing URLs have 2.4x more digits |
| `num_slashes` | int | Count of `/` | Path depth |
| `num_special_chars` | int | Count of `@ ? = & % # ;` | Obfuscation indicator |
| `num_uppercase` | int | Count of `A-Z` | Rare in legit URLs |
| `num_percent_encoding` | int | Count of `%XX` sequences | URL-encoded obfuscation |

## Group C — Structural Features

| Feature | Type | Formula / Rule | Rationale |
|---|---|---|---|
| `num_subdomains` | int | Subdomain parts before root domain | `a.b.c.d.evil.com` chain |
| `has_https` | 0/1 | Starts with `https://` | Legit sites more likely HTTPS |
| `has_ip_address` | 0/1 | Netloc matches `a.b.c.d` with 0–255 | Classic phishing trick |
| `has_at_symbol` | 0/1 | Contains `@` | `user@evil.com` URL spoofing |
| `has_double_slash_in_path` | 0/1 | Contains `//` in path | Redirect-based obfuscation |
| `path_depth` | int | Count of `/` in path | Deep path mimics admin URLs |

## Group D — Content Features

| Feature | Type | Formula / Rule | Rationale |
|---|---|---|---|
| `suspicious_keywords_count` | int | Count of 22 keywords (login, verify, etc.) | Classic phishing indicator |
| `entropy` | float | Shannon entropy of URL string | Random chars (`xk29fp`) = high entropy |

## Notes on Feature Engineering

- **No SMOTE** at this stage — data is already balanced 25K/25K.
- **No target encoding** — we avoid label leakage.
- **No domain features** based on external data (DNS, WHOIS) — keeps pipeline safe + deployable.
- **Extraction speed:** ~2,000 URLs/sec single-threaded, well within real-time deployment budget.

## Known Limitations

1. **Does not handle subdomain-aware TLDs** (e.g., `co.uk` treated as TLD `uk`). Acceptable for research; public-suffix-list is future work.
2. **`suspicious_keywords_count`** can miss new brand names; keyword list is curated manually.
3. **Entropy** for very short URLs unstable (n < 5 chars) — negligible given dataset's URL length distribution.