"""
Day 6: Adversarial URL manipulation attacks.

All attacks are pure string transformations — no network access, no URL visiting.
Preserves label integrity: legitimate stays legitimate, phishing stays phishing.
"""
import random
from urllib.parse import urlparse

# Unicode homograph mapping (Latin → visually similar Unicode)
HOMOGRAPH_MAP = {
    "a": "\u0430",  # Cyrillic a
    "e": "\u0435",  # Cyrillic e
    "o": "\u043E",  # Cyrillic o
    "p": "\u0440",  # Cyrillic p
    "c": "\u0441",  # Cyrillic c
    "x": "\u0445",  # Cyrillic x
    "y": "\u0443",  # Cyrillic y
    "i": "\u0456",  # Cyrillic i
}

# Character substitution pairs (visually similar)
CHAR_SUBS = {
    "l": "1",  # ell → one
    "o": "0",  # o → zero
    "i": "1",  # i → one
    "e": "3",  # e → three
}

SHORT_URL_DOMAINS = [
    "bit.ly", "t.ly", "tinyurl.com", "shorturl.at",
    "cutt.ly", "is.gd", "buff.ly", "rebrand.ly",
]

MISLEADING_SUBDOMAIN_SUFFIXES = [
    "secure-login", "account-verify", "customer-update",
    "signin-portal", "banking-secure", "auth-confirm",
]


def attack_character_substitution(url: str, max_chars: int = 3, seed: int = 0) -> str:
    """Replace up to N visually-similar chars in domain with lookalikes."""
    try:
        rng = random.Random(seed)
        parsed = urlparse(url)
        netloc = parsed.netloc
        if not netloc or netloc.replace(".", "").replace(":", "").isdigit():
            return url

        chars = list(netloc)
        substitutions = 0
        for i, ch in enumerate(chars):
            if substitutions >= max_chars:
                break
            if ch.lower() in CHAR_SUBS and rng.random() < 0.3:
                chars[i] = CHAR_SUBS[ch.lower()]
                substitutions += 1

        new_netloc = "".join(chars)
        if new_netloc == netloc:
            for i, ch in enumerate(chars):
                if ch.lower() in CHAR_SUBS:
                    chars[i] = CHAR_SUBS[ch.lower()]
                    new_netloc = "".join(chars)
                    break

        return url.replace(netloc, new_netloc, 1)
    except Exception:
        return url


def attack_subdomain_manipulation(url: str, seed: int = 0) -> str:
    """Prepend a misleading subdomain chain."""
    try:
        rng = random.Random(seed)
        suffix = rng.choice(MISLEADING_SUBDOMAIN_SUFFIXES)
        parsed = urlparse(url)
        if not parsed.netloc:
            return url
        new_netloc = f"{parsed.netloc}.{suffix}.xyz"
        return url.replace(parsed.netloc, new_netloc, 1)
    except Exception:
        return url


def attack_unicode_homograph(url: str, max_chars: int = 2, seed: int = 0) -> str:
    """Replace Latin chars in domain with visually-identical Unicode."""
    try:
        rng = random.Random(seed)
        parsed = urlparse(url)
        netloc = parsed.netloc
        if not netloc or netloc.replace(".", "").replace(":", "").isdigit():
            return url

        chars = list(netloc)
        substitutions = 0
        for i, ch in enumerate(chars):
            if substitutions >= max_chars:
                break
            if ch.lower() in HOMOGRAPH_MAP and rng.random() < 0.4:
                chars[i] = HOMOGRAPH_MAP[ch.lower()]
                substitutions += 1

        new_netloc = "".join(chars)
        if new_netloc == netloc:
            for i, ch in enumerate(chars):
                if ch.lower() in HOMOGRAPH_MAP:
                    chars[i] = HOMOGRAPH_MAP[ch.lower()]
                    new_netloc = "".join(chars)
                    break

        return url.replace(netloc, new_netloc, 1)
    except Exception:
        return url


def attack_url_encoding(url: str, encode_ratio: float = 0.3, seed: int = 0) -> str:
    """Percent-encode a random subset of characters in the path."""
    try:
        rng = random.Random(seed)
        parsed = urlparse(url)
        path = parsed.path
        if not path:
            return url
        encoded_path = "".join(
            f"%{ord(ch):02X}" if ch.isalnum() and rng.random() < encode_ratio else ch
            for ch in path
        )
        return url.replace(parsed.path, encoded_path, 1)
    except Exception:
        return url


def attack_url_shortening(url: str, seed: int = 0) -> str:
    """Replace URL with a short-URL placeholder (simulates shortened URL)."""
    try:
        rng = random.Random(seed)
        domain = rng.choice(SHORT_URL_DOMAINS)
        token = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=7))
        return f"https://{domain}/{token}"
    except Exception:
        return url


# Registry for iteration
ATTACKS = {
    "character_substitution": attack_character_substitution,
    "subdomain_manipulation": attack_subdomain_manipulation,
    "unicode_homograph": attack_unicode_homograph,
    "url_encoding": attack_url_encoding,
    "url_shortening": attack_url_shortening,
}


if __name__ == "__main__":
    sample_phishing = "http://paypal.com.verify-login.xyz/account/update"
    sample_legit = "https://www.google.com/search?q=test"

    print("=" * 60)
    print("ATTACK DEMO")
    print("=" * 60)
    for name, attack_fn in ATTACKS.items():
        print(f"\n[{name}]")
        print(f"  Original (phish): {sample_phishing}")
        try:
            attacked_phish = attack_fn(sample_phishing, seed=42)
        except TypeError:
            attacked_phish = attack_fn(sample_phishing)
        print(f"  Attacked:         {attacked_phish}")

        print(f"  Original (legit): {sample_legit}")
        try:
            attacked_legit = attack_fn(sample_legit, seed=42)
        except TypeError:
            attacked_legit = attack_fn(sample_legit)
        print(f"  Attacked:         {attacked_legit}")