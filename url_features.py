"""
URL feature extraction (hybrid model schema).

Matches the hybrid model (Random Forest + XGBoost + Gradient Boosting,
soft-voting ensemble) trained in the URL_Phishing_Detection_Full_Pipeline
notebook (v3).

This is the single source of truth for URL features - used by:
  - backend/main.py           (/scan/url endpoint)
  - backend/email_scanner.py  (URL analysis inside scanned emails)

extract_url_features() returns a dict with keys in FEATURE_ORDER, so it
works correctly whether the caller builds the DataFrame via
model.feature_names_in_ (matches by name) or via plain dict insertion order
(also matches, since FEATURE_ORDER is the insertion order here).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update",
    "confirm", "banking", "signin", "password", "suspend",
]

SHORTENERS = [
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
]

# Must match the training notebook's FEATURE_ORDER exactly, in this order.
FEATURE_ORDER = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
    "HasIPPattern", "HasAtSymbol", "HyphenCount", "DotCount",
    "HasSuspiciousKeyword", "URLEntropy", "UsesShortener",
    "PathLength", "HasPath",
]


def extract_url_features(url: str) -> dict:
    """
    Extract the 27 features used by the ThreatLens hybrid URL model.

    IMPORTANT: this must stay identical to the extract_features() function
    used in the training notebook. If you ever change one, change both, or
    retrain the model - otherwise predictions will silently be wrong (this
    is exactly the bug that broke the earlier version of this model).
    """
    url = str(url)
    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc
    tld = domain.split(".")[-1] if "." in domain else ""

    f = {}
    f["URLLength"] = len(url)
    f["DomainLength"] = len(domain)
    f["IsDomainIP"] = 1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain) else 0
    f["TLDLength"] = len(tld)
    f["NoOfSubDomain"] = max(domain.count(".") - 1, 0)
    f["HasObfuscation"] = 1 if "%" in url else 0
    f["NoOfObfuscatedChar"] = url.count("%")
    f["ObfuscationRatio"] = url.count("%") / len(url) if len(url) > 0 else 0
    f["NoOfLettersInURL"] = sum(c.isalpha() for c in url)
    f["LetterRatioInURL"] = f["NoOfLettersInURL"] / len(url) if len(url) > 0 else 0
    f["NoOfDegitsInURL"] = sum(c.isdigit() for c in url)
    f["DegitRatioInURL"] = f["NoOfDegitsInURL"] / len(url) if len(url) > 0 else 0
    f["NoOfEqualsInURL"] = url.count("=")
    f["NoOfQMarkInURL"] = url.count("?")
    f["NoOfAmpersandInURL"] = url.count("&")

    special_chars = sum(1 for c in url if not c.isalnum() and c not in [".", "/", ":"])
    f["NoOfOtherSpecialCharsInURL"] = special_chars
    f["SpacialCharRatioInURL"] = special_chars / len(url) if len(url) > 0 else 0
    f["IsHTTPS"] = 1 if url.startswith("https://") else 0
    f["HasIPPattern"] = 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", url) else 0
    f["HasAtSymbol"] = 1 if "@" in url else 0
    f["HyphenCount"] = url.count("-")
    f["DotCount"] = url.count(".")
    f["HasSuspiciousKeyword"] = 1 if any(k in url.lower() for k in SUSPICIOUS_KEYWORDS) else 0

    counts = Counter(url)
    probs = [c / len(url) for c in counts.values()] if len(url) > 0 else [0]
    f["URLEntropy"] = -sum(p * math.log2(p) for p in probs if p > 0)
    f["UsesShortener"] = 1 if any(s in url.lower() for s in SHORTENERS) else 0

    path = parsed.path
    f["PathLength"] = len(path)
    f["HasPath"] = 1 if path not in ("", "/") else 0

    return f


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://github.com/anthropics",
        "http://192.168.1.1/login/verify/account",
    ]

    for test_url in test_urls:
        print("\nURL:", test_url)
        feats = extract_url_features(test_url)
        for name in FEATURE_ORDER:
            print(f"  {name:<28}: {feats[name]}")
        print(f"\n  Total features: {len(feats)}")