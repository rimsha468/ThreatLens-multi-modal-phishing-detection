from urllib.parse import urlparse

import tldextract

TRUSTED_DOMAINS = {
    "google.com",
    "wikipedia.org",
    "github.com",
    "youtube.com",
    "microsoft.com",
    "amazon.com",
    "stackoverflow.com",
}


def normalize_url(url: str) -> str:
    """
    Normalize a user-entered URL.

    Examples:
        google.com          -> https://google.com
        www.google.com      -> https://www.google.com
        https://google.com  -> https://google.com
    """
    if not url:
        return ""

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return url


def extract_hostname(url: str) -> str:
    """
    Extract the hostname from a URL.

    Example:
        https://www.google.com/search -> www.google.com
    """
    normalized_url = normalize_url(url)
    if not normalized_url:
        return ""

    try:
        hostname = urlparse(normalized_url).hostname
        return hostname.lower().rstrip(".") if hostname else ""
    except ValueError:
        return ""


def get_registered_domain(url: str) -> str:
    """
    Extract the registered/root domain using tldextract.

    Examples:
        https://google.com                 -> google.com
        https://www.google.com             -> google.com
        https://mail.google.com            -> google.com
        https://example.co.uk              -> example.co.uk
        https://google.com.fake-site.com   -> fake-site.com
    """
    hostname = extract_hostname(url)
    if not hostname:
        return ""

    extracted = tldextract.extract(hostname)
    if not extracted.domain or not extracted.suffix:
        return hostname

    return f"{extracted.domain}.{extracted.suffix}"


def is_trusted_domain(url: str) -> bool:
    """
    Check whether the registered domain belongs to the ThreatLens
    trusted-domain list. Uses exact registered-domain matching.

    Examples:
        google.com                 -> True
        www.google.com             -> True
        mail.google.com            -> True
        google.com.fake-site.com   -> False
        fake-google.com            -> False
    """
    registered_domain = get_registered_domain(url)
    return bool(registered_domain) and registered_domain in TRUSTED_DOMAINS