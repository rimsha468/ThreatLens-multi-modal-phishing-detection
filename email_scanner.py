import joblib
import pandas as pd
from urlextract import URLExtract

from backend.url_features import extract_url_features, FEATURE_ORDER as URL_FEATURE_NAMES
from backend.hybrid_url_model import HybridURLModel

EMAIL_MODEL_PATH = "models/phishing_email_model.pkl"

# Load trained models once at import time
email_model = joblib.load(EMAIL_MODEL_PATH)
url_model = HybridURLModel.load("models")
url_extractor = URLExtract()

# Keyword lists used to flag common phishing patterns
URGENCY_WORDS = [
    "urgent", "immediately", "action required", "act now", "final warning",
    "account will be suspended", "account will be closed",
    "within 24 hours", "within 48 hours",
]

CREDENTIAL_WORDS = [
    "password", "passcode", "login", "username", "verify your identity",
    "security code", "otp", "one-time password",
]

FINANCIAL_WORDS = [
    "payment", "credit card", "debit card", "bank account", "billing",
    "invoice", "transaction", "refund", "wire transfer",
]

ACCOUNT_WORDS = [
    "verify your account", "confirm your account", "account verification",
    "account suspended", "account locked", "security alert",
]


def find_matches(text, keywords):
    """Return which keywords appear in the text."""
    text = text.lower()
    return [kw for kw in keywords if kw in text]


def analyze_email_indicators(subject, body):
    """Scan subject/body for common phishing-related red flags."""
    text = f"{subject or ''} {body or ''}".lower()
    indicators = []

    if matches := find_matches(text, URGENCY_WORDS):
        indicators.append({
            "type": "urgency",
            "severity": "medium",
            "message": "The email uses urgent or threatening language.",
            "matches": matches,
        })

    if matches := find_matches(text, CREDENTIAL_WORDS):
        indicators.append({
            "type": "credential_request",
            "severity": "high",
            "message": "The email appears to request login or security credentials.",
            "matches": matches,
        })

    if matches := find_matches(text, FINANCIAL_WORDS):
        indicators.append({
            "type": "financial_request",
            "severity": "high",
            "message": "The email contains financial or payment-related language.",
            "matches": matches,
        })

    if matches := find_matches(text, ACCOUNT_WORDS):
        indicators.append({
            "type": "account_request",
            "severity": "medium",
            "message": "The email contains account verification or security-related language.",
            "matches": matches,
        })

    urls = url_extractor.find_urls(body or "")
    if len(urls) >= 3:
        indicators.append({
            "type": "multiple_urls",
            "severity": "medium",
            "message": "The email contains multiple URLs.",
            "matches": urls,
        })

    return indicators


def extract_urls(text):
    """Extract and clean up URLs found in the email body."""
    if not text:
        return []

    clean_urls = []
    for url in url_extractor.find_urls(text):
        url = url.rstrip(".,!?;:")

        # If extraction grabbed leading junk, trim back to the actual http(s) part
        pos = url.find("https://")
        if pos == -1:
            pos = url.find("http://")
        if pos > 0:
            url = url[pos:]

        if url and url not in clean_urls:
            clean_urls.append(url)

    return clean_urls


def analyze_url(url):
    """Run a single URL through the trained URL model."""
    features = extract_url_features(url)
    feature_data = pd.DataFrame([features], columns=URL_FEATURE_NAMES)

    prediction = int(url_model.predict(feature_data)[0])
    probabilities = url_model.predict_proba(feature_data)[0]
    risk_score = float(probabilities[1] * 100)

    return {
        "url": url,
        "prediction": prediction,
        "classification": "Phishing" if prediction == 0 else "Legitimate",
        "risk_score": round(risk_score, 2),
    }


def calculate_indicator_score(indicators):
    """Turn the list of flagged indicators into a 0-100 score."""
    if not indicators:
        return 0.0

    severity_points = {"low": 10, "medium": 20, "high": 30}
    score = sum(severity_points.get(ind.get("severity", "low"), 0) for ind in indicators)
    return min(score, 100.0)


def calculate_combined_email_risk(email_risk, url_analysis, indicators):
    """
    Blend the email ML score, any URL risk found in the body, and
    keyword-based indicators into one final risk score/level.
    """
    indicator_score = calculate_indicator_score(indicators)
    highest_url_risk = max((item["risk_score"] for item in url_analysis), default=None)

    if highest_url_risk is not None:
        base_score = email_risk * 0.65 + highest_url_risk * 0.20 + indicator_score * 0.15
    else:
        base_score = email_risk * 0.80 + indicator_score * 0.20

    # Give a small boost when multiple independent signals agree
    suspicious_signals = 0
    if email_risk >= 60:
        suspicious_signals += 1
    if highest_url_risk is not None and highest_url_risk >= 60:
        suspicious_signals += 1
    if any(ind.get("severity") == "high" for ind in indicators):
        suspicious_signals += 1

    if suspicious_signals >= 3:
        base_score += 15
    elif suspicious_signals == 2:
        base_score += 7

    # Don't let strong ML evidence get diluted away entirely
    if email_risk >= 70 and indicator_score >= 20:
        base_score = max(base_score, 60)

    final_score = min(max(base_score, 0.0), 100.0)

    if final_score >= 75:
        risk_level = "High"
    elif final_score >= 50:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_score": round(final_score, 2),
        "risk_level": risk_level,
        "components": {
            "email_ml_risk": round(email_risk, 2),
            "highest_url_risk": round(highest_url_risk, 2) if highest_url_risk is not None else None,
            "indicator_score": round(indicator_score, 2),
            "suspicious_signals": suspicious_signals,
        },
    }


def scan_email(subject, body):
    """Run the full email scan: ML model + URL analysis + keyword indicators."""
    subject = subject or ""
    body = body or ""
    email_text = f"{subject} {body}".strip()

    prediction = int(email_model.predict([email_text])[0])
    email_risk_score = float(email_model.predict_proba([email_text])[0][1] * 100)

    urls = extract_urls(body)
    url_analysis = [analyze_url(url) for url in urls]

    indicators = analyze_email_indicators(subject, body)

    risk_result = calculate_combined_email_risk(
        email_risk=email_risk_score,
        url_analysis=url_analysis,
        indicators=indicators,
    )

    return {
        "prediction": prediction,
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "urls_found": len(urls),
        "url_analysis": url_analysis,
        "security_indicators": indicators,
        "risk_components": risk_result["components"],
    }