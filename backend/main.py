from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.email_scanner import scan_email
from backend.threat_intelligence import check_urlhaus
from backend.url_features import extract_url_features
from backend.hybrid_url_model import HybridURLModel
from backend.url_utils import (
    normalize_url,
    extract_hostname,
    get_registered_domain,
    is_trusted_domain,
)

print("I AM RUNNING THE NEW MAIN.PY")

app = FastAPI(
    title="ThreatLens API",
    description="Backend API for the ThreatLens security scanner",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Structural hybrid model (Random Forest + XGBoost + Gradient Boosting).
# Loaded from 3 separate files (url_rf_model.pkl, url_gb_model.pkl,
# url_xgb_model.json) instead of one combined pickle, since XGBoost's
# raw Booster buffer has cross-platform/version pickle fragility.
url_feature_model = HybridURLModel.load(PROJECT_ROOT / "models")

# TF-IDF + Logistic Regression text model - OPTIONAL.
# Not built yet. When present, it adds a second, independent signal
# based on learned word/character patterns rather than hand-picked
# structural features. The app runs fine without it; it just relies
# on the structural model + URLhaus + trusted-domain logic alone.
URL_TEXT_MODEL_PATH = PROJECT_ROOT / "models" / "phishing_url_text_model.pkl"

if URL_TEXT_MODEL_PATH.exists():
    url_text_model = joblib.load(URL_TEXT_MODEL_PATH)
else:
    url_text_model = None
    print("WARNING: phishing_url_text_model.pkl not found - "
          "URL scanning will run on the structural model only.")


class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    subject: str
    body: str


@app.get("/")
def home():
    return {"message": "ThreatLens API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def prepare_feature_dataframe(features):
    """
    Convert extracted URL features into a pandas DataFrame using the exact
    feature names the trained structural model expects. Avoids the warning:
    "X does not have valid feature names, but RandomForestClassifier was
    fitted with feature names."
    """
    # Newer sklearn models trained on a DataFrame expose the original
    # feature names through feature_names_in_.
    feature_names = getattr(url_feature_model, "feature_names_in_", None)

    if feature_names is not None:
        return pd.DataFrame([features], columns=list(feature_names))

    # Fallback for models that don't expose feature_names_in_ (e.g.
    # HybridURLModel) - works fine as long as extract_url_features()
    # returns a dict with keys already in the training order, which it does.
    return pd.DataFrame([features])


@app.post("/scan/url")
def scan_url(request: URLRequest):
    # --- Normalize + extract domain info ---
    original_url = request.url.strip()
    url = normalize_url(original_url)
    if not url:
        return {"error": "A valid URL is required."}

    hostname = extract_hostname(url)
    registered_domain = get_registered_domain(url)
    trusted_domain = is_trusted_domain(url)

    # --- Model 1: structural hybrid model ---
    features = extract_url_features(url)
    feature_dataframe = prepare_feature_dataframe(features)

    feature_prediction = url_feature_model.predict(feature_dataframe)[0]
    feature_probabilities = url_feature_model.predict_proba(feature_dataframe)[0]

    # Class 0 = Phishing, Class 1 = Legitimate
    feature_phishing_probability = float(feature_probabilities[0])
    feature_risk = feature_phishing_probability * 100

    # --- Model 2: TF-IDF URL text model (optional) ---
    if url_text_model is not None:
        text_prediction = url_text_model.predict([url])[0]
        text_probabilities = url_text_model.predict_proba([url])[0]
        text_phishing_probability = float(text_probabilities[0])
        text_risk = text_phishing_probability * 100
    else:
        text_prediction = None
        text_risk = 0.0

    # --- Combine the ML models ---
    if url_text_model is not None:
        # Text model gets slightly more weight - raw URL text can catch
        # suspicious words/patterns that structural features miss.
        ml_risk_score = feature_risk * 0.4 + text_risk * 0.6

        # Also respect a strong signal from either model on its own, so one
        # model can't fully cancel out the other when it's highly confident.
        strongest_model_risk = max(feature_risk, text_risk)
        ml_risk_score = max(ml_risk_score, strongest_model_risk * 0.8)
    else:
        # Text model unavailable - rely on the structural model alone.
        ml_risk_score = feature_risk

    ml_risk_score = round(ml_risk_score, 2)

    # --- URLhaus threat intelligence ---
    urlhaus_result = check_urlhaus(url)

    # --- Final risk score ---
    risk_score = ml_risk_score

    if urlhaus_result["found"]:
        # External threat intelligence is treated as strong evidence.
        risk_score = max(risk_score, 95.0)
    elif trusted_domain:
        # Trusted domains still get scanned by both ML models and URLhaus,
        # but this reputation layer stops the structural model from
        # misclassifying well-known domains (google.com, github.com, etc.)
        # as phishing. URLhaus is checked first above, so a known-malicious
        # URL can still override this.
        risk_score = min(risk_score, 10.0)

    risk_score = round(risk_score, 2)

    # --- Final classification ---
    if urlhaus_result["found"]:
        classification = "Phishing"
    elif trusted_domain:
        classification = "Legitimate"
    elif risk_score >= 50:
        classification = "Phishing"
    else:
        classification = "Legitimate"

    # 0 = Phishing, 1 = Legitimate
    prediction = 0 if classification == "Phishing" else 1

    # --- Risk level ---
    if risk_score >= 80:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # --- Explanations ---
    reasons = []

    if trusted_domain:
        reasons.append(
            f"The registered domain '{registered_domain}' is currently "
            "included in the ThreatLens trusted-domain list."
        )

    if feature_prediction == 0:
        reasons.append("The URL structure contains patterns commonly associated with phishing.")

    if text_prediction is not None and text_prediction == 0:
        reasons.append("The URL text contains patterns associated with phishing.")

    # Model disagreement
    if text_prediction is not None and feature_prediction == 1 and text_prediction == 0:
        reasons.append(
            "The URL structure appears legitimate, but the URL text model "
            "detected potentially suspicious patterns."
        )
    elif text_prediction is not None and feature_prediction == 0 and text_prediction == 1:
        reasons.append(
            "The URL structure appears suspicious, but the URL text model "
            "did not detect strong phishing patterns."
        )

    # URL-specific indicators
    lowered_url = url.lower()
    if "login" in lowered_url:
        reasons.append("The URL contains a login-related path.")
    if "verify" in lowered_url:
        reasons.append("The URL contains verification-related wording.")
    if "account" in lowered_url:
        reasons.append("The URL contains account-related wording.")
    if "@" in url:
        reasons.append("The URL contains an @ symbol, which can be used for URL obfuscation.")
    if lowered_url.startswith("http://"):
        reasons.append("The URL does not use HTTPS.")

    if (
        feature_prediction == 1
        and (text_prediction is None or text_prediction == 1)
        and not urlhaus_result["found"]
    ):
        reasons.append("Machine learning analysis classified the URL as legitimate.")

    if urlhaus_result["found"]:
        reasons.append("URLhaus has a record of this URL as a known malicious URL.")
    elif urlhaus_result["available"]:
        reasons.append("The URL was not found in the URLhaus threat intelligence database.")
    else:
        reasons.append("URLhaus threat intelligence was unavailable for this scan.")

    # --- Assessment summaries for the response ---
    if ml_risk_score >= 80:
        ml_assessment = "Highly Suspicious"
    elif ml_risk_score >= 50:
        ml_assessment = "Suspicious"
    elif ml_risk_score >= 30:
        ml_assessment = "Potentially Suspicious"
    else:
        ml_assessment = "Low Suspicion"

    reputation_assessment = "Trusted" if trusted_domain else "Unknown"

    if urlhaus_result["found"]:
        threat_assessment = "Known Malicious URL"
    elif urlhaus_result["available"]:
        threat_assessment = "No Known URLhaus Record"
    else:
        threat_assessment = "Threat Intelligence Unavailable"

    return {
        "url": url,
        "hostname": hostname,
        "registered_domain": registered_domain,

        "final_assessment": {
            "classification": classification,
            "prediction": prediction,
            "risk_score": risk_score,
            "risk_level": risk_level,
        },

        "model_analysis": {
            "feature_model": {
                "prediction": int(feature_prediction),
                "phishing_probability": round(feature_risk, 2),
            },
            "text_model": (
                {
                    "prediction": int(text_prediction),
                    "phishing_probability": round(text_risk, 2),
                }
                if text_prediction is not None
                else None
            ),
            "combined_ml_risk": ml_risk_score,
            "ml_assessment": ml_assessment,
        },

        "reputation": {
            "registered_domain": registered_domain,
            "trusted_domain": trusted_domain,
            "assessment": reputation_assessment,
        },

        "threat_intelligence": {
            "urlhaus_available": urlhaus_result["available"],
            "urlhaus_found": urlhaus_result["found"],
            "assessment": threat_assessment,
            "message": urlhaus_result["message"],
        },

        "reasons": reasons,
    }


@app.post("/scan/email")
def scan_email_endpoint(request: EmailRequest):
    result = scan_email(request.subject, request.body)

    prediction = result["prediction"]
    risk_score = result["risk_score"]
    risk_level = result["risk_level"]

    # Classification follows the blended risk level, not the raw model
    # prediction, so it always agrees with the score shown to the user.
    classification = "Phishing / Spam" if risk_level in ("Medium", "High") else "Legitimate"

    return {
        "subject": request.subject,
        "prediction": prediction,
        "classification": classification,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "urls_found": result["urls_found"],
        "url_analysis": result["url_analysis"],
        "security_indicators": result["security_indicators"],
        "risk_components": result["risk_components"],
    }
