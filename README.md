# Multi-Modal Phishing Detection System (ThreatLens)

A phishing detection system with independent detection modules per
channel. Internally the deployed app/API is named **ThreatLens** (see the
FastAPI title in `backend/main.py`). Built as a Computer Science portfolio
/ research project.

## Status

| Module | Status |
|---|---|
| URL detection | ✅ Built — RF/XGBoost/GB hybrid ensemble, trained on PhiUSIIL |
| Email detection | ✅ Built — TF-IDF + best-of-3 classifier, trained on a labeled email dataset |

## Project structure

```
backend/
    main.py                    # FastAPI app ("ThreatLens API") — /scan/url, /scan/email
    url_utils.py                 # URL normalization, hostname/registered-domain extraction, trusted-domain allowlist
    url_features.py              # 27-feature extraction for the hybrid URL model
    hybrid_url_model.py           # Loads and combines the RF/XGBoost/GB URL models
    email_scanner.py             # Email ML scoring + keyword indicators + embedded-URL analysis
    threat_intelligence.py       # URLhaus lookup for a submitted URL

data/
    PhiUSIIL_Phishing_URL_Dataset.csv   # URL training data actually used by the notebook
    LegitPhish.csv                       # Additional URL dataset (not currently wired into training)
    new_data_urls.csv                    # Raw candidate URL data for future expansion
    clean_url_dataset.csv               # Output of audit_url_dataset.py (currently unused by training)
    phishing_email_dataset.csv          # Raw email dataset
    processed/
        email_clean.csv                  # Cleaned email training data (used by train_email_model.py)
    ReadMe.md.txt                        # (dataset-specific notes)

ML/
    audit_url_dataset.py            # Cleans new_data_urls.csv -> clean_url_dataset.csv (see note below)
    test_url_generalization.py      # Curated sanity/generalization test suite for the hybrid URL model
    train_email_model.py            # Trains and selects the email phishing classifier

models/
    url_rf_model.pkl                 # Random Forest URL model
    url_gb_model.pkl                 # Gradient Boosting URL model
    url_xgb_model.json               # XGBoost URL model (native format, not pickle)
    phishing_email_model.pkl         # Email phishing classifier (TF-IDF + classifier pipeline)
    phishing_url_text_model.pkl      # Optional second URL model — not built yet (see below)

frontend/
    index.html                       # UI

URL_Phishing_Detection_Full_Pipeline.ipynb   # URL model training notebook (Colab, v4)
```

## Modules

### 1. URL Detection

Classifies a URL as phishing or legitimate from its structure.

**Training (`URL_Phishing_Detection_Full_Pipeline.ipynb`, v4 — "HTTP Bias
Fix"):**
- Loads `PhiUSIIL_Phishing_URL_Dataset.csv` from Google Drive. This is
  currently the only dataset the notebook trains on — `LegitPhish.csv`,
  `new_data_urls.csv`, and `clean_url_dataset.csv` are not yet wired in.
- **Augments the legitimate class** with additional URLs built from a set
  of realistic path templates (`/login`, `/checkout`, `/wiki/...`, etc.)
  applied to curated legitimate anchor domains (GitHub, Wikipedia, Stack
  Overflow, Amazon, etc.), specifically to correct a bias where the
  original data skewed the model toward keying off `www`/`http` vs
  `https` rather than genuine structure — hence "v4: HTTP Bias Fix".
- Extracts the same 27 structural features as `url_features.py`
  (URL/domain length, subdomain count, obfuscation/entropy measures,
  digit/letter/special-char ratios, HTTPS/IP/`@`/hyphen/dot flags,
  suspicious-keyword flag, shortener flag, path length, etc.) — see
  `FEATURE_ORDER` in `url_features.py` for the exact list.
- **Split:** a random `train_test_split` (80/20, stratified by label,
  `random_state=42`) — rows, not registered domains. There is currently no
  domain-grouped split for this dataset, so some rows from the same
  domain could appear in both train and test.
- **Models:** trains Random Forest, XGBoost, and Gradient Boosting
  individually, then combines them into a `VotingClassifier(voting="soft")`
  hybrid model. Reports accuracy/precision/recall/F1, a confusion matrix,
  and feature importances for the hybrid model.
- **Sanity test:** runs the trained hybrid model against a small set of
  real URLs before export, as a final gut-check.
- **Export:** saves the three underlying models separately —
  `url_rf_model.pkl`, `url_gb_model.pkl` via `joblib`, and
  `url_xgb_model.json` via XGBoost's own `save_model()` — specifically to
  avoid XGBoost's cross-platform/version pickle fragility. Downloads all
  three from Colab.

**Serving:**
- `HybridURLModel` (`backend/hybrid_url_model.py`) reloads the three files
  and exposes `predict()` / `predict_proba()` as a soft-voting average
  (equal weights, matching `VotingClassifier`'s default), so the rest of
  the backend doesn't need to know it isn't a single model.
- `url_features.py` is the single source of truth for feature extraction
  on the serving side, and is explicitly documented to need to stay in
  lockstep with the notebook's feature function — a past mismatch between
  the two was a real bug.
- `main.py`'s `/scan/url` endpoint also supports an **optional second
  model**, a TF-IDF + text classifier (`phishing_url_text_model.pkl`),
  which is planned but **not yet built** — if the file isn't present, the
  endpoint logs a warning and falls back to the structural hybrid model
  alone.
- **Trusted-domain override:** `url_utils.py` maintains a small allowlist
  (`google.com`, `wikipedia.org`, `github.com`, `youtube.com`,
  `microsoft.com`, `amazon.com`, `stackoverflow.com`). In `/scan/url`,
  URLhaus is still checked first (so a known-malicious URL on an allowlisted
  domain can still be flagged), but otherwise a trusted registered domain
  has its risk score capped low — this exists specifically to stop the ML
  model from occasionally misclassifying well-known sites.
- **Final risk scoring** in `/scan/url` blends: the structural model's
  phishing probability, the optional text model's phishing probability
  (weighted 40/60 toward text when both are present, with a floor so one
  model can't fully cancel a strongly confident other model), the
  trusted-domain cap, and a URLhaus override (any hit forces risk ≥ 95).
  The response includes a human-readable `reasons` list explaining the
  verdict (HTTPS/`@`/keyword checks, model agreement/disagreement, URLhaus
  status, trusted-domain status).

**Auditing & testing:**
- `audit_url_dataset.py` cleans a *separate* candidate dataset
  (`data/new_data_urls.csv` → `data/clean_url_dataset.csv`): normalizes
  URLs, removes rows with conflicting labels for the same normalized URL,
  drops normalized duplicates, extracts registered domains, and reports
  domain-level label consistency. **Not currently consumed by the training
  notebook** — it's prep for folding additional URL data into training
  later; safe to remove for now if you have no near-term plan to use
  `new_data_urls.csv`.
- `test_url_generalization.py` runs the loaded `HybridURLModel` against a
  curated, hand-labeled set of real-world-style URLs (known legitimate
  domains with realistic paths/queries, plus various phishing patterns:
  brand-lookalike subdomains, IP-based URLs, percent-encoding
  obfuscation, `@`-symbol tricks, shorteners) and reports a pass/warn/fail
  verdict: ≥95% with zero failures = pass, ≥90% = warning, below that =
  fail, don't deploy.

### 2. Email Detection

Classifies an email as phishing/spam or legitimate, and separately scores
any URLs found inside it.

**Training (`ML/train_email_model.py`):**
- Loads `data/processed/email_clean.csv` (`text`, `label` columns).
- Removes duplicate email texts before splitting, to avoid inflating test
  accuracy with near-identical train/test rows.
- Cleans text: strips HTML tags, collapses URLs to `URLTOKEN` and email
  addresses to `EMAILTOKEN` (so "does this email contain a link" becomes
  a learnable signal instead of treating every unique URL as noise).
- 80/20 stratified split.
- Compares three candidates via 5-fold cross-validated F1: Logistic
  Regression, a calibrated Linear SVM (`CalibratedClassifierCV` — needed
  since `LinearSVC` has no native `predict_proba`, and the blended risk
  score in `email_scanner.py` needs probabilities), and Multinomial Naive
  Bayes — all with `class_weight="balanced"` where applicable, with an
  explicit warning if the minority class is under 25%.
- Trains the best-CV-F1 candidate as a `Pipeline(TfidfVectorizer(1,2-gram)
  + classifier)`, evaluates accuracy/ROC-AUC/F1/confusion matrix on the
  held-out test set, and if the false-positive rate exceeds 10%, prints
  concrete next steps (spot-check false positives, raise the decision
  threshold, or reconsider `class_weight`) along with sample false
  positives.
- Saves the fitted pipeline to `models/phishing_email_model.pkl`.

**Serving (`backend/email_scanner.py`):**
- Runs the email model on the combined subject+body text for a phishing
  probability.
- Extracts URLs from the body (via `urlextract`, with cleanup for
  trailing punctuation and leading junk before `http(s)://`) and scores
  each one through the same `HybridURLModel` used for standalone URL
  scans.
- Flags keyword-based indicators in four categories — urgency language,
  credential requests, financial/payment language, and account
  verification language — plus a "contains 3+ URLs" flag.
- **Blends everything into one score** (`calculate_combined_email_risk`):
  when a URL is found, 65% email-ML score + 20% highest URL risk + 15%
  indicator score; otherwise 80% email-ML + 20% indicator score. Adds a
  small boost when 2–3 independent signals agree (high email risk, high
  URL risk, a high-severity keyword hit), and enforces a floor so strong
  ML evidence plus real indicators can't be diluted below a "Medium" risk
  level. Final score maps to Low/Medium/High.

### Threat Intelligence (URLhaus lookup)

`backend/threat_intelligence.py` checks a submitted URL against
[URLhaus](https://urlhaus.abuse.ch/) (`abuse.ch`'s malicious-URL
database) as a supplementary signal alongside the ML models.

- Sends the URL to the URLhaus API (`POST
  https://urlhaus-api.abuse.ch/v1/url/`) with an auth key.
- Returns whether URLhaus has a record for that URL (`found: True/False`),
  along with the raw URLhaus data when found.
- Requires a `URLHAUS_AUTH_KEY` environment variable (loaded via
  `python-dotenv` / a `.env` file). If it's missing, the lookup is skipped
  gracefully (`available: False`) rather than failing.
- Also degrades gracefully on a non-200 response or a connection error,
  returning a message instead of raising.

## Backend (API)

Built with **FastAPI** (`backend/main.py`, app title "ThreatLens API").

```bash
cd backend
uvicorn main:app --reload
```

**Endpoints:**
- `GET /` — health/status message
- `GET /health` — health check
- `POST /scan/url` — `{"url": "..."}` → full risk assessment (final
  verdict, per-model breakdown, reputation check, threat-intel result,
  and a plain-language `reasons` list)
- `POST /scan/email` — `{"subject": "...", "body": "..."}` → risk
  assessment, extracted/scored URLs, and triggered keyword indicators

**Note:** CORS is currently configured with `allow_origins=["*"]` and
`allow_credentials=False`. That's a reasonable default for local
development against a static frontend, but worth tightening to specific
origins before any public deployment.

**Environment:** create a `.env` file in `backend/` with:

```
URLHAUS_AUTH_KEY=your_key_here
```

## Frontend

Static UI in `frontend/index.html`, calling the backend API.

## Label convention

```
0 = phishing
1 = legitimate
```

## Setup

```bash
pip install -r requirements.txt   # add one if not already present
```

Dependencies observed across the codebase: `fastapi`, `uvicorn`,
`pydantic`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`,
`requests`, `python-dotenv`, `tldextract`, `urlextract`.

## Roadmap

- [x] URL detection: RF/GB/XGB hybrid ensemble trained on PhiUSIIL
- [x] Email detection: TF-IDF + best-of-3 classifier trained
- [x] Threat intelligence: URLhaus lookup integrated
- [x] Trusted-domain override for well-known sites
- [ ] Build the optional URL text model (`phishing_url_text_model.pkl`)
      referenced but not yet trained in `main.py`
- [ ] Decide whether to fold `LegitPhish.csv` / `new_data_urls.csv` into
      URL training, and if so, wire `audit_url_dataset.py`'s cleaned
      output into the training notebook
- [ ] Consider a domain-grouped train/test split for the URL model (the
      current split is row-level random, not domain-grouped)
- [ ] Wire frontend ↔ backend end-to-end
- [ ] Tighten CORS before any public deployment
- [ ] Deployment (containerize FastAPI backend, host frontend)
