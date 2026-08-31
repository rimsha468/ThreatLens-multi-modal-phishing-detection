import joblib
import pandas as pd

from pathlib import Path

from backend.url_features import extract_url_features, FEATURE_ORDER
from backend.hybrid_url_model import HybridURLModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models"

print("=" * 80)
print("THREATLENS — URL MODEL GENERALIZATION TEST (Hybrid Model)")
print("=" * 80)
print("\nLoading model...")

model = HybridURLModel.load(PROJECT_ROOT / "models")
print("Model loaded successfully.")

# --- Test cases ---
test_cases = [
    # Known legitimate domains - simple
    ("Legitimate", "https://www.google.com"),
    ("Legitimate", "https://www.microsoft.com"),
    ("Legitimate", "https://www.github.com"),
    ("Legitimate", "https://www.wikipedia.org"),
    ("Legitimate", "https://www.python.org"),

    # Legitimate domains - realistic paths
    ("Legitimate", "https://www.google.com/search"),
    ("Legitimate", "https://www.google.com/search?q=python"),
    ("Legitimate", "https://github.com/openai"),
    ("Legitimate", "https://github.com/openai/gpt"),
    ("Legitimate", "https://www.microsoft.com/en-us/"),
    ("Legitimate", "https://www.python.org/downloads/"),
    ("Legitimate", "https://www.wikipedia.org/wiki/Artificial_intelligence"),
    ("Legitimate", "https://en.wikipedia.org/wiki/Phishing"),
    ("Legitimate", "https://www.amazon.com/dp/B08N5WRWNW"),

    # Legitimate URLs - query parameters
    ("Legitimate", "https://www.google.com/search?q=machine+learning"),
    ("Legitimate", "https://github.com/search?q=python&type=repositories"),
    ("Legitimate", "https://www.microsoft.com/en-us/search?q=security"),
    ("Legitimate", "https://stackoverflow.com/questions/tagged/python"),

    # Legitimate HTTP
    ("Legitimate", "http://example.com"),
    ("Legitimate", "http://example.org"),
    ("Legitimate", "http://neverssl.com"),

    # Suspicious domains
    ("Phishing", "http://paypal-login-security.example.com/verify"),
    ("Phishing", "http://secure-account-verification.example.com/login"),
    ("Phishing", "http://login-account-verify.example.com/signin"),
    ("Phishing", "http://account-verification.example.com/update"),
    ("Phishing", "http://secure-login.example.com/account/verify"),

    # Subdomain impersonation
    ("Phishing", "http://google.com.security-check.example.com/login"),
    ("Phishing", "http://paypal.com.security.example.com/login"),
    ("Phishing", "http://microsoft.com.verify.example.com/account"),

    # IP-based URLs
    ("Phishing", "http://192.168.1.1/login"),
    ("Phishing", "http://192.168.1.1/login/verify/account"),
    ("Phishing", "http://10.0.0.1/account/login"),

    # Obfuscation
    ("Phishing", "http://example.com/%6c%6f%67%69%6e"),
    ("Phishing", "http://user@example.com/login"),

    # Suspicious URL structure
    ("Phishing", "http://example.com/login?user=test&verify=1"),
    ("Phishing", "http://secure-login-account.example.com/verify"),
    ("Phishing", "http://verify-your-account.example.com/login"),

    # Shorteners
    ("Phishing", "http://bit.ly/free-prize-claim-now"),
]


def prepare_features(url):
    features = extract_url_features(url)
    return pd.DataFrame([features], columns=FEATURE_ORDER)


# --- Run tests ---
results = []
correct = 0
total = len(test_cases)

for expected, url in test_cases:
    dataframe = prepare_features(url)
    prediction = int(model.predict(dataframe)[0])
    probabilities = model.predict_proba(dataframe)[0]

    # Convention matches the training notebook: 0 = Phishing, 1 = Legitimate
    phishing_probability = float(probabilities[0])
    legitimate_probability = float(probabilities[1])
    actual_prediction = "Phishing" if prediction == 0 else "Legitimate"
    passed = actual_prediction == expected

    if passed:
        correct += 1

    results.append({
        "expected": expected,
        "url": url,
        "prediction": actual_prediction,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "passed": passed,
    })

# --- Display results ---
print("\n" + "=" * 80)
print("GENERALIZATION RESULTS")
print("=" * 80)

for result in results:
    status = "PASS" if result["passed"] else "FAIL"

    print("\n" + "-" * 80)
    print(f"Expected:              {result['expected']}")
    print(f"Prediction:            {result['prediction']}")
    print(f"Phishing probability:  {result['phishing_probability'] * 100:.2f}%")
    print(f"Legitimate probability:{result['legitimate_probability'] * 100:.2f}%")
    print(f"Status:                {status}")
    print(f"URL: {result['url']}")

# --- Final score ---
accuracy = correct / total

print("\n" + "=" * 80)
print("FINAL GENERALIZATION SCORE")
print("=" * 80)
print(f"\nCorrect predictions: {correct}/{total}")
print(f"Generalization accuracy: {accuracy * 100:.2f}%")

# --- Failure analysis ---
failed_results = [r for r in results if not r["passed"]]

print("\n" + "=" * 80)
print("FAILURE ANALYSIS")
print("=" * 80)

if not failed_results:
    print("\nExcellent.")
    print("No independent generalization failures detected.")
else:
    print(f"\n{len(failed_results)} generalization failures detected.")

    for result in failed_results:
        print("\nFAILED URL:")
        print(result["url"])
        print("Expected:", result["expected"])
        print("Predicted:", result["prediction"])
        print("Phishing probability:", f"{result['phishing_probability'] * 100:.2f}%")

# --- Decision ---
print("\n" + "=" * 80)
print("MODEL DECISION")
print("=" * 80)

if accuracy >= 0.95 and not failed_results:
    print("\nGENERALIZATION CHECK: PASSED")
    print("The model is ready for API integration.")
elif accuracy >= 0.90:
    print("\nGENERALIZATION CHECK: WARNING")
    print("The model has some failures.")
    print("Review before deploying.")
else:
    print("\nGENERALIZATION CHECK: FAILED")
    print("The model is not generalizing sufficiently.")
    print("Do NOT deploy this model.")

print("\n" + "=" * 80)
print("GENERALIZATION TEST COMPLETE")
print("=" * 80)