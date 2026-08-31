import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "email_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_email_model.pkl"

RANDOM_STATE = 42


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------
# Raw email text often has noise (HTML tags, raw URLs, raw email
# addresses) that hurts a TF-IDF model. Collapsing URLs/emails into
# placeholder tokens turns "does this email contain a link" into a
# learnable signal, instead of treating every unique URL as its own
# rare, mostly-useless vocabulary term.

def clean_email_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)                # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)  # collapse URLs
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)     # collapse email addresses
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading email dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {len(df)}")

print("\nLabel distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(4) * 100, "%")

# Warn early if class imbalance looks severe
label_counts = df["label"].value_counts(normalize=True)
if label_counts.min() < 0.25:
    print(
        "\nWARNING: classes are imbalanced (minority class < 25%). "
        "class_weight='balanced' will be used below to compensate."
    )

# --------------------------------------------------
# Duplicate check
# --------------------------------------------------
# Duplicate/near-duplicate rows across train/test inflate test accuracy
# without meaning the model actually generalizes.

duplicate_count = df.duplicated(subset="text").sum()
if duplicate_count > 0:
    print(f"\nFound {duplicate_count} duplicate email texts - removing them.")
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Dataset size after dedup: {len(df)}")

# --------------------------------------------------
# Clean text
# --------------------------------------------------

print("\nCleaning email text (stripping HTML, collapsing URLs/emails)...")
df["text_clean"] = df["text"].apply(clean_email_text)

X = df["text_clean"]
y = df["label"]

# --------------------------------------------------
# Split dataset
# --------------------------------------------------

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --------------------------------------------------
# Candidate models
# --------------------------------------------------
# class_weight="balanced" costs nothing and directly compensates for
# class imbalance. LinearSVC doesn't expose predict_proba directly, so
# it's wrapped in CalibratedClassifierCV to get probability estimates
# (needed for the risk-score blending logic in email_scanner.py).

tfidf_params = dict(
    lowercase=True,
    strip_accents="unicode",
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
)

candidates = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "LinearSVC (calibrated)": CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    ),
    "MultinomialNB": MultinomialNB(),
}

print("\n" + "=" * 80)
print("MODEL COMPARISON (5-fold cross-validation, scoring=f1)")
print("=" * 80)

cv_results = {}

for name, clf in candidates.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("classifier", clf),
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    cv_results[name] = scores.mean()
    print(f"{name:<28} F1 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

best_model_name = max(cv_results, key=cv_results.get)
print(f"\nBest model by cross-validated F1: {best_model_name}")

# --------------------------------------------------
# Train the best model on the full training set
# --------------------------------------------------

print(f"\nTraining final model ({best_model_name}) on full training set...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("classifier", candidates[best_model_name]),
])

model.fit(X_train, y_train)
print("Training completed!")

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

print("\nTesting model...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)
f1 = f1_score(y_test, predictions)

print(f"\nAccuracy: {round(accuracy, 4)}")
print(f"ROC-AUC:  {round(roc_auc, 4)}")
print(f"F1:       {round(f1, 4)}")

print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["Legitimate", "Phishing"]))

cm = confusion_matrix(y_test, predictions)
print("Confusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives  (legit correctly passed):     {tn}")
print(f"False Positives (legit wrongly flagged):      {fp}  <-- your reported problem")
print(f"False Negatives (phishing wrongly passed):    {fn}")
print(f"True Positives  (phishing correctly caught):  {tp}")

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"\nFalse positive rate: {false_positive_rate * 100:.2f}%")

if false_positive_rate > 0.10:
    print(
        "\nFalse positive rate is high (>10%). Likely causes to check:\n"
        "  1. Some 'Legitimate' rows in your training data may actually contain\n"
        "     urgent/marketing language that looks phishy - spot check a sample\n"
        "     of your false positives (see below) against their labels.\n"
        "  2. Consider raising the decision threshold above the default 0.5\n"
        "     (e.g. only classify as Phishing if probability >= 0.6) to trade\n"
        "     some recall for fewer false alarms - tune this against your own\n"
        "     validation data rather than guessing a number.\n"
        "  3. If class_weight='balanced' overcorrected, try class_weight=None\n"
        "     and compare - imbalance handling can sometimes overshoot."
    )

    # Show a few actual false positives for manual inspection
    fp_mask = (y_test.values == 0) & (predictions == 1)
    if fp_mask.sum() > 0:
        print("\nSample false positives (legitimate emails flagged as phishing):")
        for text in X_test[fp_mask].head(5):
            print("-", text[:150])

# --------------------------------------------------
# Save model
# --------------------------------------------------

print("\nSaving email model...")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Email model saved successfully!")
print(f"Location: {MODEL_PATH}")

print("\nEmail model training complete!")