"""
Hybrid URL model loader.

Reconstructs the soft-voting ensemble (Random Forest + XGBoost + Gradient
Boosting) from three separately-saved files instead of one joblib pickle.

Why: XGBoost's Booster serializes its internal state via a raw C buffer,
which has known cross-platform/cross-version pickle fragility (Colab/Linux
-> local/Windows in particular). RF and GB are plain sklearn and pickle
fine. XGBoost's own save_model()/load_model() format is built to be
portable, so we use that instead of pickle for XGBoost only.

Expects these three files in the given directory:
  - url_rf_model.pkl
  - url_gb_model.pkl
  - url_xgb_model.json
"""

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb


class HybridURLModel:
    """Drop-in replacement for VotingClassifier - same predict()/predict_proba()
    interface, so no other code needs to know the difference."""

    def __init__(self, rf_model, xgb_model, gb_model):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.gb_model = gb_model
        self.classes_ = rf_model.classes_

    @classmethod
    def load(cls, models_dir):
        models_dir = Path(models_dir)
        rf_model = joblib.load(models_dir / "url_rf_model.pkl")
        gb_model = joblib.load(models_dir / "url_gb_model.pkl")

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(models_dir / "url_xgb_model.json"))

        return cls(rf_model, xgb_model, gb_model)

    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        gb_proba = self.gb_model.predict_proba(X)

        # Soft voting with equal weights, matching VotingClassifier's default
        return (rf_proba + xgb_proba + gb_proba) / 3.0

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]


if __name__ == "__main__":
    # Quick sanity check - point this at your models/ folder
    import sys

    models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    model = HybridURLModel.load(models_dir)

    print("Loaded successfully.")
    print("Classes:", model.classes_)