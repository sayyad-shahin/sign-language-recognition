"""Loads the trained model and predicts a sign from normalized features."""
from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODEL = Path("models/sign_model.joblib")


class SignClassifier:
    def __init__(self, path=DEFAULT_MODEL):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run train.py first.")
        bundle = joblib.load(path)
        self.model, self.classes = bundle["model"], bundle["classes"]

    def predict(self, features):
        """Returns (label, confidence 0-1)."""
        proba = self.model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i])
