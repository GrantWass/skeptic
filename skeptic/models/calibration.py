"""Platt scaling wrapper — must live in a shared module so joblib can find it on load."""
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class PlattScaledModel:
    """
    Wraps a fitted sklearn pipeline with Platt scaling (sigmoid calibration).
    Fits a logistic regression on the pipeline's raw probabilities vs actual
    labels on a held-out calibration set. Compatible with any sklearn version.
    """
    def __init__(self, pipe: Pipeline, platt: LogisticRegression) -> None:
        self._pipe  = pipe
        self._platt = platt

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self._pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        p   = self._platt.predict_proba(raw)[:, 1]
        return np.column_stack([1 - p, p])
