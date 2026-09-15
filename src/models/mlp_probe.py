"""
WHAT: 2-layer MLP probe (nonlinear extension of the linear probe).

WHY:  Tests whether failure information is present but *not linearly
      decodable*. Reported as an extension: if the MLP beats the linear probe,
      the signal exists in a nonlinear form; if not, linearity suffices.

HOW:  sklearn MLPClassifier on standardised features, same grouped splits and
      seeds as the linear probe so the comparison is fair.
"""

from __future__ import annotations

import numpy as np


def fit_mlp_probe(X_train: np.ndarray, y_train: np.ndarray,
                  hidden: tuple = (64, 32), max_iter: int = 500,
                  seed: int = 42):
    """Fit a standardised 2-layer MLP probe. Returns `(scaler, clf)`."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                        random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    return scaler, clf


def predict_risk_mlp(scaler, clf, X) -> np.ndarray:
    """Return P(y = 1 | X) under a fitted MLP probe."""
    return clf.predict_proba(scaler.transform(X))[:, 1]
