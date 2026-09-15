"""
WHAT: Linear probe (L2-regularised logistic regression) for failure prediction.

WHY:  Paper §Model and Probe: a linear probe that succeeds demonstrates the
      signal is *linearly decodable*, which is a stronger and more
      interpretable claim than one obtained from a high-capacity classifier.
      Complexity is an extension, not a starting point.

HOW:  Features are standardised with train-split statistics only, then a
      logistic regression is fitted. `predict_risk` returns P(y=1 | prefix).
"""

from __future__ import annotations

import numpy as np


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, C: float = 0.1,
              max_iter: int = 3000, seed: int = 42):
    """Fit a standardised logistic-regression probe.

    Returns:
        `(scaler, classifier)` tuple; scaler fitted on train only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    return scaler, clf


def predict_risk(scaler, clf, X) -> np.ndarray:
    """Return P(y = 1 | X) under a fitted probe."""
    return clf.predict_proba(scaler.transform(X))[:, 1]
