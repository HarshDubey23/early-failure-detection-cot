"""
WHAT: Baselines every probe must beat: majority-class, length-only,
      entropy-only.

WHY:  Ground rule 4: report baselines even when they beat the proposed
      method. A null result honestly reported is a valid contribution.
      These three floors separate "the probe found something" from
      "any trivial feature predicts this".

HOW:  Each baseline exposes `fit`/`predict_proba` on the same grouped splits
      as the probes, so AUROC numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np


class MajorityBaseline:
    """Predict the train prevalence for every test row (AUROC ~ 0.5)."""

    def fit(self, y_train: np.ndarray):
        self.prevalence_ = float(np.mean(y_train))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.prevalence_)


def length_only_scores(n_steps_test: np.ndarray, n_steps_train: np.ndarray,
                       y_train: np.ndarray) -> np.ndarray:
    """Rank test traces by step count in the direction seen at train time.

    If erroneous train traces are longer on average, longer test traces score
    higher, and vice versa. Returns monotone scores suitable for AUROC.
    """
    direction = np.sign(
        np.mean(n_steps_train[y_train == 1]) - np.mean(n_steps_train[y_train == 0])
    )
    if direction == 0:
        direction = 1.0
    # Tied lengths must receive tied scores. An argsort-of-argsort rank gives
    # tied traces arbitrary distinct ranks whose order depends on the sort
    # implementation, which made this baseline platform-dependent (0.539 on
    # Windows vs 0.535 on Linux for the same split). Average ranks fix that.
    from scipy.stats import rankdata

    ranks = rankdata(np.asarray(n_steps_test, dtype=float), method="average")
    scores = (ranks - 1.0) / max(1, len(ranks) - 1)
    return scores if direction > 0 else 1.0 - scores


def entropy_only_scores(entropy_test: np.ndarray) -> np.ndarray:
    """Rank by mean token entropy (more uncertain = more likely to fail)."""
    order = np.argsort(np.argsort(np.asarray(entropy_test, dtype=float)))
    return order / max(1, len(order) - 1)
