"""
WHAT: Uncertainty features - token entropy and logit margin per prefix.

WHY:  Uncertainty is the cheapest non-textual signal: it needs only the
      logits of the single generation being scored (no resampling, unlike
      semantic entropy). It sits between lexical and hidden-state cost.

HOW:  Pure numpy functions over per-token log-probability / logit arrays.
      When a generator does not expose logits, callers record NOT_RUN for
      this representation instead of substituting zeros.
"""

from __future__ import annotations

import numpy as np


def mean_token_entropy(token_logprobs: np.ndarray) -> float:
    """Mean negative log-probability over a prefix (higher = less confident).

    Example:
        >>> mean_token_entropy(np.array([-0.1, -0.5]))
        0.3
    """
    token_logprobs = np.asarray(token_logprobs, dtype=float)
    if token_logprobs.size == 0:
        return float("nan")
    return float(-np.mean(token_logprobs))


def mean_logit_margin(top1_logits: np.ndarray, top2_logits: np.ndarray) -> float:
    """Mean (top-1 minus top-2) logit gap; small margins mean uncertainty.

    Example:
        >>> mean_logit_margin(np.array([3.0, 2.0]), np.array([1.0, 1.5]))
        1.25
    """
    top1 = np.asarray(top1_logits, dtype=float)
    top2 = np.asarray(top2_logits, dtype=float)
    if top1.size == 0:
        return float("nan")
    return float(np.mean(top1 - top2))


def uncertainty_features(token_logprobs, top1_logits=None, top2_logits=None) -> np.ndarray:
    """Stack available uncertainty scalars into one feature row."""
    features = [mean_token_entropy(np.asarray(token_logprobs))]
    if top1_logits is not None and top2_logits is not None:
        features.append(mean_logit_margin(top1_logits, top2_logits))
    return np.asarray(features, dtype=float)
