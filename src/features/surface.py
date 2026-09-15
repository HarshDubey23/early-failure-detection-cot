"""
WHAT: Surface features - step count and length statistics per trace/prefix.

WHY:  Paper §Representation: surface features establish the floor any richer
      representation must clear. If a hidden-state probe cannot beat step
      counts and character statistics, internal representations contribute
      nothing beyond what is already visible in the output text.

HOW:  One function per feature group, all pure and typed, operating on step
      string lists so they work identically for full traces and prefixes.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["n_steps", "mean_step_chars", "total_chars"]


def surface_features(steps: list) -> np.ndarray:
    """Compute [n_steps, mean_step_chars, total_chars] for a step list.

    Example:
        >>> surface_features(["ab", "cdef"])
        array([ 2.,  3.,  6.])
    """
    lengths = [len(s) for s in steps]
    n_steps = float(len(steps))
    if not lengths:
        return np.array([0.0, 0.0, 0.0])
    return np.array(
        [n_steps, float(np.mean(lengths)), float(sum(lengths))],
        dtype=float,
    )


def batch_surface(prefixes: list) -> np.ndarray:
    """Stack `surface_features` over a list of step lists -> (n, 3) array."""
    return np.stack([surface_features(steps) for steps in prefixes])


def char_statistics(steps: list) -> dict:
    """Extra character statistics (digits, operators) for EDA, not the probe."""
    text = " ".join(steps)
    return {
        "n_chars": len(text),
        "digit_frac": sum(c.isdigit() for c in text) / max(1, len(text)),
        "operator_count": sum(text.count(op) for op in ("+", "-", "*", "/", "=")),
    }
