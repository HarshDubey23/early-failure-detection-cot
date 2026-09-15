"""
WHAT: Centralised random-seed handling and grouped (by-question) splitting.

WHY:  Ground rules 3 and 5. Reasoning steps within one trajectory are strongly
      correlated, so row-wise train/test splitting leaks information and
      inflates AUROC. Every split must be grouped by question identifier, and
      reproducibility must follow from the config seed alone.

HOW:  `seed_everything` seeds stdlib, numpy, and (if present) torch from one
      integer. `grouped_split` wraps sklearn's GroupShuffleSplit and then
      asserts train/test question sets are disjoint.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every RNG used in this repository from a single integer.

    Args:
        seed: The seed recorded in the config file and manifest.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is optional; GPU stages skip gracefully without it


def assert_no_leakage(
    train_groups, test_groups, context: str = "split"
) -> None:
    """Assert train and test question-ID sets are disjoint.

    Args:
        train_groups: Question IDs assigned to train.
        test_groups: Question IDs assigned to test.
        context: Label used in the error message.

    Raises:
        AssertionError: If any question ID appears on both sides.
    """
    overlap = set(train_groups) & set(test_groups)
    assert len(overlap) == 0, (
        f"LEAKAGE in {context}: {len(overlap)} question IDs appear in both "
        f"train and test (e.g. {sorted(overlap)[:5]}). Splits must be "
        f"grouped by question, never by row."
    )


def grouped_split(
    groups,
    test_size: float = 0.25,
    seed: int = 42,
    context: str = "split",
):
    """Split row indices into train/test with disjoint question groups.

    Args:
        groups: Per-row question identifier (one entry per row).
        test_size: Fraction assigned to test.
        seed: Random seed from the config file.
        context: Label for error messages and logs.

    Returns:
        `(train_idx, test_idx)` numpy index arrays with disjoint groups.

    Example:
        >>> groups = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]
        >>> tr, te = grouped_split(groups, test_size=0.25, seed=0)
        >>> assert_no_leakage([groups[i] for i in tr], [groups[i] for i in te])
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy_X = np.zeros(len(groups))
    train_idx, test_idx = next(splitter.split(dummy_X, groups=groups))
    assert_no_leakage(
        groups[train_idx], groups[test_idx], context=context
    )
    return train_idx, test_idx
