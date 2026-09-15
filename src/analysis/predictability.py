"""
WHAT: Predictability curve (E3, the core result) - AUROC vs observed prefix.

WHY:  Paper §Results "Predictability from Partial Trajectories": for prefix
      fractions, train a classifier on the prefix to predict eventual
      incorrectness. One overlaid figure per representation answers "how early
      is failure predictable?".

HOW:  `run_prefix_curve` takes a `represent(prefix_steps) -> vector` callable
      so surface, lexical, uncertainty, and hidden-state representations share
      one harness: same grouped splits, same seeds, same bootstrap CIs.
      Lexical TF-IDF has its own path (`run_tfidf_prefix_curve`) because the
      vectoriser must be fitted on train prefixes per fraction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def prefix_steps(steps: list, fraction: float) -> list:
    """First `max(1, ceil(f * n))` steps. Matches Algorithm 1, line 5."""
    if not steps:
        return []
    cutoff = max(1, math.ceil(fraction * len(steps)))
    return steps[:cutoff]


def run_prefix_curve(df: pd.DataFrame, represent, prefix_grid: list,
                     seed: int = 42, test_size: float = 0.25,
                     n_bootstrap: int = 1000, representation: str = "surface",
                     C: float = 1.0) -> pd.DataFrame:
    """Prefix-conditioned AUROC curve for a vector representation.

    Args:
        df: Processed traces with `steps`, `question_id`, `has_error`.
        represent: Callable `list[str] -> 1-D array` applied to each prefix.
        prefix_grid: Fractions of the chain observed.
        representation: Name recorded in the output table.

    Returns:
        DataFrame with fraction_observed, auroc, ci_lo, ci_hi, representation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.analysis.statistics import auroc_with_ci
    from src.utils.seeds import grouped_split

    rows = []
    for fraction in prefix_grid:
        prefixes = [prefix_steps(s, fraction) for s in df["steps"]]
        X = np.stack([np.asarray(represent(p), dtype=float) for p in prefixes])
        y = df["has_error"].astype(int).to_numpy()
        groups = df["question_id"].to_numpy()
        train_idx, test_idx = grouped_split(
            groups, test_size=test_size, seed=seed,
            context=f"prefix_curve f={fraction}",
        )
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=C)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        scores = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        stats = auroc_with_ci(y[test_idx], scores, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"fraction_observed": fraction, "representation": representation,
                     **stats, "n_test": len(test_idx)})
    return pd.DataFrame(rows)


def run_tfidf_prefix_curve(df: pd.DataFrame, prefix_grid: list, seed: int = 42,
                           test_size: float = 0.25, n_bootstrap: int = 1000,
                           max_features: int = 20000, ngram_range=(1, 2),
                           min_df: int = 2) -> pd.DataFrame:
    """Prefix-conditioned AUROC curve for the TF-IDF text baseline."""
    from sklearn.linear_model import LogisticRegression

    from src.analysis.statistics import auroc_with_ci
    from src.features.lexical import fit_tfidf, prefix_text, transform_tfidf
    from src.utils.seeds import grouped_split

    rows = []
    for fraction in prefix_grid:
        texts = [prefix_text(s, fraction) for s in df["steps"]]
        y = df["has_error"].astype(int).to_numpy()
        groups = df["question_id"].to_numpy()
        train_idx, test_idx = grouped_split(
            groups, test_size=test_size, seed=seed,
            context=f"tfidf_prefix_curve f={fraction}",
        )
        vectoriser, A = fit_tfidf(
            [texts[i] for i in train_idx], max_features=max_features,
            ngram_range=tuple(ngram_range), min_df=min_df,
        )
        B = transform_tfidf(vectoriser, [texts[i] for i in test_idx])
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(A, y[train_idx])
        scores = clf.predict_proba(B)[:, 1]
        stats = auroc_with_ci(y[test_idx], scores, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"fraction_observed": fraction, "representation": "lexical",
                     **stats, "n_test": len(test_idx)})
    return pd.DataFrame(rows)
