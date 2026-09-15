"""
WHAT: Lexical (TF-IDF over reasoning prefixes) representation.

WHY:  Paper §Representation: tests whether the failure signal is available
      from surface text alone. The honest comparison the examiners look for:
      if the text baseline matches the probe, internal states add nothing.

HOW:  Thin wrapper over sklearn's TfidfVectorizer, fitted on train prefixes
      only (fitting on test would leak vocabulary statistics).
"""

from __future__ import annotations


def prefix_text(steps: list, prefix_fraction: float) -> str:
    """Join the first `ceil(f * n)` steps into one string.

    Example:
        >>> prefix_text(["a", "b", "c", "d"], 0.5)
        'a b'
    """
    import math

    n_steps = len(steps)
    if n_steps == 0:
        return ""
    cutoff = max(1, math.ceil(prefix_fraction * n_steps))
    return " ".join(steps[:cutoff])


def fit_tfidf(train_texts: list, max_features: int = 20000,
              ngram_range=(1, 2), min_df: int = 2):
    """Fit a TF-IDF vectoriser on train prefix texts only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(
        max_features=max_features, ngram_range=tuple(ngram_range), min_df=min_df
    )
    matrix = vectoriser.fit_transform(train_texts)
    return vectoriser, matrix


def transform_tfidf(vectoriser, texts: list):
    """Transform (test) prefix texts with an already-fitted vectoriser."""
    return vectoriser.transform(texts)
