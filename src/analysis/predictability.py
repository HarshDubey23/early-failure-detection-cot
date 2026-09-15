"""
WHAT: Predictability curve (E3, the core result) - AUROC vs observed prefix.

WHY:  Paper §Results "Predictability from Partial Trajectories": for prefix
      fractions f, train a classifier on the first ceil(f*n) steps to predict
      eventual answer incorrectness y. One curve per representation answers
      "how early is failure predictable?", and putting trivial baselines on
      the same axes answers "is the signal more than trace length?".

HOW:  Every representation goes through the SAME grouped train/test split
      (computed once, by question id) and the same bootstrap CIs, so curves are
      directly comparable:
        - surface      : [n_steps, mean_step_chars, total_chars] + logistic regression
        - length-only  : rank by prefix step count (no fitting beyond direction)
        - lexical      : TF-IDF fitted on TRAIN prefixes only + logistic regression
        - surface-MLP  : same surface features, 2-layer MLP (nonlinear extension)
        - surface-GRU  : per-step surface features read in order by a GRU
      Each curve function returns the curve table and the test-set scores at
      the full prefix (f = 1.0), which feed the ROC overlay (F6).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def prefix_steps(steps: list, fraction: float) -> list:
    """First `max(1, ceil(f * n))` steps. Matches Algorithm 1, line 5.

    Example:
        >>> prefix_steps(["a", "b", "c", "d"], 0.25)
        ['a']
        >>> prefix_steps(["a", "b", "c"], 0.5)
        ['a', 'b']
    """
    if not steps:
        return []
    cutoff = max(1, math.ceil(fraction * len(steps)))
    return steps[:cutoff]


def make_split(df: pd.DataFrame, seed: int, test_size: float, context: str) -> tuple:
    """One grouped (by question_id) split shared by every representation.

    Returns:
        (train_idx, test_idx) positional index arrays; disjointness of the
        question sets is asserted inside `grouped_split`.
    """
    from src.utils.seeds import grouped_split

    return grouped_split(df["question_id"].to_numpy(), test_size=test_size,
                         seed=seed, context=context)


def _curve_row(fraction: float, representation: str, target: str, y_test: np.ndarray,
               scores: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    """Package one AUROC-with-CI point of a predictability curve."""
    from src.analysis.statistics import auroc_with_ci

    stats = auroc_with_ci(y_test, scores, n_bootstrap=n_bootstrap, seed=seed)
    return {"target": target, "representation": representation,
            "fraction_observed": fraction, **stats, "n_test": int(len(y_test)),
            "test_positive_rate": float(np.mean(y_test))}


def run_surface_curve(df: pd.DataFrame, target: str, prefix_grid: list, split: tuple,
                      seed: int, n_bootstrap: int, C: float = 1.0,
                      max_iter: int = 1000) -> tuple:
    """Surface-feature logistic-regression curve (the floor any richer detector must clear).

    Returns:
        (curve DataFrame, full-prefix test scores as a 1-D array)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.features.surface import batch_surface

    train_idx, test_idx = split
    y = df[target].astype(int).to_numpy()
    rows, full_scores = [], None
    for fraction in prefix_grid:
        X = batch_surface([prefix_steps(s, fraction) for s in df["steps"]])
        # Standardise with TRAIN statistics only (Algorithm 1, line 9).
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=max_iter, random_state=seed, C=C)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        scores = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        rows.append(_curve_row(fraction, "surface", target, y[test_idx], scores,
                               n_bootstrap, seed))
        if fraction == max(prefix_grid):
            full_scores = scores
    return pd.DataFrame(rows), full_scores


def run_length_only_curve(df: pd.DataFrame, target: str, prefix_grid: list, split: tuple,
                          seed: int, n_bootstrap: int) -> tuple:
    """Length-only baseline: rank test traces by prefix step count.

    Only the direction (longer = riskier or shorter = riskier) is learned from
    the train split. Because k = ceil(f*n), the prefix length reveals the
    total length n at every f; this baseline measures exactly how much of the
    curve that confound explains.
    """
    from src.models.baselines import length_only_scores

    train_idx, test_idx = split
    y = df[target].astype(int).to_numpy()
    rows, full_scores = [], None
    for fraction in prefix_grid:
        lengths = np.array([len(prefix_steps(s, fraction)) for s in df["steps"]], dtype=float)
        scores = length_only_scores(lengths[test_idx], lengths[train_idx], y[train_idx])
        rows.append(_curve_row(fraction, "length-only", target, y[test_idx], scores,
                               n_bootstrap, seed))
        if fraction == max(prefix_grid):
            full_scores = scores
    return pd.DataFrame(rows), full_scores


def run_tfidf_curve(df: pd.DataFrame, target: str, prefix_grid: list, split: tuple,
                    seed: int, n_bootstrap: int, max_features: int = 20000,
                    ngram_range=(1, 2), min_df: int = 2, C: float = 1.0,
                    max_iter: int = 2000) -> tuple:
    """Lexical TF-IDF curve. The vectoriser is fitted on TRAIN prefixes only.

    Fitting the vocabulary / IDF weights on test text would leak test
    statistics into the features, so `fit_tfidf` sees train rows exclusively.
    """
    from sklearn.linear_model import LogisticRegression

    from src.features.lexical import fit_tfidf, prefix_text, transform_tfidf

    train_idx, test_idx = split
    y = df[target].astype(int).to_numpy()
    rows, full_scores = [], None
    for fraction in prefix_grid:
        texts = [prefix_text(s, fraction) for s in df["steps"]]
        vectoriser, train_matrix = fit_tfidf(
            [texts[i] for i in train_idx], max_features=max_features,
            ngram_range=tuple(ngram_range), min_df=min_df,
        )
        test_matrix = transform_tfidf(vectoriser, [texts[i] for i in test_idx])
        clf = LogisticRegression(max_iter=max_iter, random_state=seed, C=C)
        clf.fit(train_matrix, y[train_idx])
        scores = clf.predict_proba(test_matrix)[:, 1]
        rows.append(_curve_row(fraction, "lexical", target, y[test_idx], scores,
                               n_bootstrap, seed))
        if fraction == max(prefix_grid):
            full_scores = scores
    return pd.DataFrame(rows), full_scores


def run_mlp_curve(df: pd.DataFrame, target: str, prefix_grid: list, split: tuple,
                  seed: int, n_bootstrap: int, hidden: tuple = (64, 32),
                  max_iter: int = 500) -> tuple:
    """Surface features through the 2-layer MLP probe (nonlinear extension)."""
    from src.features.surface import batch_surface
    from src.models.mlp_probe import fit_mlp_probe, predict_risk_mlp

    train_idx, test_idx = split
    y = df[target].astype(int).to_numpy()
    rows, full_scores = [], None
    for fraction in prefix_grid:
        X = batch_surface([prefix_steps(s, fraction) for s in df["steps"]])
        scaler, clf = fit_mlp_probe(X[train_idx], y[train_idx], hidden=tuple(hidden),
                                    max_iter=max_iter, seed=seed)
        scores = predict_risk_mlp(scaler, clf, X[test_idx])
        rows.append(_curve_row(fraction, "surface-MLP", target, y[test_idx], scores,
                               n_bootstrap, seed))
        if fraction == max(prefix_grid):
            full_scores = scores
    return pd.DataFrame(rows), full_scores


def step_feature_sequence(steps: list) -> np.ndarray:
    """Per-step features [step_chars, cumulative_chars, step_position] -> (n, 3).

    Example:
        >>> step_feature_sequence(["ab", "cde"]).tolist()
        [[2.0, 2.0, 1.0], [3.0, 5.0, 2.0]]
    """
    rows, cumulative = [], 0.0
    for position, step in enumerate(steps, start=1):
        cumulative += len(step)
        rows.append([float(len(step)), cumulative, float(position)])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def run_gru_curve(df: pd.DataFrame, target: str, prefix_grid: list, split: tuple,
                  seed: int, n_bootstrap: int, hidden_size: int = 32,
                  epochs: int = 200, lr: float = 0.01) -> tuple:
    """GRU sequence probe over per-step surface features (dynamics extension)."""
    from src.models.sequence_probe import fit_sequence_probe

    train_idx, test_idx = split
    y = df[target].astype(int).to_numpy()
    rows, full_scores = [], None
    for fraction in prefix_grid:
        sequences = [step_feature_sequence(prefix_steps(s, fraction)) for s in df["steps"]]
        # Standardise per feature with TRAIN rows only, then log-compress lengths.
        train_stack = np.concatenate([np.log1p(sequences[i]) for i in train_idx])
        mean, std = train_stack.mean(axis=0), train_stack.std(axis=0) + 1e-8
        scaled = [(np.log1p(seq) - mean) / std for seq in sequences]
        model = fit_sequence_probe([scaled[i] for i in train_idx], y[train_idx], seed=seed,
                                   epochs=epochs, hidden_size=hidden_size, lr=lr)
        scores = model.predict_proba_list([scaled[i] for i in test_idx])
        rows.append(_curve_row(fraction, "surface-GRU", target, y[test_idx], scores,
                               n_bootstrap, seed))
        if fraction == max(prefix_grid):
            full_scores = scores
    return pd.DataFrame(rows), full_scores
