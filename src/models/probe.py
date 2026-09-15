"""
WHAT: Linear probe (L2-regularised logistic regression) for failure prediction,
      and the layer x depth-bin probing sweep of E4.

WHY:  Paper §Model and Probe: a linear probe that succeeds demonstrates the
      signal is *linearly decodable*, which is a stronger and more
      interpretable claim than one obtained from a high-capacity classifier.
      Complexity is an extension, not a starting point. The brief (E4) asks for
      a sweep over layers instead of trusting a single layer.

HOW:  Features are standardised with train-split statistics only, then a
      logistic regression is fitted. `predict_risk` returns P(y=1 | state).
      `layer_depth_auroc` makes ONE grouped split over all step-states (by
      question id, so every sample of a question and every step of a sample
      lands on the same side) and reuses it for every layer and depth bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def layer_depth_auroc(meta: pd.DataFrame, states_by_layer: dict, depth_bins: list,
                      seed: int, test_size: float, n_bootstrap: int, C: float,
                      max_iter: int, min_test_states: int = 25) -> pd.DataFrame:
    """Linear-probe AUROC for every (layer, depth bin) cell.

    Args:
        meta: One row per step-state with question_id, rel_pos (k / n) and
            final_correct (graded outcome of the whole trajectory).
        states_by_layer: layer -> (n_states, hidden_size) array aligned with `meta`.
        depth_bins: [[lo, hi], ...]; a state belongs to a bin when lo < rel_pos <= hi.
        min_test_states: Cells with fewer test states, or one class on either
            side, are reported with status "SKIPPED_TOO_FEW" rather than a number.

    Returns:
        DataFrame with layer, depth_bin, auroc, ci_lo, ci_hi, n_train, n_test, status.

    Example:
        A 2-layer, 2-bin sweep returns 4 rows, one per cell.
    """
    from src.analysis.statistics import auroc_with_ci
    from src.utils.seeds import grouped_split

    if "final_correct" not in meta.columns:
        raise ValueError("layer_depth_auroc: metadata lacks graded 'final_correct' labels.")
    y_all = 1 - meta["final_correct"].astype(int).to_numpy()  # target = FAILURE
    rel_pos = meta["rel_pos"].to_numpy()
    train_idx, test_idx = grouped_split(meta["question_id"].to_numpy(), test_size, seed,
                                        context="E4 probe split (all layers, all bins)")
    rows = []
    for layer in sorted(states_by_layer):
        states = states_by_layer[layer]
        for lo, hi in depth_bins:
            label = f"{lo:.0%}-{min(hi, 1.0):.0%}"
            tr = train_idx[(rel_pos[train_idx] > lo) & (rel_pos[train_idx] <= hi)]
            te = test_idx[(rel_pos[test_idx] > lo) & (rel_pos[test_idx] <= hi)]
            row = {"layer": layer, "depth_bin": label, "n_train": int(len(tr)),
                   "n_test": int(len(te))}
            if (len(te) < min_test_states or len(np.unique(y_all[tr])) < 2
                    or len(np.unique(y_all[te])) < 2):
                rows.append({**row, "auroc": float("nan"), "ci_lo": float("nan"),
                             "ci_hi": float("nan"), "status": "SKIPPED_TOO_FEW"})
                continue
            scaler, clf = fit_probe(states[tr], y_all[tr], C, max_iter, seed)
            stats = auroc_with_ci(y_all[te], predict_risk(scaler, clf, states[te]),
                                  n_bootstrap, seed, groups=meta["question_id"].to_numpy()[te])
            rows.append({**row, **stats, "status": "OK"})
    return pd.DataFrame(rows)
