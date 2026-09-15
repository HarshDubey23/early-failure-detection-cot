"""
WHAT: Calibration analysis (E6) - reliability diagrams, ECE, threshold choice.

WHY:  Paper §Discussion "Why Selectivity Is the Design Constraint": the
      threshold tau is calibrated against a corruption budget. A detector that
      is accurate but uncalibrated cannot safely drive that policy - its
      "0.8 risk" must mean an 80% failure rate, not just rank correctly.

HOW:  Reuses `expected_calibration_error` from statistics.py. `threshold_table`
      reports precision/recall/intervention-rate per threshold so tau is chosen
      against an explicit corruption constraint, not max recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y_true: np.ndarray, probs: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Per-bin accuracy vs confidence for the reliability diagram (F10)."""
    from src.analysis.statistics import expected_calibration_error

    result = expected_calibration_error(y_true, probs, n_bins=n_bins)
    return pd.DataFrame(result["bins"])


def threshold_table(y_true: np.ndarray, probs: np.ndarray,
                    thresholds: list) -> pd.DataFrame:
    """Precision/recall/intervention-rate at each candidate threshold.

    The researcher picks tau where corruption (1 - precision on the
    would-succeed population, approximated here) fits the budget.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    rows = []
    for threshold in thresholds:
        alarmed = probs >= threshold
        rows.append(
            {
                "threshold": threshold,
                "intervention_rate": float(alarmed.mean()),
                "precision": float(y_true[alarmed].mean()) if alarmed.sum() else float("nan"),
                "recall": float(alarmed[y_true == 1].mean()) if (y_true == 1).sum() else float("nan"),
                "n_alarmed": int(alarmed.sum()),
            }
        )
    return pd.DataFrame(rows)
