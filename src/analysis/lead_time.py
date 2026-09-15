"""
WHAT: Detection lead time (E5) - Delta = e - k* over a threshold sweep.

WHY:  Paper §Problem Formulation: lead time is positive when the detector
      fires before the ground-truth error and negative after. It determines
      whether any mitigation is possible at all. Reported as a distribution,
      not a mean alone, plus the fraction of traces with Delta > 0.

HOW:  `lead_times` replays per-step risk series against thresholds. The risk
      series come from the prefix detectors (E3); this module only consumes
      them, so timing evaluation stays decoupled from model fitting.
"""

from __future__ import annotations

import pandas as pd


def first_alarm(risks: list, threshold: float) -> int | None:
    """1-indexed first step with risk >= threshold; None when never alarming.

    Example:
        >>> first_alarm([0.1, 0.6, 0.4], 0.5)
        2
    """
    for i, risk in enumerate(risks, start=1):
        if risk >= threshold:
            return i
    return None


def lead_times(df: pd.DataFrame, risk_series: dict,
               thresholds: list) -> pd.DataFrame:
    """Compute Delta = e - k* per trace (with first-error labels) per threshold.

    Args:
        df: Processed traces with trace_id, first_error_idx, has_error.
        risk_series: Mapping trace_id -> list of per-step risks (1-indexed).
        thresholds: Threshold sweep values.

    Returns:
        DataFrame with trace_id, threshold, k_star, first_error_idx, delta,
        warned_before_error (delta > 0).
    """
    rows = []
    annotated = df[df["has_error"] & df["first_error_idx"].notna()]
    for _, row in annotated.iterrows():
        risks = risk_series.get(row["trace_id"], [])
        for threshold in thresholds:
            k_star = first_alarm(risks, threshold)
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "threshold": threshold,
                    "k_star": k_star,
                    "first_error_idx": row["first_error_idx"],
                    "delta": (
                        row["first_error_idx"] - k_star
                        if k_star is not None else float("nan")
                    ),
                    "warned_before_error": (
                        bool(row["first_error_idx"] - k_star > 0)
                        if k_star is not None else False
                    ),
                    "alarmed": k_star is not None,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=["trace_id", "threshold", "k_star", "first_error_idx",
                     "delta", "warned_before_error", "alarmed"]
        )
    return result


def lead_time_summary(lead_df: pd.DataFrame) -> pd.DataFrame:
    """Per-threshold summary: alarm rate, P(Delta>0), median Delta."""
    if lead_df.empty:
        return pd.DataFrame(
            columns=["threshold", "alarm_rate", "frac_warned_before_error",
                     "median_delta"]
        )
    return (
        lead_df.groupby("threshold")
        .agg(alarm_rate=("alarmed", "mean"),
             frac_warned_before_error=("warned_before_error", "mean"),
             median_delta=("delta", "median"))
        .reset_index()
    )
