"""
WHAT: Error-onset characterisation (E1) - where failures originate.

WHY:  Paper §Results "Where Failures Originate": absolute first-error index,
      relative onset rho = e/n, and steps-remaining (n - e), per tier and
      pooled, each with bootstrap 95% CIs. The steps-remaining quantity bounds
      the mitigation opportunity: steps during which intervention is possible.

HOW:  Pure pandas/numpy over the processed dataframe. `describe_onset` returns
      the per-subset + ALL table (paper Table II); CIs come from
      `src/analysis/statistics.py::bootstrap_mean_ci`.
"""

from __future__ import annotations

import pandas as pd


def describe_onset(df: pd.DataFrame, n_bootstrap: int = 1000,
                   seed: int = 42) -> pd.DataFrame:
    """Per-tier and pooled onset statistics with bootstrap 95% CIs.

    Args:
        df: Processed traces with n_steps, has_error, relative_onset.
        n_bootstrap: Bootstrap resamples for CIs.
        seed: Config seed.

    Returns:
        DataFrame indexed by subset (+ "ALL") with n_traces, pct_erroneous,
        median_steps, mean_err_idx, mean/median relative_onset (+ CI bounds),
        mean_steps_remaining.
    """
    from .statistics import bootstrap_mean_ci

    rows = []
    for subset, group in list(df.groupby("subset")) + [("ALL", df)]:
        erroneous = group[group["has_error"]]
        mean_rho, lo, hi = bootstrap_mean_ci(
            erroneous["relative_onset"].dropna().to_numpy(),
            n_bootstrap=n_bootstrap, seed=seed,
        )
        rows.append(
            {
                "subset": subset,
                "n_traces": len(group),
                "pct_erroneous": 100.0 * group["has_error"].mean(),
                "median_steps": group["n_steps"].median(),
                "mean_err_idx": erroneous["first_error_idx"].mean(),
                "mean_relative_onset": mean_rho,
                "relative_onset_ci_lo": lo,
                "relative_onset_ci_hi": hi,
                "median_relative_onset": erroneous["relative_onset"].median(),
                "mean_steps_remaining": erroneous["steps_remaining"].mean(),
                "frac_first_half": float(
                    (erroneous["relative_onset"] <= 0.5).mean()
                ) if len(erroneous) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("subset")
