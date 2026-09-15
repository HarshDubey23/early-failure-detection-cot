"""
WHAT: Error-onset characterisation (E1) - where failures originate.

WHY:  Paper §Results "Where Failures Originate": absolute first-error index e,
      relative onset rho = e/n, and steps remaining (n - e), per tier and
      pooled, each with bootstrap 95% CIs. Steps remaining bounds the
      mitigation opportunity: the number of steps during which intervention
      is still possible.

HOW:  Pure pandas/numpy over the processed dataframe. `describe_onset` returns
      the per-subset + ALL table (paper Table I). `uniform_step_expected_rho`
      gives the mean rho that would be expected if the error step were drawn
      uniformly from {1..n} for each trace's own n - the right reference point,
      because rho = e/n can never be below 1/n.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def uniform_step_expected_rho(n_steps: np.ndarray) -> float:
    """Mean of rho if e ~ Uniform{1..n} independently per trace: mean((n+1)/(2n)).

    Example:
        >>> uniform_step_expected_rho(np.array([2, 4]))
        0.6875
    """
    n_steps = np.asarray(n_steps, dtype=float)
    return float(np.mean((n_steps + 1.0) / (2.0 * n_steps)))


def describe_onset(df: pd.DataFrame, target: str, n_bootstrap: int = 1000,
                   seed: int = 42) -> pd.DataFrame:
    """Per-tier and pooled onset statistics with bootstrap 95% CIs.

    Args:
        df: Processed traces with n_steps, has_error, first_error_idx,
            relative_onset, steps_remaining and the detector target column.
        target: Outcome column reported as pct_answer_incorrect.
        n_bootstrap: Bootstrap resamples for CIs.
        seed: Config seed.

    Returns:
        DataFrame indexed by subset (+ "ALL").
    """
    from .statistics import bootstrap_mean_ci

    rows = []
    for subset, group in list(df.groupby("subset")) + [("ALL", df)]:
        erroneous = group[group["has_error"]]
        rho = erroneous["relative_onset"].dropna().to_numpy()
        mean_rho, rho_lo, rho_hi = bootstrap_mean_ci(rho, n_bootstrap=n_bootstrap, seed=seed)
        mean_left, left_lo, left_hi = bootstrap_mean_ci(
            erroneous["steps_remaining"].dropna().to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "subset": subset,
                "n_traces": int(len(group)),
                "n_erroneous": int(len(erroneous)),
                "pct_erroneous": 100.0 * float(group["has_error"].mean()),
                "pct_answer_incorrect": 100.0 * float(group[target].astype(float).mean()),
                "median_steps": float(group["n_steps"].median()),
                "max_steps": int(group["n_steps"].max()),
                "mean_err_idx": float(erroneous["first_error_idx"].mean()),
                "median_err_idx": float(erroneous["first_error_idx"].median()),
                "mean_relative_onset": mean_rho,
                "relative_onset_ci_lo": rho_lo,
                "relative_onset_ci_hi": rho_hi,
                "median_relative_onset": float(np.median(rho)) if len(rho) else float("nan"),
                "expected_mean_rho_uniform_step": (
                    uniform_step_expected_rho(erroneous["n_steps"].to_numpy())
                    if len(erroneous) else float("nan")
                ),
                "pct_errors_first_half": 100.0 * float((rho <= 0.5).mean()) if len(rho) else float("nan"),
                # First error AT the final step (e = n), observed vs expected if e ~ Uniform{1..n}.
                "pct_errors_final_step": (
                    100.0 * float((erroneous["first_error_idx"] == erroneous["n_steps"]).mean())
                    if len(erroneous) else float("nan")
                ),
                "expected_pct_final_step_uniform": (
                    100.0 * float(np.mean(1.0 / erroneous["n_steps"].to_numpy(dtype=float)))
                    if len(erroneous) else float("nan")
                ),
                "mean_steps_remaining": mean_left,
                "steps_remaining_ci_lo": left_lo,
                "steps_remaining_ci_hi": left_hi,
            }
        )
    return pd.DataFrame(rows).set_index("subset")


def paper_table_latex(table: pd.DataFrame, tier_labels: dict) -> str:
    """LaTeX body rows for paper Table I (Subset, Traces, % err., Med. n, Mean rho).

    Only the rows are emitted, so the fragment drops into the paper's own
    booktabs tabular without changing its column spec.
    """
    lines = []
    for key, label in tier_labels.items():
        if key not in table.index:
            continue
        row = table.loc[key]
        name = f"\\textbf{{{label}}}" if key == "ALL" else label
        if key == "ALL":
            lines.append("\\midrule")
        lines.append(f"{name} & {int(row['n_traces'])} & {row['pct_erroneous']:.1f} & "
                     f"{row['median_steps']:.0f} & {row['mean_relative_onset']:.3f} \\\\")
    return "\n".join(lines) + "\n"
