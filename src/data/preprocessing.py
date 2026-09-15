"""
WHAT: The six preprocessing rules P1-P6, each a named, tested function.

WHY:  The paper's §Experimental Setup defines exactly six rules with explicit
      rationales. Implementing each as a separate pure function means the
      researcher can explain and defend each one in the viva, and each one is
      unit-tested on a synthetic fixture with a known answer.

HOW:  `apply_all` runs P1-P6 in order on a trace list. P6 (grouped splitting)
      lives in `src/utils/seeds.py` and is imported here for completeness of
      the documented pipeline.
"""

from __future__ import annotations

import math

from .loaders import Trace


def apply_p1_merge_tiers(traces: list) -> list:
    """P1: merge difficulty tiers, retaining tier as the `subset` covariate.

    Reason: enables testing whether the failure pattern is
    difficulty-dependent instead of assuming it is uniform.

    Worked example: traces with subsets GSM8K/MATH stay distinguishable after
    merging because `subset` is preserved on each trace.
    """
    merged = list(traces)  # concatenation; `subset` column keeps the tier
    return merged


def apply_p2_count_steps(trace: Trace) -> Trace:
    """P2: record `n_steps` per trace (the denominator for normalisation)."""
    trace.n_steps = len(trace.steps)
    return trace


def apply_p3_error_free_code(trace: Trace, error_free_code: int = -1) -> Trace:
    """P3: normalise the error-free sentinel to `first_error_idx=None`.

    Args:
        trace: A trace whose `first_error_idx` may hold the raw sentinel.
        error_free_code: Source-dataset code for "no error" (ProcessBench: -1).

    Worked example: `first_error_idx=-1` -> None (target `has_error=False`);
    `first_error_idx=3` is untouched.
    """
    if trace.first_error_idx == error_free_code:
        trace.first_error_idx = None
    return trace


def apply_p4_relative_onset(first_error_idx: int | None, n_steps: int) -> float | None:
    """P4: normalised onset `rho = e / n` so traces of differing length compare.

    Step 4 of 5 and step 4 of 40 are not the same event; without this, every
    positional claim is invalid.

    `first_error_idx` must be the paper's 1-indexed e in {1, ..., n}; loaders
    convert the datasets' 0-indexed labels before this point. The assertion
    makes an indexing slip fail loudly instead of silently shifting rho.

    Example:
        >>> apply_p4_relative_onset(2, 10)
        0.2
        >>> apply_p4_relative_onset(None, 10) is None
        True
    """
    if first_error_idx is None or not n_steps:
        return None
    if not 1 <= first_error_idx <= n_steps:
        raise ValueError(
            f"P4: first_error_idx={first_error_idx} outside 1..n={n_steps}. "
            "Was a 0-indexed label passed without conversion?"
        )
    return first_error_idx / n_steps


def apply_p5_min_steps(trace: Trace, min_steps: int = 2) -> bool:
    """P5: keep only traces with `n_steps >= min_steps`.

    Reason: trajectory shape is undefined for a single step.
    Returns True when the trace is kept.
    """
    return len(trace.steps) >= min_steps and trace.n_steps >= min_steps


def apply_p6_grouped_split_notice() -> str:
    """P6: document the grouped-split rule; enforcement lives in seeds.py.

    Steps inside one trace are highly correlated; row-wise splitting leaks and
    inflates AUROC. This is the most common methodological error in this area.
    """
    return (
        "P6: all splits grouped by question_id with disjointness assertion "
        "(see src/utils/seeds.py::grouped_split)."
    )


def apply_all(
    traces: list,
    min_steps: int = 2,
    error_free_code: int = -1,
) -> list:
    """Apply P1-P5 to a trace list and return the cleaned list.

    (P6 constrains every later split; see `grouped_split`.)

    Args:
        traces: Raw traces from any loader.
        min_steps: P5 threshold.
        error_free_code: P3 sentinel.

    Returns:
        Cleaned traces with `n_steps` counted and relative onset computable.
    """
    merged = apply_p1_merge_tiers(traces)
    cleaned = []
    for trace in merged:
        trace = apply_p3_error_free_code(trace, error_free_code)
        trace = apply_p2_count_steps(trace)
        if not apply_p5_min_steps(trace, min_steps):
            continue
        cleaned.append(trace)
    _ = apply_p6_grouped_split_notice()
    return cleaned


def quality_report(raw_traces: list, cleaned_traces: list, frame) -> dict:
    """Data-quality numbers behind the viva statement "no missing values after cleaning".

    Structural missingness (first_error_idx / relative_onset / steps_remaining
    empty for error-free traces) is separated from unexpected missingness in
    any other column, which must be zero.

    Returns:
        Dict with n_raw, n_clean, dropped_by_p5, n_questions, duplicate_trace_ids,
        frac_has_error, frac_answer_incorrect, unexpected_missing.
    """
    structural = {"first_error_idx", "relative_onset", "steps_remaining", "generator"}
    unexpected = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
        if column not in structural and int(frame[column].isna().sum()) > 0
    }
    # Structural missingness must line up exactly with error-free traces.
    mismatched = int((frame["first_error_idx"].isna() != ~frame["has_error"]).sum())
    if mismatched:
        unexpected["first_error_idx_vs_has_error"] = mismatched
    return {
        "n_raw": len(raw_traces),
        "n_clean": len(cleaned_traces),
        "dropped_by_p5": len(raw_traces) - len(cleaned_traces),
        "n_questions": int(frame["question_id"].nunique()),
        "duplicate_trace_ids": int(frame["trace_id"].duplicated().sum()),
        "frac_has_error": float(frame["has_error"].mean()),
        "frac_answer_incorrect": float(frame["answer_incorrect"].astype(float).mean()),
        "unexpected_missing": unexpected,
    }


def relative_onset_column(first_error_idx, n_steps) -> float:
    """Vector helper: NaN for error-free traces (structural missingness)."""
    if first_error_idx is None or (isinstance(first_error_idx, float) and math.isnan(first_error_idx)):
        return float("nan")
    if not n_steps:
        return float("nan")
    return float(first_error_idx) / float(n_steps)
