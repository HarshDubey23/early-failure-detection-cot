"""
WHAT: Detection lead time (E5) - Delta = e - k* over a threshold sweep - plus
      the step-by-step risk series r_1, r_2, ... that lead time and the
      intervention replay (E8) both consume.

WHY:  Paper §Problem Formulation: lead time is positive when the detector
      fires before the ground-truth error and negative after. It determines
      whether any mitigation is possible at all. Reported as a distribution,
      not a mean alone, plus the fraction of traces with Delta > 0 and the
      false-alarm rate on error-free traces (an alarm is only useful if it is
      rare on traces that are fine).

HOW:  `stepwise_risk_series` is the offline analogue of Algorithm 1's online
      monitor loop. For every absolute step count k it fits one detector on
      TRAIN traces that have at least k steps (their first k steps -> y), then
      scores the first k steps of every TEST trace. A test trace of length n
      therefore gets a risk series r_1..r_n in which r_k depends only on
      s_1..s_k - nothing from the future is visible. `lead_times` replays those
      series against thresholds.
"""

from __future__ import annotations

import numpy as np
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


def _featurise(prefixes: list, representation: str, vectoriser=None,
               max_features: int = 20000, ngram_range=(1, 2), min_df: int = 2):
    """Turn step-prefix lists into a design matrix for one step count k.

    Returns:
        (matrix, fitted_vectoriser_or_None). When `vectoriser` is given it is
        reused (test side); otherwise one is fitted (train side).
    """
    if representation == "surface":
        from src.features.surface import batch_surface

        return batch_surface(prefixes), None
    if representation == "lexical":
        from src.features.lexical import fit_tfidf, transform_tfidf

        texts = [" ".join(p) for p in prefixes]
        if vectoriser is None:
            vectoriser, matrix = fit_tfidf(texts, max_features=max_features,
                                           ngram_range=tuple(ngram_range), min_df=min_df)
            return matrix, vectoriser
        return transform_tfidf(vectoriser, texts), vectoriser
    raise ValueError(f"Unknown representation '{representation}' (surface|lexical).")


def stepwise_risk_series(df: pd.DataFrame, target: str, split: tuple, representation: str,
                         seed: int, min_train_traces: int = 50,
                         max_features: int = 20000, ngram_range=(1, 2),
                         min_df: int = 2) -> dict:
    """Risk series r_1..r_n for every TEST trace, one detector per step count k.

    Args:
        df: Processed traces (steps, question_id, target column).
        target: Label column the detector predicts (e.g. "answer_incorrect").
        split: (train_idx, test_idx) grouped by question id.
        representation: "surface" or "lexical".
        seed: Config seed.
        min_train_traces: A detector for step k is fitted only if at least this
            many train traces have >= k steps and both classes occur; beyond
            that point the last fitted detector is reused (recorded in the
            returned metadata, not hidden).

    Returns:
        {"risks": {trace_id: [r_1, ..., r_n]}, "fitted_up_to_k": int}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_idx, test_idx = split
    steps = list(df["steps"])
    y = df[target].astype(int).to_numpy()
    trace_ids = df["trace_id"].to_numpy()
    max_n = max((len(steps[i]) for i in test_idx), default=0)

    risks = {trace_ids[i]: [] for i in test_idx}
    last_model, fitted_up_to_k = None, 0
    for k in range(1, max_n + 1):
        eligible_train = [i for i in train_idx if len(steps[i]) >= k]
        labels_k = y[eligible_train]
        if len(eligible_train) >= min_train_traces and len(np.unique(labels_k)) == 2:
            matrix, vectoriser = _featurise([steps[i][:k] for i in eligible_train],
                                            representation, None, max_features,
                                            ngram_range, min_df)
            scaler = StandardScaler().fit(matrix) if representation == "surface" else None
            design = scaler.transform(matrix) if scaler is not None else matrix
            clf = LogisticRegression(max_iter=2000, random_state=seed).fit(design, labels_k)
            last_model = (vectoriser, scaler, clf)
            fitted_up_to_k = k
        if last_model is None:
            raise RuntimeError("stepwise_risk_series: could not fit a detector at k=1 "
                               "(too few train traces or a single class).")
        vectoriser, scaler, clf = last_model
        eligible_test = [i for i in test_idx if len(steps[i]) >= k]
        if not eligible_test:
            continue
        matrix, _ = _featurise([steps[i][:k] for i in eligible_test], representation,
                               vectoriser, max_features, ngram_range, min_df)
        design = scaler.transform(matrix) if scaler is not None else matrix
        step_risks = clf.predict_proba(design)[:, 1]
        for i, risk in zip(eligible_test, step_risks):
            risks[trace_ids[i]].append(float(risk))
    return {"risks": risks, "fitted_up_to_k": fitted_up_to_k}


def lead_times(df: pd.DataFrame, risk_series: dict, thresholds: list) -> pd.DataFrame:
    """Compute Delta = e - k* per annotated trace, per threshold.

    Only traces that (a) have a risk series (i.e. are in the test split) and
    (b) carry a first-error annotation contribute a Delta.

    Returns:
        DataFrame with trace_id, threshold, k_star, first_error_idx, delta,
        warned_before_error (delta > 0), alarmed.
    """
    rows = []
    annotated = df[df["has_error"] & df["first_error_idx"].notna()
                   & df["trace_id"].isin(list(risk_series.keys()))]
    for _, row in annotated.iterrows():
        risks = risk_series[row["trace_id"]]
        error_step = int(row["first_error_idx"])
        for threshold in thresholds:
            k_star = first_alarm(risks, threshold)
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "threshold": threshold,
                    "k_star": k_star,
                    "first_error_idx": error_step,
                    "delta": float(error_step - k_star) if k_star is not None else float("nan"),
                    "warned_before_error": bool(k_star is not None and error_step - k_star > 0),
                    "alarmed": k_star is not None,
                }
            )
    columns = ["trace_id", "threshold", "k_star", "first_error_idx", "delta",
               "warned_before_error", "alarmed"]
    return pd.DataFrame(rows, columns=columns)


def false_alarm_rates(df: pd.DataFrame, risk_series: dict, thresholds: list) -> pd.DataFrame:
    """Alarm rate on error-free test traces (no process error at all), per threshold."""
    clean = df[~df["has_error"] & df["trace_id"].isin(list(risk_series.keys()))]
    rows = []
    for threshold in thresholds:
        alarmed = [first_alarm(risk_series[t], threshold) is not None for t in clean["trace_id"]]
        rows.append({"threshold": threshold,
                     "false_alarm_rate_error_free": float(np.mean(alarmed)) if alarmed else float("nan"),
                     "n_error_free": int(len(alarmed))})
    return pd.DataFrame(rows)


def lead_time_summary(lead_df: pd.DataFrame) -> pd.DataFrame:
    """Per-threshold summary.

    Columns: alarm_rate (fraction of erroneous traces that ever alarm),
    frac_warned_before_error (P(Delta > 0) over ALL erroneous traces, never
    alarming counts as not warned), frac_alarm_at_error (Delta = 0),
    median_delta and quartiles over traces that alarmed.
    """
    columns = ["threshold", "n_traces", "alarm_rate", "frac_warned_before_error",
               "frac_alarm_at_error", "median_delta", "delta_q25", "delta_q75"]
    if lead_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for threshold, group in lead_df.groupby("threshold"):
        deltas = group["delta"].dropna()
        rows.append({
            "threshold": threshold,
            "n_traces": int(len(group)),
            "alarm_rate": float(group["alarmed"].mean()),
            "frac_warned_before_error": float(group["warned_before_error"].mean()),
            "frac_alarm_at_error": float((group["delta"] == 0).mean()),
            "median_delta": float(deltas.median()) if len(deltas) else float("nan"),
            "delta_q25": float(deltas.quantile(0.25)) if len(deltas) else float("nan"),
            "delta_q75": float(deltas.quantile(0.75)) if len(deltas) else float("nan"),
        })
    return pd.DataFrame(rows, columns=columns)
