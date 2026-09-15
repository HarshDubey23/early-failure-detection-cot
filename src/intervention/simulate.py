"""
WHAT: Offline simulation of Algorithm 2 against logged trajectories (E8).

WHY:  Paper §Conclusion "Phase II": report recovery against corruption at
      matched token budgets, comparing with self-consistency and an oracle.
      No live model is available in this build, so what happens AFTER an
      intervention cannot be observed - it has to be assumed. This module
      makes every such assumption an explicit, config-declared parameter so
      the output is read as "what the policy would achieve IF verification
      and regeneration behaved like this", never as measured model behaviour.

HOW:  Measured from data (not assumed):
        - which test traces the detector alarms on, and at which step
          (real risk series from `stepwise_risk_series`),
        - which of those traces would fail (ProcessBench `final_answer_correct`),
        - how many tokens a regeneration from the backtrack step would cost
          (characters of the discarded suffix / `chars_per_token`).
      Assumed (config `intervention.assumed`):
        - P(repair | intervene on a failing trace) for verify / regenerate,
        - P(break  | intervene on a succeeding trace) for verify / regenerate,
        - P(the revert safeguard catches a break), since in reality the risk
          re-estimate after a revision is itself noisy,
        - token cost of one verification call.
      Comparators: no intervention, always intervene (tau = 0), an oracle that
      knows the true error location, and self-consistency N in {2,4,8} under
      an independent-samples model (stated as such).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from .policies import decide


@dataclass
class InterventionAssumptions:
    """Parameters that CANNOT be measured without a live model (all assumed)."""

    p_verify_repair: float
    p_verify_corrupt: float
    p_regen_repair: float
    p_regen_corrupt: float
    p_revert_catches_corruption: float
    verify_tokens: int
    chars_per_token: float


def suffix_tokens(steps: list, from_step: int, chars_per_token: float) -> float:
    """Approximate tokens regenerated when backtracking to 1-indexed `from_step`.

    Example:
        >>> suffix_tokens(["aaaa", "bbbbbbbb"], 2, 4.0)
        2.0
    """
    discarded = steps[max(0, from_step - 1):]
    return sum(len(step) for step in discarded) / chars_per_token


def _empty_counts() -> dict:
    return {"repaired": 0, "broken": 0, "intervened": 0, "tokens": 0.0}


def _summarise(name: str, counts: dict, n_fail: int, n_succeed: int) -> dict:
    """Turn raw counts into the E8 metrics row."""
    n_total = n_fail + n_succeed
    recovery = counts["repaired"] / max(1, n_fail)
    corruption = counts["broken"] / max(1, n_succeed)
    return {
        "policy": name,
        "recovery_rate": recovery,
        "corruption_rate": corruption,
        "net_benefit": recovery - corruption,
        "net_accuracy_change": (counts["repaired"] - counts["broken"]) / max(1, n_total),
        "intervention_rate": counts["intervened"] / max(1, n_total),
        "extra_tokens_per_question": counts["tokens"] / max(1, n_total),
        "n_traces": n_total,
    }


def simulate_policy(df: pd.DataFrame, risk_series: dict, threshold: float,
                    escalation_threshold: float, token_budget: int,
                    assumptions: InterventionAssumptions, target: str,
                    seed: int, name: str | None = None) -> dict:
    """Replay Algorithm 2 once per test trace at the first alarm.

    Args:
        df: Test traces with trace_id, steps and the outcome column `target`.
        risk_series: trace_id -> [r_1, ..., r_n] (only test traces).
        threshold: tau; below it the policy continues (does nothing).
        escalation_threshold: tau_hi; at or above it the policy regenerates.
        token_budget: B; an exhausted budget forces CONTINUE.
        assumptions: Assumed repair / corruption / revert parameters.
        target: Outcome column; 1 means the trace would fail.
        seed: Seed for the Bernoulli draws (config seed).

    Returns:
        Metrics row (see `_summarise`).
    """
    rng = np.random.default_rng(seed)
    counts = _empty_counts()
    evaluated = df[df["trace_id"].isin(list(risk_series.keys()))]
    n_fail = int(evaluated[target].astype(int).sum())
    n_succeed = int(len(evaluated) - n_fail)
    for _, row in evaluated.iterrows():
        risks = risk_series[row["trace_id"]]
        would_fail = bool(row[target])
        alarm_index = next((i for i, r in enumerate(risks) if r >= threshold), None)
        if alarm_index is None:
            continue
        decision = decide(risks[alarm_index], risks[: alarm_index + 1], threshold,
                          escalation_threshold, token_budget)
        if decision.action == "continue":
            continue
        counts["intervened"] += 1
        if decision.action == "verify":
            counts["tokens"] += assumptions.verify_tokens
            p_repair, p_corrupt = assumptions.p_verify_repair, assumptions.p_verify_corrupt
        else:
            counts["tokens"] += suffix_tokens(row["steps"], decision.backtrack_to,
                                              assumptions.chars_per_token)
            p_repair, p_corrupt = assumptions.p_regen_repair, assumptions.p_regen_corrupt
        if would_fail:
            # A failing trace is recovered only if the revision actually fixes it.
            if rng.random() < p_repair:
                counts["repaired"] += 1
        else:
            # A succeeding trace is corrupted only if the revision breaks it AND
            # the revert safeguard (Algorithm 2, line 12) fails to catch that.
            broke = rng.random() < p_corrupt
            caught = rng.random() < assumptions.p_revert_catches_corruption
            if broke and not caught:
                counts["broken"] += 1
    return _summarise(name or f"selective tau={threshold:g}", counts, n_fail, n_succeed)


def simulate_oracle(df: pd.DataFrame, risk_series: dict, assumptions: InterventionAssumptions,
                    target: str) -> dict:
    """Upper bound: intervene exactly on failing traces, regenerate from the true e, always repair."""
    counts = _empty_counts()
    evaluated = df[df["trace_id"].isin(list(risk_series.keys()))]
    n_fail = int(evaluated[target].astype(int).sum())
    n_succeed = int(len(evaluated) - n_fail)
    for _, row in evaluated.iterrows():
        if not bool(row[target]):
            continue
        error_step = int(row["first_error_idx"]) if pd.notna(row["first_error_idx"]) else 1
        counts["intervened"] += 1
        counts["repaired"] += 1
        counts["tokens"] += suffix_tokens(row["steps"], error_step, assumptions.chars_per_token)
    return _summarise("oracle (knows e)", counts, n_fail, n_succeed)


def majority_vote_accuracy(per_sample_accuracy: float, n_samples: int) -> float:
    """P(majority of N independent samples is correct); ties broken at random.

    Binary-outcome model: each sample is correct with probability p,
    independently. Real samples from one model are positively correlated, so
    this is an optimistic model of self-consistency.

    Example:
        >>> round(majority_vote_accuracy(0.6, 3), 3)
        0.648
    """
    p = per_sample_accuracy
    total = 0.0
    for k in range(n_samples + 1):
        probability = comb(n_samples, k) * p**k * (1 - p) ** (n_samples - k)
        if 2 * k > n_samples:
            total += probability
        elif 2 * k == n_samples:
            total += 0.5 * probability
    return total


def self_consistency_row(df: pd.DataFrame, risk_series: dict, n_samples: int,
                         assumptions: InterventionAssumptions, target: str) -> dict:
    """Self-consistency with N fresh samples under the independent-samples model.

    Per-sample accuracy p is the MEASURED accuracy of the logged test traces.
    Under independence the voted answer is correct with probability
    p_maj = majority_vote_accuracy(p, N) regardless of the original sample, so
    recovery = p_maj and corruption = 1 - p_maj. Extra tokens = (N - 1) x the
    measured mean trace length in tokens.
    """
    evaluated = df[df["trace_id"].isin(list(risk_series.keys()))]
    per_sample_accuracy = 1.0 - float(evaluated[target].astype(int).mean())
    p_majority = majority_vote_accuracy(per_sample_accuracy, n_samples)
    mean_tokens = float(np.mean([sum(len(s) for s in steps) for steps in evaluated["steps"]])
                        / assumptions.chars_per_token)
    return {
        "policy": f"self-consistency N={n_samples} (iid model)",
        "recovery_rate": p_majority,
        "corruption_rate": 1.0 - p_majority,
        "net_benefit": p_majority - (1.0 - p_majority),
        "net_accuracy_change": p_majority - per_sample_accuracy,
        "intervention_rate": 1.0,
        "extra_tokens_per_question": (n_samples - 1) * mean_tokens,
        "n_traces": int(len(evaluated)),
    }


def compare_policies(df: pd.DataFrame, risk_series: dict, thresholds: list,
                     escalation_threshold: float, token_budget: int,
                     assumptions: InterventionAssumptions, self_consistency_n: list,
                     target: str, seed: int) -> pd.DataFrame:
    """Every E8 row: selective thresholds, no-op, always-intervene, oracle, SC-N."""
    n_total = int(df["trace_id"].isin(list(risk_series.keys())).sum())
    rows = [{"policy": "no intervention", "recovery_rate": 0.0, "corruption_rate": 0.0,
             "net_benefit": 0.0, "net_accuracy_change": 0.0, "intervention_rate": 0.0,
             "extra_tokens_per_question": 0.0, "n_traces": n_total}]
    for threshold in thresholds:
        rows.append(simulate_policy(df, risk_series, threshold, escalation_threshold,
                                    token_budget, assumptions, target, seed))
    rows.append(simulate_policy(df, risk_series, 0.0, escalation_threshold, token_budget,
                                assumptions, target, seed, name="always intervene (tau=0)"))
    rows.append(simulate_oracle(df, risk_series, assumptions, target))
    for n_samples in self_consistency_n:
        rows.append(self_consistency_row(df, risk_series, n_samples, assumptions, target))
    return pd.DataFrame(rows)
