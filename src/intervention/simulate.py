"""
WHAT: Offline simulation of Algorithm 2 against logged trajectories (E8).

WHY:  Paper §Conclusion "Phase II": report recovery against corruption at
      matched token budgets, comparing with self-consistency and PRM-guided
      baselines. Offline replay on logged outcomes is the honest, cheap
      version: no live model calls inside the evaluation loop.

HOW:  Each trajectory has a known outcome (would_fail) and a risk series. The
      simulator applies `decide` at the first alarm and models stochastic
      repair: verification fixes a failing trace with probability p_verify and
      corrupts a succeeding trace with probability q_verify (and analogously
      for regeneration). Probabilities are config-declared *assumptions*,
      varied in a sensitivity sweep - never presented as measured model
      behaviour. Comparators: no-op, always-intervene, self-consistency
      N in {2,4,8} (majority-vote model over resampled outcomes), oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policies import decide, should_revert


def simulate(df: pd.DataFrame, risk_series: dict, threshold: float,
             escalation_threshold: float, token_budget: int,
             p_verify: float = 0.4, q_verify: float = 0.05,
             p_regen: float = 0.6, q_regen: float = 0.10,
             verify_cost: int = 32, regen_cost: int = 128,
             seed: int = 42) -> dict:
    """Replay the selective policy over logged trajectories.

    Args:
        df: Traces with trace_id and has_error (would_fail = has_error).
        risk_series: trace_id -> per-step risk list.
        threshold / escalation_threshold / token_budget: Algorithm 2 params.
        p_verify/q_verify/p_regen/q_regen: assumed repair/corruption probs.
        verify_cost/regen_cost: assumed token costs.
        seed: Config seed for the stochastic repair draws.

    Returns:
        Dict with recovery_rate, corruption_rate, net_benefit,
        intervention_rate, extra_tokens_per_question.
    """
    rng = np.random.default_rng(seed)
    n_recovered = n_failed = 0
    n_corrupted = n_succeeded = 0
    n_intervened = 0
    extra_tokens = 0
    for _, row in df.iterrows():
        risks = list(risk_series.get(row["trace_id"], []))
        would_fail = bool(row["has_error"])
        if would_fail:
            n_failed += 1
        else:
            n_succeeded += 1
        # First alarm at threshold.
        alarm_at = next((i for i, r in enumerate(risks) if r >= threshold), None)
        if alarm_at is None:
            continue
        decision = decide(risks[alarm_at], risks[: alarm_at + 1],
                          threshold, escalation_threshold, token_budget)
        if decision.action == "continue":
            continue
        n_intervened += 1
        if decision.action == "verify":
            extra_tokens += verify_cost
            repaired = rng.random() < (p_verify if would_fail else 1.0 - q_verify)
            # Revert safeguard: revision kept only if risk would drop.
            risk_after = risks[alarm_at] * (0.5 if repaired else 1.2)
            if should_revert(risks[alarm_at], risk_after):
                repaired = False
        else:
            extra_tokens += regen_cost
            repaired = rng.random() < (p_regen if would_fail else 1.0 - q_regen)
            risk_after = risks[alarm_at] * (0.3 if repaired else 1.1)
            if should_revert(risks[alarm_at], risk_after):
                repaired = False
        if would_fail and repaired:
            n_recovered += 1
        if not would_fail and not repaired:
            n_corrupted += 1
    recovery = n_recovered / max(1, n_failed)
    corruption = n_corrupted / max(1, n_succeeded)
    return {
        "threshold": threshold,
        "recovery_rate": recovery,
        "corruption_rate": corruption,
        "net_benefit": recovery - corruption,
        "intervention_rate": n_intervened / max(1, len(df)),
        "extra_tokens_per_question": extra_tokens / max(1, len(df)),
    }


def compare_policies(df: pd.DataFrame, risk_series: dict, thresholds: list,
                     **kwargs) -> pd.DataFrame:
    """Run `simulate` over the threshold sweep plus named comparators.

    Comparators: no_intervention (all zeros), always_intervene (threshold 0),
    oracle (perfect repair, zero corruption - upper bound), self_consistency
    rows are added by the calling script from outcome resampling.
    """
    rows = [simulate(df, risk_series, t, **kwargs) for t in thresholds]
    rows.append({"threshold": "no_intervention", "recovery_rate": 0.0,
                 "corruption_rate": 0.0, "net_benefit": 0.0,
                 "intervention_rate": 0.0, "extra_tokens_per_question": 0.0})
    oracle = simulate(df, risk_series, 0.0, **{**kwargs, "p_verify": 1.0,
                     "q_verify": 0.0, "p_regen": 1.0, "q_regen": 0.0})
    oracle["threshold"] = "oracle"
    rows.append(oracle)
    return pd.DataFrame(rows)
