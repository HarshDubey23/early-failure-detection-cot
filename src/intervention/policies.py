"""
WHAT: Intervention policies - continue, verify, backtrack, regenerate (E8).

WHY:  Paper Algorithm 2 (Cost-Aware Selective Intervention). Its defining
      property is that the default action is *to do nothing*: given Huang et
      al.'s finding that unconditional revision degrades accuracy, the policy
      escalates from cheap verification to costly regeneration and reverts any
      revision that does not reduce estimated risk.

HOW:  `decide` is the pure policy function (testable without any model).
      `simulate.py` replays it against logged trajectories offline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyDecision:
    action: str  # "continue" | "verify" | "regenerate"
    backtrack_to: int | None = None
    reason: str = ""


def decide(risk: float, risks_so_far: list, threshold: float,
           escalation_threshold: float, token_budget: int) -> PolicyDecision:
    """Apply Algorithm 2 to one alarm state.

    Args:
        risk: Current-step risk r_k.
        risks_so_far: Risks r_1..r_k (for backtracking to peak risk).
        threshold: tau - below it, always continue.
        escalation_threshold: tau_hi - above it, regenerate instead of verify.
        token_budget: Remaining budget B; exhausted budget forces continue.

    Example:
        >>> decide(0.9, [0.2, 0.9], 0.5, 0.8, 256).action
        'regenerate'
        >>> decide(0.1, [0.1], 0.5, 0.8, 256).action
        'continue'
    """
    if risk < threshold:
        return PolicyDecision("continue", reason="risk below tau (default: do nothing)")
    if token_budget <= 0:
        return PolicyDecision("continue", reason="budget exhausted")
    # Backtrack to peak risk: b = argmax_{i <= k} r_i (1-indexed).
    backtrack_to = int(max(range(len(risks_so_far)),
                           key=lambda i: risks_so_far[i])) + 1
    if risk >= escalation_threshold:
        return PolicyDecision("regenerate", backtrack_to,
                              reason="risk above tau_hi: costly regeneration")
    return PolicyDecision("verify", backtrack_to,
                          reason="cheaper verification first")


def should_revert(risk_before: float, risk_after: float) -> bool:
    """Line 12 safeguard: revert revisions that do not reduce estimated risk."""
    return risk_after >= risk_before
