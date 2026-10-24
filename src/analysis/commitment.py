"""
WHAT: Counterfactual commitment analysis: where does a reasoning trajectory
      actually become doomed, and is that the annotated first error?

WHY:  ProcessBench marks the first *incorrect* step e. The paper's thesis is
      that failure is decided at a *choke point*, which need not be e. Stage 11
      shows observationally that ~1 in 4 first errors is not fatal and that
      text detectors cannot tell fatal from non-fatal errors. This module turns
      the question into a measurement: from every prefix T_{1:k} of a trace we
      sample m continuations from the trace's own generator and estimate the
      success probability

          v_k = P(final answer correct | question, T_{1:k}),   k = 0..n

      v_0 is the from-scratch solve rate (problem difficulty for that model),
      so every quantity below is read relative to it. A choke point is a step
      whose inclusion collapses v; an error that leaves v high is recoverable.

      Related designs: Thought Anchors (Bogdan et al., 2025) and "The Point of
      No Return" (Merrill & Srivastava, 2026) resample from prefixes to find
      decisive sentences (the latter for deception). Neither compares the
      counterfactual decision point with human first-error annotations, which
      is what this module does.

HOW:  Pure functions over v-curves (lists of (n_correct, m) per prefix) so the
      analysis is unit-testable without a GPU. Generation lives in
      scripts/12_counterfactual_rollouts.py.
"""

from __future__ import annotations

import math
import re

import numpy as np

# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (brace-matched), or None.

    Example:
        >>> last_boxed(r"so \\boxed{\\frac{1}{2}} and finally \\boxed{x^{2}+1}")
        'x^{2}+1'
        >>> last_boxed("no box") is None
        True
    """
    if not text:
        return None
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None


_STRIP = [r"\\left", r"\\right", r"\\!", r"\\,", r"\\;", r"\\ ", r"\\\$", r"\$", r"\\text\{[^}]*\}",
          r"\\mbox\{[^}]*\}", r"\^\{?\\circ\}?", r"\\%", r"%", r"\s+"]


def normalise_answer(ans: str | None) -> str | None:
    """Light, conservative normalisation used when math-verify is unavailable.

    Example:
        >>> normalise_answer(r"\\$1{,}000") == normalise_answer("1000")
        True
        >>> normalise_answer(r"\\dfrac{1}{2}") == normalise_answer(r"\\frac12")
        False
    """
    if ans is None:
        return None
    s = ans.strip()
    s = s.replace("{,}", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for pattern in _STRIP:
        s = re.sub(pattern, "", s)
    s = s.rstrip(".").replace(",", "") if re.fullmatch(r"[\d,\.\-]+", s.rstrip(".")) else s
    try:
        value = float(s)
        return repr(round(value, 6))
    except ValueError:
        return s


def answers_match(pred: str | None, gold: str | None, use_math_verify: bool = True) -> bool:
    """True when the predicted final answer is equivalent to the gold answer.

    Uses HuggingFace math-verify when installed (symbolic equivalence);
    otherwise falls back to `normalise_answer` string equality.
    """
    if pred is None or gold is None:
        return False
    if use_math_verify:
        try:
            from math_verify import parse, verify

            g = parse("$" + gold + "$")
            p = parse("$" + pred + "$")
            if g and p:
                return bool(verify(g, p))
        except ImportError:
            pass
        except Exception:  # noqa: BLE001 - parser failures fall back to string match
            pass
    return normalise_answer(pred) == normalise_answer(gold)


# ---------------------------------------------------------------------------
# Value curves
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    Example:
        >>> lo, hi = wilson(0, 16); round(lo, 3), round(hi, 3)
        (0.0, 0.194)
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def value_curve(counts: list) -> np.ndarray:
    """[(n_correct, m), ...] for k = 0..K  ->  array of v_k."""
    return np.array([c / m if m else np.nan for c, m in counts], dtype=float)


def decisive_step(v: np.ndarray) -> int | None:
    """Step k (1-indexed) whose inclusion causes the largest drop v_{k-1} - v_k.

    Earliest step on ties. None when the curve never drops.

    Example:
        >>> decisive_step(np.array([0.8, 0.75, 0.2, 0.1]))
        2
    """
    drops = v[:-1] - v[1:]
    if len(drops) == 0 or np.nanmax(drops) <= 0:
        return None
    return int(np.nanargmax(drops)) + 1


def commitment_step(v: np.ndarray, delta: float) -> int | None:
    """Point of no return: smallest k with v_j <= delta for every j >= k.

    0 means the problem is already lost from the question alone for this
    model (capability-limited). None means the trajectory never commits.

    Example:
        >>> commitment_step(np.array([0.6, 0.5, 0.05, 0.3, 0.0, 0.0]), 0.1)
        4
        >>> commitment_step(np.array([0.05, 0.0, 0.0]), 0.1)
        0
    """
    below = v <= delta
    if not below[-1]:
        return None
    k = len(v) - 1
    while k > 0 and below[k - 1]:
        k -= 1
    return int(k)


def classify_failing_trace(v: np.ndarray, e: int, delta: float) -> str:
    """Where does a failing trajectory commit, relative to its first error e?

    Categories:
      capability-limited   v_0 <= delta: the model rarely solves it from scratch
      latent-commitment    commits before the written error (c < e)
      commits-at-error     c == e
      recoverable-error    commits after the error (c > e): the error alone did not doom it
      never-commits        the curve never stays below delta (noisy / lucky ending)

    Example:
        >>> classify_failing_trace(np.array([0.7, 0.6, 0.0, 0.0]), 2, 0.1)
        'commits-at-error'
        >>> classify_failing_trace(np.array([0.7, 0.05, 0.0, 0.0]), 3, 0.1)
        'latent-commitment'
    """
    c = commitment_step(v, delta)
    if v[0] <= delta:
        return "capability-limited"
    if c is None:
        return "never-commits"
    if c < e:
        return "latent-commitment"
    if c == e:
        return "commits-at-error"
    return "recoverable-error"


def backtrack_values(v: np.ndarray, tokens: np.ndarray | None, points: dict) -> dict:
    """Success probability (and expected continuation tokens) of resampling
    from just before each candidate step b: the value is v_{b-1}.

    `points` maps a policy name to a 1-indexed step b (or None to skip).
    Restart-from-scratch is v_0.
    """
    out = {"restart": (float(v[0]), float(tokens[0]) if tokens is not None else float("nan"))}
    for name, b in points.items():
        if b is None:
            continue
        k = max(0, int(b) - 1)
        out[name] = (float(v[k]), float(tokens[k]) if tokens is not None else float("nan"))
    return out
