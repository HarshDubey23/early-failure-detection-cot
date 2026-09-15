"""
WHAT: Statistical tests (E2) plus AUROC-with-CI and ECE primitives.

WHY:  Turns "the plot looks different" into a defensible claim. A p-value
      without an effect size is incomplete (brief §4), so every test returns
      both. Bootstrap CIs make AUROC numbers honest about sampling noise.

HOW:  Thin, typed wrappers over scipy/sklearn with fixed seeds. Each function
      documents exactly what it measures in its docstring.
"""

from __future__ import annotations

import numpy as np


def ks_uniform(values: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test of relative onset against Uniform(0,1).

    Measures whether errors concentrate in a region of the trajectory rather
    than occurring at arbitrary positions. Effect size D is the max CDF gap.
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    result = stats.kstest(values, "uniform")
    return {"statistic_D": float(result.statistic), "p_value": float(result.pvalue)}


def mann_whitney_length(lengths_err: np.ndarray, lengths_ok: np.ndarray) -> dict:
    """Mann-Whitney U on trace length, erroneous vs correct.

    Measures whether failing traces differ in length (the length confound).
    Effect size is rank-biserial r = 1 - 2U/(n1*n2).
    """
    from scipy import stats

    result = stats.mannwhitneyu(lengths_err, lengths_ok, alternative="two-sided")
    n1, n2 = len(lengths_err), len(lengths_ok)
    rank_biserial = 1.0 - 2.0 * float(result.statistic) / max(1, n1 * n2)
    return {
        "U": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial_r": float(rank_biserial),
    }


def kruskal_across_tiers(groups: list) -> dict:
    """Kruskal-Wallis on relative onset across difficulty tiers.

    Measures whether onset position shifts systematically with difficulty.
    Effect size eta^2 = (H - k + 1) / (n - k).
    """
    from scipy import stats

    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[~np.isnan(g)] for g in cleaned]
    result = stats.kruskal(*cleaned)
    k_groups = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    eta_sq = (float(result.statistic) - k_groups + 1) / max(1, n_total - k_groups)
    return {
        "H": float(result.statistic),
        "p_value": float(result.pvalue),
        "eta_squared": float(max(0.0, eta_sq)),
    }


def auroc_with_ci(y_true: np.ndarray, scores: np.ndarray,
                  n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """AUROC with seeded bootstrap 95% CI.

    Measures P(a random failing trace scores above a random succeeding one).
    0.5 = random, 0.7 = useful, 0.8 = strong. Reported instead of accuracy
    because classes are imbalanced.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    point = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores[idx]))
    return {
        "auroc": point,
        "ci_lo": float(np.quantile(boot, 0.025)) if boot else float("nan"),
        "ci_hi": float(np.quantile(boot, 0.975)) if boot else float("nan"),
    }


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 1000,
                      seed: int = 42) -> tuple:
    """Mean with bootstrap 95% CI. Returns (mean, lo, hi); NaNs on empty input."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(values, len(values), replace=True)))
            for _ in range(n_bootstrap)]
    return (float(np.mean(values)), float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 10) -> dict:
    """Expected Calibration Error with `n_bins` equal-width bins.

    Measures whether predicted risks match empirical frequencies. A detector
    that is accurate but uncalibrated cannot safely drive a threshold policy.
    """
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    from itertools import pairwise

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, details = 0.0, []
    for lo, hi in pairwise(edges):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(probs[mask]))
        weight = float(mask.sum() / len(probs))
        ece += weight * abs(acc - conf)
        details.append({"lo": lo, "hi": hi, "acc": acc, "conf": conf,
                        "weight": weight, "n": int(mask.sum())})
    return {"ece": float(ece), "bins": details}
