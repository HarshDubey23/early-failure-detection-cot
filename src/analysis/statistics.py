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
    return {"statistic_D": float(result.statistic), "p_value": float(result.pvalue),
            "n": int(len(values))}


def mann_whitney_length(lengths_err: np.ndarray, lengths_ok: np.ndarray) -> dict:
    """Mann-Whitney U on trace length, erroneous vs correct.

    Measures whether failing traces differ in length (the length confound).
    Effect size is the rank-biserial correlation r = 2*U1/(n1*n2) - 1, where
    U1 is the statistic for the ERRONEOUS group: r > 0 means erroneous traces
    tend to be longer, r < 0 shorter, r = 0 no tendency.

    Example:
        >>> round(mann_whitney_length(np.array([5, 6, 7]), np.array([1, 2, 3]))["rank_biserial_r"], 2)
        1.0
    """
    from scipy import stats

    lengths_err = np.asarray(lengths_err, dtype=float)
    lengths_ok = np.asarray(lengths_ok, dtype=float)
    result = stats.mannwhitneyu(lengths_err, lengths_ok, alternative="two-sided")
    n1, n2 = len(lengths_err), len(lengths_ok)
    rank_biserial = 2.0 * float(result.statistic) / max(1, n1 * n2) - 1.0
    return {
        "U": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial_r": float(rank_biserial),
        "median_len_erroneous": float(np.median(lengths_err)) if n1 else float("nan"),
        "median_len_correct": float(np.median(lengths_ok)) if n2 else float("nan"),
        "n_erroneous": int(n1),
        "n_correct": int(n2),
    }


def _null_cdf_uniform_steps(grid: np.ndarray, n_steps: np.ndarray) -> np.ndarray:
    """CDF of rho under e ~ Uniform{1..n_i} per trace, averaged over traces.

    For one trace, P(e/n <= x) = floor(x * n) / n (clipped to [0, 1]); the
    pooled null CDF is the mean of those step functions.
    """
    counts = np.floor(np.outer(grid, n_steps) + 1e-9)
    return np.mean(np.clip(counts, 0, n_steps) / n_steps, axis=1)


def _ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)
    return np.searchsorted(ordered, grid + 1e-9, side="right") / len(ordered)


def ks_discrete_uniform_null(first_error_idx: np.ndarray, n_steps: np.ndarray,
                             n_simulations: int = 1000, seed: int = 42) -> dict:
    """Monte Carlo goodness-of-fit test of rho against "error step uniform over the trace's own steps".

    Why this exists: rho = e/n lives on a lattice ({1/n, ..., 1}) and can never
    be below 1/n, so errors spread perfectly evenly over steps do NOT give a
    continuous Uniform(0,1) rho - short traces alone make a KS test against
    Uniform(0,1) reject. The appropriate null keeps every trace's observed n
    and draws e ~ Uniform{1..n}.

    Two statistics are compared with their Monte Carlo null distributions:
      - D_vs_null: max distance between the observed ECDF of rho and the null
        CDF (a KS-type distance to the correct discrete null), and
      - the mean of rho (two-sided), whose sign says whether errors sit
        earlier or later than the uniform-step null.
    For reference it also reports how large the KS distance to Uniform(0,1)
    typically is under the null (null_D_uniform_q95).

    Returns:
        Dict of observed statistics, null summaries and Monte Carlo p-values.
    """
    from scipy import stats

    first_error_idx = np.asarray(first_error_idx, dtype=float)
    n_steps = np.asarray(n_steps, dtype=float)
    observed_rho = first_error_idx / n_steps
    grid = np.unique(np.concatenate([np.arange(1, n + 1) / n for n in np.unique(n_steps)]))
    null_cdf = _null_cdf_uniform_steps(grid, n_steps)
    null_mean = float(np.mean((n_steps + 1.0) / (2.0 * n_steps)))

    observed_d_null = float(np.max(np.abs(_ecdf(observed_rho, grid) - null_cdf)))
    observed_mean = float(observed_rho.mean())
    rng = np.random.default_rng(seed)
    d_null, means, d_uniform = [], [], []
    for _ in range(n_simulations):
        simulated_rho = (np.floor(rng.random(len(n_steps)) * n_steps) + 1.0) / n_steps
        d_null.append(float(np.max(np.abs(_ecdf(simulated_rho, grid) - null_cdf))))
        means.append(float(simulated_rho.mean()))
        d_uniform.append(float(stats.kstest(simulated_rho, "uniform").statistic))
    d_null, means, d_uniform = np.asarray(d_null), np.asarray(means), np.asarray(d_uniform)
    return {
        "observed_D_vs_null": observed_d_null,
        "null_D_vs_null_q95": float(np.quantile(d_null, 0.95)),
        "p_value": float((1 + np.sum(d_null >= observed_d_null)) / (1 + n_simulations)),
        "observed_mean_rho": observed_mean,
        "null_mean_rho": null_mean,
        "p_value_mean": float((1 + np.sum(np.abs(means - null_mean) >= abs(observed_mean - null_mean)))
                              / (1 + n_simulations)),
        "direction": "earlier than null" if observed_mean < null_mean else "later than null",
        "observed_D_uniform": float(stats.kstest(observed_rho, "uniform").statistic),
        "null_D_uniform_mean": float(d_uniform.mean()),
        "null_D_uniform_q95": float(np.quantile(d_uniform, 0.95)),
        "n_simulations": int(n_simulations),
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
        "k_groups": int(k_groups),
        "n": int(n_total),
    }


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank formula (ties get average ranks).

    AUROC = (sum of the ranks of positives - n1 (n1 + 1) / 2) / (n1 n0): the
    fraction of (positive, negative) pairs ordered correctly, ties counting
    one half. Identical to sklearn's roc_auc_score, but fast enough to call
    thousands of times inside a bootstrap.

    Example:
        >>> auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.75
    """
    from scipy.stats import rankdata

    y_true = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _resample_indices(rng, n_rows: int, group_rows: list | None) -> np.ndarray:
    """One bootstrap resample: rows (plain) or whole question groups (cluster)."""
    if group_rows is None:
        return rng.integers(0, n_rows, n_rows)
    picked = rng.integers(0, len(group_rows), len(group_rows))
    return np.concatenate([group_rows[i] for i in picked])


def _group_rows(groups) -> list | None:
    """Row indices per group, so resampling a group takes all of its rows."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
    return np.split(order, boundaries)


def auroc_with_ci(y_true: np.ndarray, scores: np.ndarray,
                  n_bootstrap: int = 1000, seed: int = 42, groups=None) -> dict:
    """AUROC with a seeded bootstrap 95% CI.

    Measures P(a random failing trace scores above a random succeeding one);
    0.5 is chance. When `groups` (question ids) is given, the bootstrap
    resamples whole questions (a cluster bootstrap): traces of the same
    problem are correlated, so resampling single traces would understate the
    uncertainty.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    point = auroc(y_true, scores)
    rng = np.random.default_rng(seed)
    rows = _group_rows(groups)
    boot = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(rng, len(y_true), rows)
        value = auroc(y_true[idx], scores[idx])
        if not np.isnan(value):
            boot.append(value)
    return {
        "auroc": point,
        "ci_lo": float(np.quantile(boot, 0.025)) if boot else float("nan"),
        "ci_hi": float(np.quantile(boot, 0.975)) if boot else float("nan"),
    }


def paired_auroc_difference(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                            n_bootstrap: int = 1000, seed: int = 42, groups=None) -> dict:
    """AUROC(a) - AUROC(b) on the SAME test traces, with a paired bootstrap.

    Both detectors are re-scored on each resample, so the correlation between
    them (same traces, same labels) is respected; this is a proper test of
    whether one detector beats another, which "the two CIs overlap" is not.
    The two-sided p-value is 2 x the smaller tail of the bootstrap
    distribution around 0 (bounded below by the resolution 1/n_bootstrap).
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    difference = auroc(y_true, scores_a) - auroc(y_true, scores_b)
    rng = np.random.default_rng(seed)
    rows = _group_rows(groups)
    boot = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(rng, len(y_true), rows)
        a, b = auroc(y_true[idx], scores_a[idx]), auroc(y_true[idx], scores_b[idx])
        if not (np.isnan(a) or np.isnan(b)):
            boot.append(a - b)
    boot = np.asarray(boot)
    tail = min(np.mean(boot <= 0), np.mean(boot >= 0)) if len(boot) else float("nan")
    return {
        "difference": float(difference),
        "ci_lo": float(np.quantile(boot, 0.025)) if len(boot) else float("nan"),
        "ci_hi": float(np.quantile(boot, 0.975)) if len(boot) else float("nan"),
        "p_value": float(max(2.0 * tail, 1.0 / max(1, len(boot)))) if len(boot) else float("nan"),
        "n_bootstrap": int(len(boot)),
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
