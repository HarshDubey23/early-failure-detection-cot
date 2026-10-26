"""
WHAT: Paired statistics for comparing policies item by item.

WHY:  All policies are evaluated on the same items and the same bank samples, so differences are
      paired. Items that share a source (the answerable and unanswerable HotpotQA versions of one
      question) are not independent, so resampling and sign-flipping happen at the cluster level.

HOW:  cluster bootstrap percentile intervals; a cluster sign-flip permutation test for the mean
      paired difference; Holm's step-down adjustment; TOST for equivalence within a margin.
"""
from __future__ import annotations

import numpy as np


def cluster_means(values: np.ndarray, clusters: np.ndarray) -> tuple:
    """Per-cluster sums and counts (so the bootstrap mean is item-weighted)."""
    keys, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys)).astype(float)
    return sums, counts


def boot_ci(values, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Mean of `values` and its cluster-bootstrap (1 - alpha) percentile interval.

    Example:
        >>> m, lo, hi = boot_ci(np.r_[np.zeros(50), np.ones(50)], np.arange(100), n_boot=500)
        >>> round(m, 2), lo < 0.5 < hi
        (0.5, True)
    """
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v, c = v[ok], np.asarray(clusters)[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    sums, counts = cluster_means(v, c)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def signflip_p(diff, clusters, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for mean(diff) = 0, flipping the sign of whole clusters.

    Example:
        >>> signflip_p(np.full(40, 0.2) + np.random.default_rng(0).normal(0, 0.05, 40), np.arange(40)) < 0.01
        True
    """
    d = np.asarray(diff, dtype=float)
    ok = ~np.isnan(d)
    d, c = d[ok], np.asarray(clusters)[ok]
    if len(d) == 0:
        return float("nan")
    sums, counts = cluster_means(d, c)
    n = counts.sum()
    obs = abs(sums.sum() / n)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    stats = np.abs((flips * sums).sum(axis=1) / n)
    return float((1 + (stats >= obs - 1e-12).sum()) / (n_perm + 1))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values.

    Example:
        >>> holm({"a": 0.01, "b": 0.04, "c": 0.03})
        {'a': 0.03, 'c': 0.06, 'b': 0.06}
    """
    out = {k: float("nan") for k, p in pvals.items() if p != p}  # an empty comparison is not in the family
    items = sorted(((k, p) for k, p in pvals.items() if p == p), key=lambda kv: kv[1])
    m, running = len(items), 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = round(running, 10)
    return out


def tost(diff, clusters, margin: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Equivalence within +-margin via the 90% cluster-bootstrap interval of the mean difference."""
    m, lo, hi = boot_ci(diff, clusters, n_boot=n_boot, seed=seed, alpha=0.10)
    if lo > -margin and hi < margin:
        verdict = "equivalent"
    elif hi < -margin or lo > margin:
        verdict = "different"
    else:
        verdict = "inconclusive"
    return {"mean": m, "lo90": lo, "hi90": hi, "margin": margin, "verdict": verdict}


def binary_agreement(pred, truth) -> dict:
    """Precision / recall / F1 / accuracy / Cohen's kappa of binary predictions."""
    p, t = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    tp, fp, fn, tn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()
    n = len(p)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else float("nan")
    kappa = (acc - pe) / (1 - pe) if n and pe < 1 else float("nan")
    return {"n": int(n), "precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc),
            "kappa": float(kappa), "rate_pred": float(p.mean()) if n else float("nan"),
            "rate_true": float(t.mean()) if n else float("nan")}
