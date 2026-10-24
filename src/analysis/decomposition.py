"""
WHAT: Failure decomposition (error onset vs error fatality) and the
      difficulty / generator controls for prefix-conditioned failure detection.

WHY:  Two questions the original study left open decide whether "choke point"
      detection is a real phenomenon or a relabelled difficulty predictor:

      1. Is the first error the point of failure? ProcessBench contains 521
         traces whose first annotated error is followed by a CORRECT final
         answer. An error that does not determine the outcome is, by the
         paper's own definition, not a choke point. We measure how often that
         happens and where.
      2. Does the reasoning prefix carry information beyond the question and
         the generator? Bulut (arXiv:2609.03436) shows pooled probe AUROCs can
         be reproduced by question difficulty alone and fall to chance within
         a problem. We test the same thing for text detectors on ProcessBench:
         question-only and generator/tier-only baselines, the increment of the
         prefix over them, and within-problem pairwise concordance.

HOW:  Grouped K-fold cross-fitting by question id (every trace gets an
      out-of-fold score from a model that never saw its problem), repeated
      over several seeds; question-level cluster bootstrap for intervals;
      paired bootstrap for differences. Every feature transform is fitted on
      the training folds only.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Outcome groups
# ---------------------------------------------------------------------------

CLEAN, RECOVERED, FATAL = "clean", "recovered", "fatal"


def outcome_group(has_error: bool, answer_incorrect: bool) -> str:
    """Three-way outcome: no process error / error but correct answer / error and wrong answer.

    Example:
        >>> outcome_group(False, False), outcome_group(True, False), outcome_group(True, True)
        ('clean', 'recovered', 'fatal')
    """
    if not has_error:
        if answer_incorrect:
            return "clean-but-wrong"  # does not occur in ProcessBench; kept explicit
        return CLEAN
    return FATAL if answer_incorrect else RECOVERED


def add_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach has_error, answer_incorrect, group, e, rho and steps_after_error."""
    out = df.copy()
    out["answer_incorrect"] = (~out["final_correct"].astype(bool)).astype(int)
    out["has_error"] = out["first_error_idx"].notna().astype(int)
    out["group"] = [outcome_group(bool(h), bool(a))
                    for h, a in zip(out["has_error"], out["answer_incorrect"])]
    out["e"] = out["first_error_idx"]
    out["rho"] = out["e"] / out["n_steps"]
    out["steps_after_error"] = out["n_steps"] - out["e"]
    return out


# ---------------------------------------------------------------------------
# Grouped cross-fitting
# ---------------------------------------------------------------------------


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign every row a fold so that all rows of one question share a fold.

    Example:
        >>> f = grouped_folds(np.array(["a", "a", "b", "c", "c", "d"]), 2, 0)
        >>> bool(f[0] == f[1] and f[3] == f[4])
        True
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    permuted = unique[rng.permutation(len(unique))]
    fold_of = {g: i % n_folds for i, g in enumerate(permuted)}
    folds = np.array([fold_of[g] for g in groups])
    for k in range(n_folds):  # leakage assertion: no question on both sides
        assert not set(groups[folds == k]) & set(groups[folds != k])
    return folds


def prefix_by_fraction(steps: list, fraction: float) -> list:
    """First max(1, ceil(f*n)) steps (same rule as Algorithm 1, line 5)."""
    if not steps:
        return []
    return steps[: max(1, math.ceil(fraction * len(steps)))]


class FeatureSpec:
    """A named recipe that turns rows into a (sparse) matrix, fitted on train rows only.

    kind:
      - "text":    TF-IDF over `texts` (one string per row)
      - "onehot":  one-hot over `categories` (one tuple of strings per row)
      - "numeric": standardised dense columns `values` (n_rows x d)
    """

    def __init__(self, name: str, kind: str, data, tfidf_kwargs: dict | None = None):
        self.name, self.kind, self.data = name, kind, data
        self.tfidf_kwargs = tfidf_kwargs or {}

    def fit_transform(self, train_idx: np.ndarray, test_idx: np.ndarray):
        from scipy import sparse

        if self.kind == "text":
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(**self.tfidf_kwargs)
            tr = vec.fit_transform([self.data[i] for i in train_idx])
            te = vec.transform([self.data[i] for i in test_idx])
            return tr, te
        if self.kind == "onehot":
            from sklearn.preprocessing import OneHotEncoder

            enc = OneHotEncoder(handle_unknown="ignore")
            arr = np.asarray(self.data, dtype=object)
            return enc.fit_transform(arr[train_idx]), enc.transform(arr[test_idx])
        if self.kind == "numeric":
            from sklearn.preprocessing import StandardScaler

            arr = np.asarray(self.data, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            sc = StandardScaler().fit(arr[train_idx])
            return sparse.csr_matrix(sc.transform(arr[train_idx])), sparse.csr_matrix(sc.transform(arr[test_idx]))
        raise ValueError(f"unknown feature kind {self.kind}")


def cross_fit_scores(specs: list, y: np.ndarray, folds: np.ndarray, C: float = 1.0,
                     seed: int = 0, rows: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold P(y=1) from logistic regression on the hstack of `specs`.

    `rows` restricts fitting and scoring to a subset (e.g. erroneous traces);
    other rows get NaN.
    """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    scores = np.full(n, np.nan)
    for k in np.unique(folds[rows]):
        test_idx = rows[folds[rows] == k]
        train_idx = rows[folds[rows] != k]
        blocks_tr, blocks_te = [], []
        for spec in specs:
            tr, te = spec.fit_transform(train_idx, test_idx)
            blocks_tr.append(tr)
            blocks_te.append(te)
        X_tr, X_te = sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()
        clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        scores[test_idx] = clf.predict_proba(X_te)[:, 1]
    return scores


# ---------------------------------------------------------------------------
# Within-problem concordance (isolates trajectory signal from question difficulty)
# ---------------------------------------------------------------------------


def within_problem_pairs(groups: np.ndarray, y: np.ndarray) -> list:
    """All (positive, negative) index pairs that share a question id."""
    pairs = []
    frame = pd.DataFrame({"g": groups, "y": y, "i": np.arange(len(y))})
    for _, sub in frame.groupby("g"):
        pos = sub.loc[sub.y == 1, "i"].to_numpy()
        neg = sub.loc[sub.y == 0, "i"].to_numpy()
        pairs += [(p, q) for p in pos for q in neg]
    return pairs


def concordance(scores: np.ndarray, pairs: list) -> float:
    """Fraction of within-problem (fail, success) pairs ordered correctly; ties = 1/2.

    Example:
        >>> concordance(np.array([0.9, 0.1, 0.5, 0.5]), [(0, 1), (2, 3)])
        0.75
    """
    if not pairs:
        return float("nan")
    p = np.array([a for a, _ in pairs])
    q = np.array([b for _, b in pairs])
    return float(np.mean((scores[p] > scores[q]) + 0.5 * (scores[p] == scores[q])))


def concordance_ci(scores_by_name: dict, pairs: list, groups: np.ndarray,
                   n_bootstrap: int, seed: int, reference: str | None = None) -> pd.DataFrame:
    """Within-problem concordance per detector with a problem-level bootstrap,
    plus paired differences against `reference`."""
    by_problem: dict = {}
    for a, b in pairs:
        by_problem.setdefault(groups[a], []).append((a, b))
    keys = list(by_problem)
    rng = np.random.default_rng(seed)
    boots = {name: [] for name in scores_by_name}
    diffs = {name: [] for name in scores_by_name}
    for _ in range(n_bootstrap):
        picked = rng.integers(0, len(keys), len(keys))
        sample = [pr for i in picked for pr in by_problem[keys[i]]]
        values = {name: concordance(s, sample) for name, s in scores_by_name.items()}
        for name, v in values.items():
            boots[name].append(v)
            if reference is not None:
                diffs[name].append(v - values[reference])
    rows = []
    for name, s in scores_by_name.items():
        point = concordance(s, pairs)
        row = {"detector": name, "within_problem_concordance": point,
               "ci_lo": float(np.quantile(boots[name], 0.025)),
               "ci_hi": float(np.quantile(boots[name], 0.975)),
               "n_pairs": len(pairs), "n_problems": len(keys)}
        if reference is not None and name != reference:
            d = np.asarray(diffs[name])
            row.update({"diff_vs_" + reference: point - concordance(scores_by_name[reference], pairs),
                        "diff_ci_lo": float(np.quantile(d, 0.025)),
                        "diff_ci_hi": float(np.quantile(d, 0.975)),
                        "diff_p": float(max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / n_bootstrap))})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-correction markers after the first error
# ---------------------------------------------------------------------------

CORRECTION_PATTERN = re.compile(
    r"\b(wait|mistake|error|incorrect|wrong|oops|re-?check|re-?evaluat\w*|re-?examin\w*|"
    r"re-?calculat\w*|re-?consider\w*|re-?visit\w*|correct(ion|ing|ed)\b|let me correct|actually|"
    r"let'?s (double|re)|let me (double|re)|on second thought|however, (this|that|we))",
    re.IGNORECASE,
)


def has_correction_marker(text: str) -> bool:
    """True when the text contains an explicit self-correction / re-checking phrase.

    Example:
        >>> has_correction_marker("Wait, that is not right. Let me recheck."), has_correction_marker("So x = 4.")
        (True, False)
    """
    return bool(CORRECTION_PATTERN.search(text or ""))
