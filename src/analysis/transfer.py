"""
WHAT: Cross-domain transfer (E7) - train a full-trace detector on mathematical
      reasoning (ProcessBench), evaluate it on other reasoning domains.

WHY:  Paper §Limitations and Orgad et al. (2024): probes trained on one
      distribution transfer poorly to another. Measuring the AUROC drop from
      in-domain to out-of-domain tells the reader whether the detector learned
      something about reasoning failure in general or about the surface of
      maths solutions. A large drop is an expected, publishable finding.

HOW:  `fit_full_trace_detector` fits surface or lexical logistic regression on
      the TRAIN questions of the source domain. In-domain AUROC is measured on
      the held-out source TEST questions; transfer AUROC on every target
      subset (and the pooled target). The drop is in-domain minus transfer.
      Target domains share no question ids with the source by construction,
      so no split is needed on their side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_full_trace_detector(train_df: pd.DataFrame, target: str, representation: str,
                            seed: int, max_features: int = 20000, ngram_range=(1, 2),
                            min_df: int = 2):
    """Fit a whole-trace detector; returns a `score(df) -> np.ndarray` callable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_train = train_df[target].astype(int).to_numpy()
    if representation == "surface":
        from src.features.surface import batch_surface

        scaler = StandardScaler().fit(batch_surface(list(train_df["steps"])))
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
            scaler.transform(batch_surface(list(train_df["steps"]))), y_train)

        def score(frame: pd.DataFrame) -> np.ndarray:
            return clf.predict_proba(scaler.transform(batch_surface(list(frame["steps"]))))[:, 1]

        return score
    if representation == "lexical":
        from src.features.lexical import fit_tfidf, transform_tfidf

        vectoriser, matrix = fit_tfidf([" ".join(s) for s in train_df["steps"]],
                                       max_features=max_features,
                                       ngram_range=tuple(ngram_range), min_df=min_df)
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(matrix, y_train)

        def score(frame: pd.DataFrame) -> np.ndarray:
            texts = [" ".join(s) for s in frame["steps"]]
            return clf.predict_proba(transform_tfidf(vectoriser, texts))[:, 1]

        return score
    raise ValueError(f"Unknown representation '{representation}'.")


def transfer_table(source_df: pd.DataFrame, target_df: pd.DataFrame, target: str,
                   split: tuple, representations: list, seed: int, n_bootstrap: int,
                   source_name: str, **tfidf_kwargs) -> pd.DataFrame:
    """In-domain vs out-of-domain AUROC per representation and target subset.

    Returns:
        Rows with representation, evaluation_domain, auroc, ci_lo, ci_hi,
        n_eval, positive_rate, in_domain_auroc, auroc_drop.
    """
    from src.analysis.statistics import auroc_with_ci

    train_idx, test_idx = split
    train_df = source_df.iloc[train_idx]
    source_test = source_df.iloc[test_idx]
    rows = []
    for representation in representations:
        score = fit_full_trace_detector(train_df, target, representation, seed, **tfidf_kwargs)
        in_domain = auroc_with_ci(source_test[target].astype(int).to_numpy(),
                                  score(source_test), n_bootstrap, seed)
        rows.append({"representation": representation,
                     "evaluation_domain": f"{source_name} (in-domain test)",
                     **in_domain, "n_eval": int(len(source_test)),
                     "positive_rate": float(source_test[target].mean()),
                     "in_domain_auroc": in_domain["auroc"], "auroc_drop": 0.0})
        domains = [(f"BBM: {name}", group) for name, group in target_df.groupby("subset")]
        domains.append(("BBM: all tasks", target_df))
        for domain_name, group in domains:
            y_eval = group[target].astype(int).to_numpy()
            if len(np.unique(y_eval)) < 2:
                continue
            stats = auroc_with_ci(y_eval, score(group), n_bootstrap, seed)
            rows.append({"representation": representation, "evaluation_domain": domain_name,
                         **stats, "n_eval": int(len(group)),
                         "positive_rate": float(np.mean(y_eval)),
                         "in_domain_auroc": in_domain["auroc"],
                         "auroc_drop": in_domain["auroc"] - stats["auroc"]})
    return pd.DataFrame(rows)


def leave_one_tier_out(df: pd.DataFrame, target: str, representation: str, seed: int,
                       n_bootstrap: int, **tfidf_kwargs) -> pd.DataFrame:
    """Within-maths transfer: train on three ProcessBench tiers, test on the fourth.

    Questions never repeat across tiers (each tier is a different source
    benchmark), which is checked explicitly rather than assumed.
    """
    from src.analysis.statistics import auroc_with_ci
    from src.utils.seeds import assert_no_leakage

    rows = []
    for held_out in sorted(df["subset"].unique()):
        train_df, eval_df = df[df["subset"] != held_out], df[df["subset"] == held_out]
        assert_no_leakage(train_df["question_id"], eval_df["question_id"],
                          context=f"leave-one-tier-out {held_out}")
        y_eval = eval_df[target].astype(int).to_numpy()
        if len(np.unique(y_eval)) < 2:
            continue
        score = fit_full_trace_detector(train_df, target, representation, seed, **tfidf_kwargs)
        rows.append({"representation": representation, "held_out_tier": held_out,
                     **auroc_with_ci(y_eval, score(eval_df), n_bootstrap, seed),
                     "n_eval": int(len(eval_df))})
    return pd.DataFrame(rows)
