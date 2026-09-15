"""Stage 08: lead time (E5) + calibration (E6) + transfer (E7) -> F9, F10, F13."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.calibration import reliability_table, threshold_table
from src.analysis.lead_time import (
    lead_time_summary,
    lead_times,
)
from src.analysis.statistics import (
    auroc_with_ci,
    expected_calibration_error,
)
from src.config import load_config
from src.features.surface import batch_surface
from src.utils.io import Manifest, atomic_write_dataframe
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import grouped_split, seed_everything
from src.viz import figures


def risk_series_from_surface(df: pd.DataFrame, seed: float = 42) -> tuple:
    """Fit one surface probe per prefix length; return (risks dict, test info).

    Each trace's risk series is produced by the detector evaluated at its own
    prefixes - the offline analogue of Algorithm 1's online monitor loop.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler


    steps_list = list(df["steps"])
    max_n = max((len(s) for s in steps_list), default=0)
    y = df["has_error"].astype(int).to_numpy()
    groups = df["question_id"].to_numpy()
    tr, te = grouped_split(groups, 0.25, seed, context="lead_time_risks")
    probes = {}
    for k in range(1, max_n + 1):
        X = batch_surface([s[:k] for s in steps_list])
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        probes[k] = (scaler, clf)
    risks: dict = {}
    for i in df.index:
        series = []
        for k in range(1, len(df.loc[i, "steps"]) + 1):
            scaler, clf = probes[k]
            x = batch_surface([df.loc[i, "steps"][:k]])
            series.append(float(clf.predict_proba(scaler.transform(x))[0, 1]))
        risks[df.loc[i, "trace_id"]] = series
    return risks, (y[te], te)


def main() -> None:
    parser = argparse.ArgumentParser(description="08: E5+E6+E7 analysis")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("08_lead_time_analysis", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    prefix = "synthetic_" if cfg.data.synthetic else ""
    df = pd.read_csv(Path(cfg.data.processed_dir) / f"{prefix}processed_traces.csv")
    df["steps"] = df["steps"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    risks, (y_te, te_idx) = risk_series_from_surface(df, cfg.seed)

    # E5: lead time sweep.
    lead_df = lead_times(df, risks, list(cfg.intervention.thresholds))
    lead_summary = lead_time_summary(lead_df)
    atomic_write_dataframe(lead_df, Path(cfg.outputs.tables_dir) / f"{prefix}table_lead_time.csv")
    atomic_write_dataframe(lead_summary, Path(cfg.outputs.tables_dir) / f"{prefix}table_lead_summary.csv")
    figures.plot_f9(lead_df, cfg.outputs.figures_dir)

    # E6: calibration on the test rows (full-prefix risks).
    te_ids = df.iloc[te_idx]["trace_id"].tolist()
    probs = np.array([risks[t][-1] if risks[t] else 0.5 for t in te_ids])
    ece = expected_calibration_error(y_te, probs, cfg.metrics.ece_bins)
    rel = reliability_table(y_te, probs, cfg.metrics.ece_bins)
    atomic_write_dataframe(rel, Path(cfg.outputs.tables_dir) / f"{prefix}table_reliability.csv")
    thresh = threshold_table(y_te, probs, list(cfg.intervention.thresholds))
    atomic_write_dataframe(thresh, Path(cfg.outputs.tables_dir) / f"{prefix}table_thresholds.csv")
    figures.plot_f10(rel, ece["ece"], cfg.outputs.figures_dir)

    # E7: cross-domain transfer - train on math-like tiers, eval on held-out
    # tier as the available proxy; external corpora when their loaders resolve.
    transfer_rows = []
    subsets = sorted(df["subset"].unique())
    if len(subsets) > 1:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        for held in subsets:
            tr_df = df[df["subset"] != held]
            te_df = df[df["subset"] == held]
            Xtr, Xte = batch_surface(list(tr_df["steps"])), batch_surface(list(te_df["steps"]))
            ytr, yte = tr_df["has_error"].astype(int).to_numpy(), te_df["has_error"].astype(int).to_numpy()
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, random_state=cfg.seed).fit(
                scaler.transform(Xtr), ytr)
            stats = auroc_with_ci(yte, clf.predict_proba(scaler.transform(Xte))[:, 1],
                                  cfg.metrics.bootstrap_resamples, cfg.seed)
            transfer_rows.append({"evaluation_domain": held, **stats})
    transfer_df = pd.DataFrame(transfer_rows)
    atomic_write_dataframe(transfer_df, Path(cfg.outputs.tables_dir) / f"{prefix}table_transfer.csv")
    figures.plot_f13(transfer_df if not transfer_df.empty else
                     pd.DataFrame(columns=["evaluation_domain", "auroc"]),
                     cfg.outputs.figures_dir)

    summary = [
        (
            f"Lead time: P(Delta>0) @ tau={cfg.intervention.thresholds[0]} = "
            f"{lead_summary['frac_warned_before_error'].iloc[0]:.3f}"
        ) if not lead_summary.empty else "Lead time: NOT_RUN",
        f"ECE (surface detector, {cfg.metrics.ece_bins} bins): {ece['ece']:.4f}",
        (
            f"Transfer domains evaluated: {len(transfer_df)} "
            f"(drop vs in-domain reported in table_transfer.csv)"
        ),
        "Figures: F9 (lead time), F10 (reliability), F13 (transfer)",
    ]
    log.info("analysis complete")
    manifest.log("08_lead_time_analysis", args.config, "OK",
                 {"ece": ece["ece"], "transfer_domains": len(transfer_df)})
    print_summary_block("08_lead_time_analysis", summary)


if __name__ == "__main__":
    main()
