"""Stage 05: baseline detectors + predictability curve (E3) -> F6, F7."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.predictability import (
    run_prefix_curve,
    run_tfidf_prefix_curve,
)
from src.analysis.statistics import auroc_with_ci
from src.config import load_config
from src.features.surface import batch_surface
from src.utils.io import Manifest, atomic_write_dataframe
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import grouped_split, seed_everything
from src.viz import figures


def main() -> None:
    parser = argparse.ArgumentParser(description="05: baselines + E3 curve")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("05_train_baselines", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    prefix = "synthetic_" if cfg.data.synthetic else ""
    df = pd.read_csv(Path(cfg.data.processed_dir) / f"{prefix}processed_traces.csv")
    df["steps"] = df["steps"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    y = df["has_error"].astype(int).to_numpy()

    # Baseline 1: surface features (full trace), grouped split.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = batch_surface(list(df["steps"]))
    tr, te = grouped_split(df["question_id"].to_numpy(), cfg.split.test_size,
                           cfg.seed, context="surface_baseline")
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    clf.fit(scaler.transform(X[tr]), y[tr])
    surface_scores = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    surface = auroc_with_ci(y[te], surface_scores,
                            cfg.metrics.bootstrap_resamples, cfg.seed)

    # E3: prefix curves for surface + lexical (same harness family).
    surface_curve = run_prefix_curve(
        df, lambda steps: __import__("src.features.surface", fromlist=["surface_features"]).surface_features(steps),
        cfg.prefix_grid, cfg.seed, cfg.split.test_size,
        cfg.metrics.bootstrap_resamples, representation="surface",
    )
    lexical_curve = run_tfidf_prefix_curve(
        df, cfg.prefix_grid, cfg.seed, cfg.split.test_size,
        cfg.metrics.bootstrap_resamples, cfg.baselines.tfidf_max_features,
        tuple(cfg.baselines.tfidf_ngram_range), cfg.baselines.tfidf_min_df,
    )
    curve = pd.concat([surface_curve, lexical_curve], ignore_index=True)
    atomic_write_dataframe(curve, Path(cfg.outputs.tables_dir) / f"{prefix}table2_prefix_auroc.csv")
    (Path(cfg.outputs.tables_dir) / f"{prefix}table2_prefix_auroc.tex").write_text(
        curve.to_latex(index=False, float_format="%.4f"))

    # F6 needs test-set scores at full prefix: reuse surface scores + lexical.
    figures.plot_f7(curve, cfg.outputs.figures_dir)
    from sklearn.metrics import roc_curve  # noqa: F401 (kept for F6 data export)
    pd.DataFrame({"y_true": y[te], "surface_score": surface_scores}).to_csv(
        Path(cfg.outputs.tables_dir) / f"{prefix}f6_scores.csv", index=False)
    figures.plot_f6(y[te], {"Surface": surface_scores}, cfg.outputs.figures_dir)

    best = curve.loc[curve["auroc"].idxmax()]
    summary = [
        (
            f"Surface-only AUROC (full trace): {surface['auroc']:.4f} "
            f"[{surface['ci_lo']:.3f}, {surface['ci_hi']:.3f}]"
        ),
        (
            f"Best prefix AUROC              : {best['auroc']:.4f} "
            f"({best['representation']} @ {best['fraction_observed']:.0%})"
        ),
        "Figures: F6 (ROC), F7 (predictability curve, PNG+PDF)",
    ]
    log.info("baselines complete: %s", summary)
    manifest.log("05_train_baselines", args.config, "OK",
                 {"surface_auroc": surface["auroc"], "best_auroc": float(best["auroc"])})
    print_summary_block("05_train_baselines", summary)


if __name__ == "__main__":
    main()
