"""Stage 05: baselines + predictability curves (E3) -> tables, F6, F7. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import predictability as pred
from src.analysis.statistics import auroc_with_ci
from src.config import load_config
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures


def main() -> None:
    parser = argparse.ArgumentParser(description="05: baselines + E3 curves")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("05_train_baselines", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables, figs = Path(cfg.outputs.tables_dir), cfg.outputs.figures_dir

    df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl")
    # ONE grouped split by question id, shared by every representation and target.
    split = pred.make_split(df, cfg.seed, cfg.split.test_size, context="E3 shared split")
    train_idx, test_idx = split
    tfidf = {"max_features": cfg.baselines.tfidf_max_features,
             "ngram_range": tuple(cfg.baselines.tfidf_ngram_range),
             "min_df": cfg.baselines.tfidf_min_df, "C": cfg.baselines.logistic_C}
    common = {"prefix_grid": cfg.prefix_grid, "split": split, "seed": cfg.seed,
              "n_bootstrap": cfg.metrics.bootstrap_resamples}

    curves, full_scores, scores_primary = [], {}, {}
    for target in (cfg.detector.target, cfg.detector.secondary_target):
        primary = target == cfg.detector.target
        log.info("E3 curves for target=%s", target)
        runs = [
            ("surface", lambda t=target: pred.run_surface_curve(df, t, C=cfg.baselines.logistic_C, **common)),
            ("length-only", lambda t=target: pred.run_length_only_curve(df, t, **common)),
            ("lexical", lambda t=target: pred.run_tfidf_curve(df, t, **tfidf, **common)),
        ]
        if primary:  # nonlinear / sequence extensions on the paper's target only
            runs += [
                ("surface-MLP", lambda t=target: pred.run_mlp_curve(
                    df, t, hidden=tuple(cfg.probes.mlp_hidden), max_iter=cfg.probes.mlp_max_iter, **common)),
                ("surface-GRU", lambda t=target: pred.run_gru_curve(
                    df, t, hidden_size=cfg.probes.gru_hidden, epochs=cfg.probes.gru_epochs,
                    lr=cfg.probes.gru_lr, **common)),
            ]
        for name, run in runs:
            curve, scores_by_fraction = run()
            curves.append(curve)
            if primary:
                scores_primary[name] = scores_by_fraction
                full_scores[name] = scores_by_fraction[max(cfg.prefix_grid)]
            log.info("  %s done (AUROC at 100%% = %.4f)", name, curve["auroc"].iloc[-1])

    curve = pd.concat(curves, ignore_index=True)
    atomic_write_dataframe(curve, tables / "table2_prefix_auroc.csv")
    atomic_write_text(tables / "table2_prefix_auroc.tex", curve.to_latex(index=False, float_format="%.3f"))

    # Paired-bootstrap tests: is one detector really better on the SAME test traces?
    groups_test = df["question_id"].to_numpy()[test_idx]
    y_primary = df[cfg.detector.target].astype(int).to_numpy()[test_idx]
    comparisons = pred.paired_comparisons(
        y_primary, groups_test, scores_primary, [tuple(pair) for pair in cfg.robustness.comparison_pairs],
        cfg.prefix_grid, cfg.metrics.bootstrap_resamples, cfg.seed, cfg.detector.target)
    atomic_write_dataframe(comparisons, tables / "table2b_paired_auroc_differences.csv")

    # Stability: the same curves on independent grouped splits.
    robustness = pred.split_robustness(df, cfg.detector.target, cfg.robustness.fractions,
                                       cfg.robustness.split_seeds, cfg.split.test_size, tfidf,
                                       cfg.baselines.logistic_C)
    atomic_write_dataframe(robustness, tables / "table2c_split_robustness.csv")
    atomic_write_dataframe(pred.summarise_robustness(robustness), tables / "table2c_split_robustness_summary.csv")

    # Majority-class baseline: a constant score; AUROC computed, not assumed.
    y_test = df[cfg.detector.target].astype(int).to_numpy()[test_idx]
    prevalence = float(df[cfg.detector.target].astype(int).to_numpy()[train_idx].mean())
    majority = auroc_with_ci(y_test, np.full(len(y_test), prevalence), 10, cfg.seed)
    status = pd.DataFrame([
        {"representation": "majority-class", "status": "OK", "auroc_full_prefix": majority["auroc"],
         "note": f"constant score = train prevalence {prevalence:.3f}"},
        {"representation": "uncertainty (token entropy / logit margin)", "status": "NOT_RUN",
         "auroc_full_prefix": float("nan"),
         "note": "ProcessBench stores step text only; no token log-probabilities exist for its "
                 "generators. Requires re-generating traces with an open model (GPU)."},
        {"representation": "hidden-state (linear probe)", "status": "NOT_RUN",
         "auroc_full_prefix": float("nan"),
         "note": "Requires stage 06 activations (GPU); see table3_probe_heatmap.csv."},
    ])
    atomic_write_dataframe(status, tables / "table2_representation_status.csv")

    scores_frame = pd.DataFrame({"trace_id": df["trace_id"].to_numpy()[test_idx],
                                 "question_id": df["question_id"].to_numpy()[test_idx],
                                 "y_true": y_test, **full_scores})
    atomic_write_dataframe(scores_frame, tables / "full_prefix_test_scores.csv")

    figures.plot_f6(y_test, full_scores, figs)
    primary_curve = curve[curve["target"] == cfg.detector.target]
    figures.plot_f7(primary_curve, figs)
    figures.plot_f7(curve[curve["target"] == cfg.detector.secondary_target], figs,
                    name="fig7b_auroc_vs_prefix_process_error")

    def auroc_at(representation: str, fraction: float) -> str:
        row = primary_curve[(primary_curve["representation"] == representation)
                            & np.isclose(primary_curve["fraction_observed"], fraction)]
        if row.empty:
            return "n/a"
        r = row.iloc[0]
        return f"{r['auroc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"

    summary = [f"Split: {len(train_idx)} train / {len(test_idx)} test traces, "
               f"question-disjoint (asserted). Target y = {cfg.detector.target}."]
    for representation in ("lexical", "surface", "length-only", "surface-MLP", "surface-GRU"):
        summary.append(f"{representation:<12} AUROC @25%: {auroc_at(representation, 0.25):<24} "
                       f"@100%: {auroc_at(representation, 1.0)}")
    for row in comparisons[np.isclose(comparisons["fraction_observed"], 0.25)
                           | np.isclose(comparisons["fraction_observed"], 1.0)].itertuples():
        summary.append(f"paired {row.comparison:<24} @{100 * row.fraction_observed:g}%: "
                       f"{row.difference:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] p={row.p_value:.3f}")
    summary += [f"majority-class AUROC: {majority['auroc']:.3f}",
                "uncertainty + hidden-state: NOT_RUN (see table2_representation_status.csv)",
                "Figures: F6 (ROC), F7 (curve, paper target), F7b (curve, process-error target)"]
    for line in summary:
        log.info(line)
    manifest.log("05_train_baselines", args.config, "OK",
                 {"n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
    print_summary_block("05_train_baselines", summary)


if __name__ == "__main__":
    main()
