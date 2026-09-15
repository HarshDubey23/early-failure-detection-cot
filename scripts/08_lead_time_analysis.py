"""Stage 08: lead time (E5) + calibration (E6) + cross-domain transfer (E7) -> F9, F10, F13."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.calibration import reliability_table, threshold_table
from src.analysis.lead_time import (
    false_alarm_rates,
    lead_time_summary,
    lead_times,
    online_auroc_by_step,
    stepwise_risk_series,
)
from src.analysis.predictability import make_split
from src.analysis.statistics import expected_calibration_error
from src.analysis.transfer import leave_one_tier_out, transfer_table
from src.config import load_config
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures

PROBABILISTIC_DETECTORS = ("lexical", "surface", "surface-MLP", "surface-GRU")


def main() -> None:
    parser = argparse.ArgumentParser(description="08: E5 + E6 + E7")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("08_lead_time_analysis", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables, figs = Path(cfg.outputs.tables_dir), cfg.outputs.figures_dir
    target = cfg.detector.target
    tfidf = {"max_features": cfg.baselines.tfidf_max_features,
             "ngram_range": tuple(cfg.baselines.tfidf_ngram_range),
             "min_df": cfg.baselines.tfidf_min_df}

    df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl")
    transfer_df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_transfer.jsonl")
    split = make_split(df, cfg.seed, cfg.split.test_size, context="E3 shared split")

    # ---- E5: step-by-step risk series on held-out questions, then lead time.
    series = stepwise_risk_series(df, target, split, cfg.detector.stepwise_representation,
                                  cfg.seed, cfg.detector.min_train_traces_per_step, **tfidf)
    risks = series["risks"]
    atomic_write_text(tables / "stepwise_risk_series.json",
                      json.dumps({"representation": cfg.detector.stepwise_representation,
                                  "target": target, "fitted_up_to_k": series["fitted_up_to_k"],
                                  "risks": risks}))
    online = online_auroc_by_step(df, risks, target, cfg.online.max_step, cfg.online.min_traces,
                                  cfg.metrics.bootstrap_resamples, cfg.seed)
    atomic_write_dataframe(online, tables / "table_online_auroc_by_step.csv")
    figures.plot_f15(online, figs)
    thresholds = list(cfg.intervention.thresholds)
    lead_df = lead_times(df, risks, thresholds)
    lead_summary = lead_time_summary(lead_df)
    false_alarms = false_alarm_rates(df, risks, thresholds)
    lead_summary = lead_summary.merge(false_alarms, on="threshold", how="left")
    atomic_write_dataframe(lead_df, tables / "table_lead_time.csv")
    atomic_write_dataframe(lead_summary, tables / "table_lead_summary.csv")
    histogram_threshold = 0.5 if 0.5 in thresholds else thresholds[len(thresholds) // 2]
    figures.plot_f9(lead_df, lead_summary, false_alarms, histogram_threshold, figs)

    # ---- E6: calibration of every probabilistic detector at the full prefix (test split).
    scores = pd.read_csv(tables / "full_prefix_test_scores.csv")
    reliability, eces, ece_rows = {}, {}, []
    for detector in PROBABILISTIC_DETECTORS:
        if detector not in scores.columns:
            continue
        ece = expected_calibration_error(scores["y_true"], scores[detector], cfg.metrics.ece_bins)
        reliability[detector] = reliability_table(scores["y_true"], scores[detector], cfg.metrics.ece_bins)
        eces[detector] = ece["ece"]
        ece_rows.append({"detector": detector, "ece": ece["ece"], "n_test": int(len(scores))})
        atomic_write_dataframe(reliability[detector], tables / f"table_reliability_{detector}.csv")
    atomic_write_dataframe(pd.DataFrame(ece_rows), tables / "table_calibration_ece.csv")
    atomic_write_dataframe(threshold_table(scores["y_true"], scores["lexical"], thresholds),
                           tables / "table_thresholds_lexical.csv")
    figures.plot_f10(reliability, eces, figs)

    # ---- E7: train on ProcessBench (maths), evaluate on BIG-Bench Mistake; plus leave-one-tier-out.
    transfer = transfer_table(df, transfer_df, target, split, ["lexical", "surface"], cfg.seed,
                              cfg.metrics.bootstrap_resamples, "ProcessBench", **tfidf)
    atomic_write_dataframe(transfer, tables / "table_transfer.csv")
    atomic_write_text(tables / "table_transfer.tex", transfer.to_latex(index=False, float_format="%.3f"))
    loto = pd.concat([leave_one_tier_out(df, target, r, cfg.seed, cfg.metrics.bootstrap_resamples, **tfidf)
                      for r in ("lexical", "surface")], ignore_index=True)
    atomic_write_dataframe(loto, tables / "table_transfer_leave_one_tier_out.csv")
    status = pd.DataFrame([{"domain": "StrategyQA", "status": "NOT_RUN",
                            "reason": "StrategyQA provides question decompositions, not model reasoning "
                                      "traces with graded answers; a detector needs generated traces "
                                      "(GPU generation, stage 06-style) to be evaluated on it."}])
    atomic_write_dataframe(status, tables / "table_transfer_status.csv")
    figures.plot_f13(transfer, figs)

    at_half = lead_summary[lead_summary["threshold"] == histogram_threshold].iloc[0]
    pooled = transfer[transfer["evaluation_domain"] == "BBM: all tasks"].set_index("representation")
    summary = [
        f"Risk series: {cfg.detector.stepwise_representation} detectors fitted for k=1..{series['fitted_up_to_k']}, "
        f"{len(risks)} test traces",
        f"Lead time @ tau={histogram_threshold}: alarm rate {at_half['alarm_rate']:.3f}, "
        f"P(Delta>0) {at_half['frac_warned_before_error']:.3f}, median Delta {at_half['median_delta']:.1f}, "
        f"false alarms on error-free traces {at_half['false_alarm_rate_error_free']:.3f}",
    ]
    summary += [f"Online AUROC after k={int(r.steps_observed)} steps (n>k, {r.n_traces} traces): "
                f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" for r in online.itertuples()]
    summary += [f"ECE ({cfg.metrics.ece_bins} bins) {name:<12}: {value:.4f}" for name, value in eces.items()]
    for representation in pooled.index:
        row = pooled.loc[representation]
        summary.append(f"Transfer {representation:<8}: in-domain {row['in_domain_auroc']:.3f} -> "
                       f"BBM all {row['auroc']:.3f} (drop {row['auroc_drop']:+.3f})")
    summary.append("StrategyQA transfer: NOT_RUN (no generated traces; see table_transfer_status.csv)")
    for line in summary:
        log.info(line)
    manifest.log("08_lead_time_analysis", args.config, "OK",
                 {"ece": eces, "n_risk_series": len(risks)})
    print_summary_block("08_lead_time_analysis", summary)


if __name__ == "__main__":
    main()
