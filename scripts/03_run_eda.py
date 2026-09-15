"""Stage 03: EDA (E1) -> Table I (+ BIG-Bench Mistake supplement) and figures F1-F5. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.onset import describe_generators, describe_onset, paper_table_latex
from src.config import load_config
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures

PAPER_TIER_LABELS = {"gsm8k": "GSM8K", "math": "MATH", "olympiadbench": "OlympiadBench",
                     "omnimath": "OmniMATH", "ALL": "All"}


def main() -> None:
    parser = argparse.ArgumentParser(description="03: EDA -> Table I, F1-F5")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("03_run_eda", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables, figs = Path(cfg.outputs.tables_dir), cfg.outputs.figures_dir
    target = cfg.detector.target

    df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl")
    transfer = read_traces_frame(Path(cfg.data.processed_dir) / "processed_transfer.jsonl")

    table1 = describe_onset(df, target, cfg.metrics.bootstrap_resamples, cfg.seed)
    atomic_write_dataframe(table1.reset_index(), tables / "table1_descriptive.csv")
    atomic_write_text(tables / "table1_descriptive.tex",
                      table1.reset_index().to_latex(index=False, float_format="%.3f"))
    atomic_write_text(tables / "table1_paper_rows.tex", paper_table_latex(table1, PAPER_TIER_LABELS))
    atomic_write_dataframe(describe_generators(df, target), tables / "table1c_generators.csv")
    table1b = describe_onset(transfer, target, cfg.metrics.bootstrap_resamples, cfg.seed)
    atomic_write_dataframe(table1b.reset_index(), tables / "table1b_descriptive_transfer.csv")

    # KS values (from stage 04) annotate F4 when available; F4 is re-drawn by 04 otherwise.
    stats_path = tables / "table_stats_tests.json"
    ks = json.loads(stats_path.read_text())["ks_uniform"] if stats_path.exists() else None
    written = []
    written += figures.plot_f1(df, target, figs, extra_df=transfer)
    written += figures.plot_f2(df, target, figs)
    written += figures.plot_f3(df, figs)
    written += figures.plot_f4(df, figs, ks["statistic_D"] if ks else None, ks["p_value"] if ks else None)
    written += figures.plot_f5(df, figs)

    pooled = table1.loc["ALL"]
    summary = [
        f"Traces analysed (ProcessBench)   : {int(pooled['n_traces'])}",
        f"Traces with a process error      : {pooled['pct_erroneous']:.1f}%",
        f"Traces with incorrect final answer: {pooled['pct_answer_incorrect']:.1f}%",
        f"Median steps per trace           : {pooled['median_steps']:.0f} (max {int(pooled['max_steps'])})",
        f"Mean first-error index e         : {pooled['mean_err_idx']:.2f}",
        f"Mean relative onset rho          : {pooled['mean_relative_onset']:.3f} "
        f"[{pooled['relative_onset_ci_lo']:.3f}, {pooled['relative_onset_ci_hi']:.3f}]",
        f"  (expected if e ~ Uniform{{1..n}}: {pooled['expected_mean_rho_uniform_step']:.3f})",
        f"First errors in first half       : {pooled['pct_errors_first_half']:.1f}%",
        f"Mean steps after first error     : {pooled['mean_steps_remaining']:.2f}",
        f"Transfer corpus traces (BBM)     : {int(table1b.loc['ALL', 'n_traces'])}",
        f"Figures written                  : {len(written)} files (F1-F5, PNG+PDF)",
    ]
    for line in summary:
        log.info(line)
    manifest.log("03_run_eda", args.config, "OK",
                 {"n": int(pooled["n_traces"]), "figures": len(written),
                  "synthetic": cfg.data.synthetic})
    print_summary_block("03_run_eda", summary)


if __name__ == "__main__":
    main()
