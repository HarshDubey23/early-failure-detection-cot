"""Stage 03: EDA (E1) -> Table 1 + Figures F1-F5. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.onset import describe_onset
from src.config import load_config
from src.utils.io import Manifest, atomic_write_dataframe
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures


def main() -> None:
    parser = argparse.ArgumentParser(description="03: EDA -> F1-F5")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("03_run_eda", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    prefix = "synthetic_" if cfg.data.synthetic else ""
    processed = Path(cfg.data.processed_dir) / f"{prefix}processed_traces.csv"
    if not processed.exists():
        print_summary_block("03_run_eda", [f"FAILED: {processed} missing."])
        manifest.log("03_run_eda", args.config, "FAILED", {})
        sys.exit(1)
    # Steps column round-trips as string; re-parse for prefix/lexical stages.
    import ast

    df = pd.read_csv(processed)
    df["steps"] = df["steps"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )

    table1 = describe_onset(df, cfg.metrics.bootstrap_resamples, cfg.seed)
    atomic_write_dataframe(table1.reset_index(),
                           Path(cfg.outputs.tables_dir) / f"{prefix}table1_descriptive.csv")
    # LaTeX fragment for the paper (Table II).
    tex = table1.reset_index().to_latex(index=False, float_format="%.3f")
    (Path(cfg.outputs.tables_dir) / f"{prefix}table1_descriptive.tex").write_text(tex)

    written = []
    written += figures.plot_f1(df, cfg.outputs.figures_dir)
    written += figures.plot_f2(df, cfg.outputs.figures_dir)
    written += figures.plot_f3(df, cfg.outputs.figures_dir)
    written += figures.plot_f4(df, cfg.outputs.figures_dir)
    written += figures.plot_f5(df, cfg.outputs.figures_dir)

    all_row = table1.loc["ALL"] if "ALL" in table1.index else None
    summary = [
        f"Traces analysed              : {len(df)}",
        f"Erroneous traces             : {100 * df['has_error'].mean():.1f}%",
        f"Median steps per trace       : {df['n_steps'].median():.0f}",
        f"Mean first-error index       : {df['first_error_idx'].mean():.2f}",
        f"Mean relative onset          : {all_row['mean_relative_onset']:.3f}"
        if all_row is not None else "n/a",
        f"Mean steps after first error : {all_row['mean_steps_remaining']:.2f}"
        if all_row is not None else "n/a",
        f"Figures written              : {len(written)} files (F1-F5, PNG+PDF)",
    ]
    log.info("eda complete on %d traces", len(df))
    manifest.log("03_run_eda", args.config, "OK",
                 {"n": len(df), "figures": len(written),
                  "synthetic": cfg.data.synthetic})
    print_summary_block("03_run_eda", summary)


if __name__ == "__main__":
    main()
