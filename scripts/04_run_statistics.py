"""Stage 04: statistical tests (E2) with effect sizes; re-draws F4 annotated with KS. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistics import (
    kruskal_across_tiers,
    ks_discrete_uniform_null,
    ks_uniform,
    mann_whitney_length,
)
from src.config import load_config
from src.utils.io import Manifest, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures


def tests_for(df, target: str, n_simulations: int, seed: int) -> dict:
    """Every E2 test on one corpus."""
    erroneous = df[df["has_error"]]
    groups = [g["relative_onset"].dropna().to_numpy() for _, g in erroneous.groupby("subset")]
    return {
        "ks_uniform": ks_uniform(erroneous["relative_onset"].to_numpy()),
        "ks_discrete_uniform_step_null": ks_discrete_uniform_null(
            erroneous["first_error_idx"].to_numpy(), erroneous["n_steps"].to_numpy(),
            n_simulations, seed),
        "mann_whitney_length_process_error": mann_whitney_length(
            df.loc[df["has_error"], "n_steps"].to_numpy(),
            df.loc[~df["has_error"], "n_steps"].to_numpy()),
        f"mann_whitney_length_{target}": mann_whitney_length(
            df.loc[df[target].astype(bool), "n_steps"].to_numpy(),
            df.loc[~df[target].astype(bool), "n_steps"].to_numpy()),
        "kruskal_tiers": kruskal_across_tiers(groups) if len(groups) > 1 else
        {"H": float("nan"), "p_value": float("nan"), "eta_squared": float("nan"),
         "note": "single tier; test not applicable"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="04: statistical tests E2")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("04_run_statistics", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    target = cfg.detector.target

    df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl")
    transfer = read_traces_frame(Path(cfg.data.processed_dir) / "processed_transfer.jsonl")
    n_sim = cfg.metrics.bootstrap_resamples
    results = tests_for(df, target, n_sim, cfg.seed)
    results_transfer = tests_for(transfer, target, n_sim, cfg.seed)
    tables = Path(cfg.outputs.tables_dir)
    atomic_write_text(tables / "table_stats_tests.json", json.dumps(results, indent=2))
    atomic_write_text(tables / "table_stats_tests_transfer.json", json.dumps(results_transfer, indent=2))
    figures.plot_f4(df, cfg.outputs.figures_dir, results["ks_uniform"]["statistic_D"],
                    results["ks_uniform"]["p_value"])

    ks, ksd = results["ks_uniform"], results["ks_discrete_uniform_step_null"]
    mw, kw = results["mann_whitney_length_process_error"], results["kruskal_tiers"]
    mw_y = results[f"mann_whitney_length_{target}"]
    summary = [
        f"KS rho vs Uniform(0,1)          : D={ks['statistic_D']:.4f}, p={ks['p_value']:.3e} (n={ks['n']})",
        f"Discrete null e~Uniform{{1..n}} (MC): D_vs_null={ksd['observed_D_vs_null']:.4f} "
        f"(null 95%={ksd['null_D_vs_null_q95']:.4f}), p={ksd['p_value']:.4f}; "
        f"mean rho {ksd['observed_mean_rho']:.3f} vs null {ksd['null_mean_rho']:.3f} "
        f"({ksd['direction']}, p={ksd['p_value_mean']:.4f})",
        f"  KS-to-Uniform(0,1) expected under that null: mean D={ksd['null_D_uniform_mean']:.4f}, "
        f"95%={ksd['null_D_uniform_q95']:.4f} (observed {ksd['observed_D_uniform']:.4f})",
        "  -> " + ("rho departs from the uniform-over-steps null."
                   if ksd["p_value"] < 0.05 else
                   "rho is NOT distinguishable from errors spread evenly over steps: report honestly."),
        f"Mann-Whitney length (process err): p={mw['p_value']:.3e}, r={mw['rank_biserial_r']:+.3f} "
        f"(median {mw['median_len_erroneous']:.0f} vs {mw['median_len_correct']:.0f})",
        f"Mann-Whitney length ({target}): p={mw_y['p_value']:.3e}, r={mw_y['rank_biserial_r']:+.3f}",
        f"Kruskal-Wallis rho across tiers : H={kw['H']:.3f}, p={kw['p_value']:.3e}, "
        f"eta2={kw.get('eta_squared', float('nan')):.4f}",
        f"[BBM] KS rho vs Uniform(0,1)    : D={results_transfer['ks_uniform']['statistic_D']:.4f}, "
        f"p={results_transfer['ks_uniform']['p_value']:.3e}",
    ]
    for line in summary:
        log.info(line)
    manifest.log("04_run_statistics", args.config, "OK",
                 {"ks_D": ks["statistic_D"], "ks_p": ks["p_value"], "ks_discrete_p": ksd["p_value"]})
    print_summary_block("04_run_statistics", summary)


if __name__ == "__main__":
    main()
