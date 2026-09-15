"""Stage 04: statistical tests (E2). Thin CLI wrapper - no logic here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistics import (
    kruskal_across_tiers,
    ks_uniform,
    mann_whitney_length,
)
from src.config import load_config
from src.utils.io import Manifest, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="04: statistical tests E2")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("04_run_statistics", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    prefix = "synthetic_" if cfg.data.synthetic else ""
    processed = Path(cfg.data.processed_dir) / f"{prefix}processed_traces.csv"
    df = pd.read_csv(processed)
    erroneous = df[df["has_error"]]

    ks = ks_uniform(erroneous["relative_onset"].dropna().to_numpy())
    mw = mann_whitney_length(
        df[df["has_error"]]["n_steps"].to_numpy(),
        df[~df["has_error"]]["n_steps"].to_numpy(),
    )
    groups = [g["relative_onset"].dropna().to_numpy()
              for _, g in erroneous.groupby("subset")]
    kw = kruskal_across_tiers(groups) if len(groups) > 1 else {
        "H": float("nan"), "p_value": float("nan"), "eta_squared": float("nan"),
        "note": "single tier; test not applicable",
    }
    results = {"ks_uniform": ks, "mann_whitney_length": mw, "kruskal_tiers": kw}
    atomic_write_text(Path(cfg.outputs.tables_dir) / f"{prefix}table_stats_tests.json",
                      json.dumps(results, indent=2))
    summary = [
        f"KS vs Uniform(0,1)      : D={ks['statistic_D']:.4f}, p={ks['p_value']:.3e}",
        "  -> " + ("NOT uniform: errors concentrate (detection window supported)."
                   if ks["p_value"] < 0.05 else
                   "Cannot reject uniformity: report honestly; window weaker than hypothesised."),
        f"Mann-Whitney on length  : p={mw['p_value']:.3e}, rank-biserial r={mw['rank_biserial_r']:+.3f}",
        f"Kruskal-Wallis tiers    : H={kw['H']:.3f}, p={kw['p_value']:.3e}, eta2={kw.get('eta_squared', float('nan')):.3f}",
    ]
    log.info("statistics: %s", results)
    manifest.log("04_run_statistics", args.config, "OK",
                 {"ks_D": ks["statistic_D"], "ks_p": ks["p_value"]})
    print_summary_block("04_run_statistics", summary)


if __name__ == "__main__":
    main()
