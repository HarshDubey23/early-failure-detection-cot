"""Stage 07: linear probes per layer x depth bin (E4) -> F8, F14. Skips when stage 06 did not run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.models.probe import layer_depth_auroc
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures

EMPTY_COLUMNS = ["layer", "depth_bin", "n_train", "n_test", "auroc", "ci_lo", "ci_hi", "status"]


def main() -> None:
    parser = argparse.ArgumentParser(description="07: train probes E4")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("07_train_probes", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables, figs = Path(cfg.outputs.tables_dir), cfg.outputs.figures_dir
    processed = Path(cfg.data.processed_dir)

    meta_path = processed / "activation_metadata.csv"
    layers = [layer for layer in cfg.probes.layer_sweep
              if (processed / f"hidden_states_L{layer}.npy").exists()]
    if not meta_path.exists() or not layers:
        reason = "no activation tensors: GPU stage 06 did not run on this machine"
        log.info("SKIPPED: %s", reason)
        atomic_write_dataframe(pd.DataFrame(columns=EMPTY_COLUMNS), tables / "table3_probe_heatmap.csv")
        atomic_write_text(tables / "table3_probe_heatmap.tex", f"% NOT_RUN: {reason}\n")
        figures.plot_f8(pd.DataFrame(columns=EMPTY_COLUMNS), figs, reason="GPU stage 06 did not run")
        figures.plot_f14(None, None, None, figs, reason="GPU stage 06 did not run")
        manifest.log("07_train_probes", args.config, "SKIPPED_GPU_ABSENT", {"reason": reason})
        print_summary_block("07_train_probes", [f"SKIPPED: {reason}. F8/F14 written as NOT RUN panels."])
        return

    meta = pd.read_csv(meta_path)
    states_by_layer = {layer: np.load(processed / f"hidden_states_L{layer}.npy") for layer in layers}
    heat = layer_depth_auroc(meta, states_by_layer, cfg.probes.depth_bins, cfg.seed,
                             cfg.split.test_size, cfg.metrics.bootstrap_resamples,
                             cfg.probes.logistic_C, cfg.probes.logistic_max_iter)
    atomic_write_dataframe(heat, tables / "table3_probe_heatmap.csv")
    atomic_write_text(tables / "table3_probe_heatmap.tex", heat.to_latex(index=False, float_format="%.3f"))
    figures.plot_f8(heat, figs)
    layer = cfg.probes.default_layer if cfg.probes.default_layer in layers else layers[0]
    figures.plot_f14(states_by_layer[layer], 1 - meta["final_correct"].astype(int).to_numpy(),
                     meta["rel_pos"].to_numpy(), figs)

    ok = heat[heat["status"] == "OK"]
    summary = [f"Probe cells: {len(heat)} ({len(ok)} trained, {len(heat) - len(ok)} too few states)"]
    if not ok.empty:
        best = ok.loc[ok["auroc"].idxmax()]
        summary.append(f"Best cell: layer {best['layer']} depth {best['depth_bin']} "
                       f"AUROC {best['auroc']:.3f} [{best['ci_lo']:.3f}, {best['ci_hi']:.3f}]")
    summary.append(f"F14 PCA drawn from layer {layer}.")
    manifest.log("07_train_probes", args.config, "OK", {"cells": len(heat), "trained": len(ok)})
    print_summary_block("07_train_probes", summary)


if __name__ == "__main__":
    main()
