"""Stage 07: probes per layer x depth bin (E4) -> F8, F14."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistics import auroc_with_ci
from src.config import load_config
from src.models.probe import fit_probe, predict_risk
from src.utils.io import Manifest, atomic_write_dataframe
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import grouped_split, seed_everything
from src.viz import figures


def main() -> None:
    parser = argparse.ArgumentParser(description="07: train probes E4")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("07_train_probes", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    meta_path = Path(cfg.data.processed_dir) / "activation_metadata.csv"
    layers_available = [l for l in cfg.probes.layer_sweep
                        if (Path(cfg.data.processed_dir) / f"hidden_states_L{l}.npy").exists()]
    if not meta_path.exists() or not layers_available:
        msg = ("SKIPPED: no activation tensors found (GPU stage did not run). "
               "F8/F14 will read NOT_RUN; everything else is unaffected.")
        log.info(msg)
        empty = pd.DataFrame(columns=["layer", "depth_bin", "auroc", "ci_lo", "ci_hi"])
        atomic_write_dataframe(
            empty, Path(cfg.outputs.tables_dir) / "table3_probe_heatmap.csv")
        figures.plot_f8(empty, cfg.outputs.figures_dir)
        manifest.log("07_train_probes", args.config, "SKIPPED_GPU_ABSENT",
                     {"reason": "no activation tensors"})
        print_summary_block("07_train_probes", [msg])
        return

    meta = pd.read_csv(meta_path)
    # Labels: activation metadata has no gold outcomes in smoke; fall back to
    # the processed outcome column when question_ids overlap, else NOT_RUN.
    rows = []
    for layer in layers_available:
        H = np.load(Path(cfg.data.processed_dir) / f"hidden_states_L{layer}.npy")
        for lo, hi in cfg.probes.depth_bins:
            mask = (meta["rel_pos"] > lo) & (meta["rel_pos"] <= hi)
            idx = np.where(mask.to_numpy())[0]
            # Without outcome labels on generated traces, probe training needs
            # graded answers; absence is recorded, not worked around.
            if "final_correct" not in meta.columns or meta.loc[mask, "final_correct"].nunique() < 2:
                log.info("layer %d bin %.0f-%.0f%%: no outcome labels, skipped",
                         layer, 100 * lo, 100 * hi)
                continue
            y = (1 - meta.loc[mask, "final_correct"].astype(int)).to_numpy()
            groups = meta.loc[mask, "question_id"].to_numpy()
            tr, te = grouped_split(groups, cfg.split.test_size, cfg.seed,
                                   context=f"probe L{layer} bin {lo}-{hi}")
            scaler, clf = fit_probe(H[idx][tr], y[tr], cfg.probes.logistic_C,
                                    cfg.probes.logistic_max_iter, cfg.seed)
            stats = auroc_with_ci(y[te], predict_risk(scaler, clf, H[idx][te]),
                                  cfg.metrics.bootstrap_resamples, cfg.seed)
            rows.append({"layer": layer, "depth_bin": f"{lo:.0%}-{hi:.0%}",
                         **stats, "n_test": len(te)})
            log.info("layer %d depth %.0f-%.0f%%: AUROC=%.4f", layer,
                     100 * lo, 100 * hi, stats["auroc"])
    heat = pd.DataFrame(rows)
    atomic_write_dataframe(heat, Path(cfg.outputs.tables_dir) / "table3_probe_heatmap.csv")
    (Path(cfg.outputs.tables_dir) / "table3_probe_heatmap.tex").write_text(
        heat.to_latex(index=False, float_format="%.4f") if not heat.empty else "% NOT_RUN\n")
    figures.plot_f8(heat if not heat.empty else
                    pd.DataFrame(columns=["layer", "depth_bin", "auroc"]),
                    cfg.outputs.figures_dir)
    # F14: PCA over the default layer when available.
    default = cfg.probes.default_layer
    if default in layers_available and not heat.empty:
        H = np.load(Path(cfg.data.processed_dir) / f"hidden_states_L{default}.npy")
        y_all = ((1 - meta["final_correct"].astype(int)).to_numpy()
                 if "final_correct" in meta.columns else np.zeros(len(meta)))
        figures.plot_f14(H, y_all, meta["rel_pos"].to_numpy(), cfg.outputs.figures_dir)
        f14 = "F14 written (PCA)."
    else:
        f14 = "F14 NOT_RUN (no activations)."
    summary = [f"Probe cells trained: {len(heat)}",
               f"Best cell AUROC: {heat['auroc'].max():.4f}" if not heat.empty else "Best: NOT_RUN",
               f14]
    manifest.log("07_train_probes", args.config, "OK", {"cells": len(heat)})
    print_summary_block("07_train_probes", summary)


if __name__ == "__main__":
    main()
