"""Stage 11b: figures for the decomposition (reads the decomp_* tables written by stage 11).

F16  fatality of first errors by difficulty tier and by relative onset
F17  does the prefix add anything beyond question + generator? (AUROC with CIs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.viz import style


def fig16(tables: Path, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    a = pd.read_csv(tables / "decomp_A_fatality.csv")
    overall = a[a.stratum == "all"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=style.DOUBLE, sharey=True)
    for ax, stratum, title, order in [
        (axes[0], "tier", "Difficulty tier", ["gsm8k", "math", "olympiadbench", "omnimath"]),
        (axes[1], "relative_onset", "Relative onset of the first error",
         ["(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "last step"]),
    ]:
        sub = a[a.stratum == stratum].set_index("level").loc[order]
        x = np.arange(len(sub))
        ax.errorbar(x, sub.fatality, yerr=[sub.fatality - sub.ci_lo, sub.ci_hi - sub.fatality],
                    fmt="o", color=style.PALETTE["error"], capsize=2, ms=4)
        for xi, (f, n) in enumerate(zip(sub.fatality, sub.n_errors)):
            ax.annotate(f"n={n}", (xi, 0.52), ha="center", fontsize=6, color=style.PALETTE["neutral"])
        ax.axhline(overall.fatality, color=style.PALETTE["neutral"], ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=15)
        ax.set_xlabel(title)
        ax.set_ylim(0.5, 1.02)
        ax.set_xlim(-0.5, len(sub) - 0.5)
    axes[0].set_ylabel("P(final answer wrong | first error)")
    axes[0].annotate(f"dashed: all first errors, {overall.fatality:.3f}", (0.98, 0.97), xycoords="axes fraction",
                     ha="right", va="top", fontsize=6, color=style.PALETTE["neutral"])
    style.save(fig, "fig16_error_fatality", outdir)
    plt.close(fig)


def fig17(tables: Path, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    b = pd.read_csv(tables / "decomp_B_controls_auroc.csv").set_index("detector")
    fig, ax = plt.subplots(figsize=style.SINGLE)
    fracs = [25, 50, 100]
    x = np.arange(len(fracs))
    qg = b.loc["question+generator"]
    ax.axhspan(qg.ci_lo, qg.ci_hi, color=style.PALETTE["neutral"], alpha=0.15, lw=0)
    ax.axhline(qg.auroc_primary, color=style.PALETTE["neutral"], ls="--", lw=0.9,
               label="question + generator (no reasoning)")
    for offset, name, colour, label in [
        (-0.12, "prefix {}%", style.PALETTE["sky"], "prefix only"),
        (0.12, "Q+G+prefix {}%", style.PALETTE["accent"], "question + generator + prefix"),
    ]:
        rows = b.loc[[name.format(f) for f in fracs]]
        ax.errorbar(x + offset, rows.auroc_primary,
                    yerr=[rows.auroc_primary - rows.ci_lo, rows.ci_hi - rows.auroc_primary],
                    fmt="o", color=colour, capsize=2, ms=4, label=label)
    ax.axhline(0.5, color=style.PALETTE["black"], lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in fracs])
    ax.set_xlabel("Observed prefix of the chain")
    ax.set_ylabel("AUROC, answer incorrect (out-of-fold)")
    ax.set_ylim(0.48, 0.66)
    ax.legend(loc="lower right", frameon=False)
    style.save(fig, "fig17_difficulty_control", outdir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="11b: decomposition figures")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    style.setup()
    tables, figs = Path(cfg.outputs.tables_dir), Path(cfg.outputs.figures_dir)
    fig16(tables, figs)
    fig17(tables, figs)
    print("wrote fig16_error_fatality, fig17_difficulty_control")


if __name__ == "__main__":
    main()
