"""Stage 13b: figures for the counterfactual commitment experiment (reads stage-13 tables).

F18  event study: success probability v aligned at the annotated first error (r = k - e),
     for fatal and recovered traces, with clean traces aligned at a random step as a placebo
F19  where failing traces commit, relative to the annotated first error (category shares)
F20  where to backtrack: success of resampling from before the first error vs before the
     counterfactual decisive step vs restarting (sample-split)

Usage:
  python scripts/13b_commitment_figures.py --tables results/commitment/tables_A results/commitment/tables_B \
      --labels "Qwen2.5-1.5B" "Qwen2.5-Math-7B" --out results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import style  # noqa: E402

GROUP_COLOUR = {"fatal": style.PALETTE["error"], "recovered": style.PALETTE["orange"], "clean": style.PALETTE["correct"]}
CATS = ["capability-limited", "latent-commitment", "commits-at-error", "recoverable-error"]  # never-commits: 0 in all runs


def fig18(tables: list, labels: list, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(min(7.16, 3.5 * len(tables)), 2.6), sharey=True, squeeze=False)
    for ax, t, label in zip(axes[0], tables, labels):
        ev = pd.read_csv(t / "event_study.csv")
        for grp in ["clean", "recovered", "fatal"]:
            sub = ev[ev.group == grp].sort_values("r")
            if sub.empty:
                continue
            name = "clean (random anchor)" if grp == "clean" else grp
            ax.plot(sub.r, sub.mean_v, marker="o", ms=3, color=GROUP_COLOUR[grp], label=name)
            ax.fill_between(sub.r, sub.ci_lo, sub.ci_hi, color=GROUP_COLOUR[grp], alpha=0.15, lw=0)
        ax.axvline(-0.5, color=style.PALETTE["neutral"], ls="--", lw=0.8)
        ax.set_title(label)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("P(correct | first k steps)")
    axes[0][len(tables) // 2].set_xlabel("steps relative to the annotated first error (k − e)")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    style.save(fig, "fig18_event_study", out)
    plt.close(fig)


def fig19(tables: list, labels: list, out: Path, delta: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SINGLE)
    width = 0.8 / len(tables)
    x = np.arange(len(CATS))
    colours = [style.PALETTE["accent"], style.PALETTE["purple"], style.PALETTE["sky"], style.PALETTE["orange"]]
    for i, (t, label) in enumerate(zip(tables, labels)):
        c = pd.read_csv(t / "commitment_categories.csv")
        c = c[np.isclose(c.delta, delta)].set_index("category").reindex(CATS)
        ax.bar(x + (i - (len(tables) - 1) / 2) * width, c.share, width,
               yerr=[c.share - c.ci_lo, c.ci_hi - c.share], color=colours[i % 4], label=label, capsize=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-", "\n", 1) for s in CATS], fontsize=6)
    ax.set_ylabel(f"share of fatal traces (δ = {delta:g})")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6)
    style.save(fig, "fig19_commitment_categories", out)
    plt.close(fig)


def fig20(tables: list, labels: list, out: Path) -> None:
    import matplotlib.pyplot as plt

    keys = [("bt_restart", "restart"), ("bt_before_error", "before\nfirst error"),
            ("bt_before_decisive", "before\ndecisive step"), ("bt_best_prefix_oracle", "best prefix\n(oracle)")]
    fig, ax = plt.subplots(figsize=style.SINGLE)
    width = 0.8 / len(tables)
    x = np.arange(len(keys))
    colours = [style.PALETTE["accent"], style.PALETTE["purple"], style.PALETTE["sky"], style.PALETTE["orange"]]
    for i, (t, label) in enumerate(zip(tables, labels)):
        s = json.loads((t / "summary.json").read_text())
        bt = s.get("backtrack", {})
        vals = np.array([bt.get(k, [np.nan] * 3) for k, _ in keys], dtype=float)
        ax.bar(x + (i - (len(tables) - 1) / 2) * width, vals[:, 0], width,
               yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], color=colours[i % 4], label=label, capsize=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k[1] for k in keys], fontsize=6)
    ax.set_ylabel("success after resampling (fatal traces)")
    ax.set_ylim(0, 0.92)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6)
    style.save(fig, "fig20_backtracking", out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="13b: commitment figures")
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--delta", type=float, default=0.125)
    args = ap.parse_args()
    style.setup()
    tables = [Path(t) for t in args.tables]
    out = Path(args.out)
    fig18(tables, args.labels, out)
    fig19(tables, args.labels, out, args.delta)
    fig20(tables, args.labels, out)
    print("wrote fig18_event_study, fig19_commitment_categories, fig20_backtracking")


if __name__ == "__main__":
    main()
