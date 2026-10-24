"""Stage 21 (CPU): figures for measurement reliability, policies and answer curves.

  F21 fig21_reliability     split-half correlation of v_k vs continuations per prefix, and the
                            posterior probability that the annotated first error is the decisive step
  F22 fig22_policies        success vs matched end-to-end budget for each policy (fatal traces)
  F23 fig23_answer_curves   answer curve a_k aligned at the annotated first error, by outcome group
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

LABEL = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen2-1.5B-Instruct": "Qwen2-1.5B",
         "Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
COLOURS = ["accent", "purple", "sky", "orange"]
POLICY_NAME = {"P0_restart": "restart", "P1_prm_first_error": "PRM first error", "P2_prm_largest_drop": "PRM largest drop",
               "P3_probe_search": "probe search", "P4_learned": "learned", "P5_human_first_error": "human first error",
               "P6_oracle": "oracle (all rollouts counted)"}


def fig21(c: Path, gens: list, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for i, g in enumerate(gens):
        d = c / f"robustness_{g}_allbatches"
        if not (d / "precision_vs_m.csv").exists():
            d = c / f"robustness_{g}"
        if not (d / "precision_vs_m.csv").exists():
            continue
        p = pd.read_csv(d / "precision_vs_m.csv")
        axes[0].plot(p.m_total, p.split_half_r, marker="o", ms=3, color=style.PALETTE[COLOURS[i % 4]], label=LABEL.get(g, g))
        post = pd.read_csv(d / "decisive_posterior.csv")
        axes[1].hist(post.p_d_eq_e, bins=np.linspace(0, 1, 11), histtype="step", lw=1.4,
                     color=style.PALETTE[COLOURS[i % 4]], label=LABEL.get(g, g))
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("continuations per prefix (m)")
    axes[0].set_ylabel("split-half correlation of $v_k$")
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel("posterior P(decisive step = annotated error)")
    axes[1].set_ylabel("fatal traces")
    style.save(fig, "fig21_reliability", out)
    plt.close(fig)


def fig22(c: Path, gens: list, out: Path) -> None:
    import matplotlib.pyplot as plt

    f = c / "policies" / "policy_summary.csv"
    if not f.exists():
        return
    S = pd.read_csv(f)
    S = S[(S.accounting == "flops") & (S.population == "all_fatal")]
    gens = [g for g in gens if g in set(S.generator)]
    fig, axes = plt.subplots(1, len(gens), figsize=(min(7.16, 1.9 * len(gens)), 2.4), sharey=True, squeeze=False)
    order = [p for p in POLICY_NAME if p in set(S.policy)]
    cmap = plt.get_cmap("tab10")
    for ax, g in zip(axes[0], gens):
        sub = S[S.generator == g]
        for j, pol in enumerate(order):
            s = sub[sub.policy == pol].sort_values("budget")
            if s.empty:
                continue
            ax.plot(s.budget, s.success, marker="o", ms=2.5, lw=1.2, color=cmap(j),
                    ls="--" if pol in ("P5_human_first_error", "P6_oracle") else "-", label=POLICY_NAME[pol])
        ax.set_xscale("log", base=2)
        ax.set_title(LABEL.get(g, g), fontsize=8)
        ax.set_xlabel("budget (restart attempts)")
    axes[0][0].set_ylabel("P(solved), fatal traces")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.12))
    style.save(fig, "fig22_policies", out)
    plt.close(fig)


def fig23(c: Path, gens: list, out: Path) -> None:
    import matplotlib.pyplot as plt

    f = c / "answer_curves" / "answer_event_study.csv"
    if not f.exists():
        return
    A = pd.read_csv(f)
    gens = [g for g in gens if g in set(A.generator)]
    fig, axes = plt.subplots(1, len(gens), figsize=(min(7.16, 1.9 * len(gens)), 2.3), sharey=True, squeeze=False)
    colour = {"fatal": style.PALETTE["error"], "recovered": style.PALETTE["orange"], "clean": style.PALETTE["correct"]}
    for ax, g in zip(axes[0], gens):
        for grp in ["clean", "recovered", "fatal"]:
            s = A[(A.generator == g) & (A.group == grp)].sort_values("r")
            ax.plot(s.r, s.a, marker="o", ms=2.5, color=colour[grp], label=grp)
        ax.axvline(-0.5, color=style.PALETTE["neutral"], ls="--", lw=0.8)
        ax.set_title(LABEL.get(g, g), fontsize=8)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("forced-answer accuracy $a_k$")
    axes[0][len(gens) // 2].set_xlabel("steps relative to the annotated first error (k − e)")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.1))
    style.save(fig, "fig23_answer_curves", out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="21: Run C figures")
    ap.add_argument("--commitment", default="results/commitment")
    ap.add_argument("--gens", nargs="+", default=list(LABEL))
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    style.setup()
    c, out = Path(args.commitment), Path(args.out)
    fig21(c, args.gens, out)
    fig22(c, args.gens, out)
    fig23(c, args.gens, out)
    print(json.dumps({"written": [p.name for p in sorted(out.glob("fig2[123]_*.png"))]}))


if __name__ == "__main__":
    main()
