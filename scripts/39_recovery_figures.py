"""Stage 39 (CPU): F24, hallucination rate vs end-to-end cost for every policy (Run R).

Reads results/recovery/final/summary.csv (stage 34). One panel per benchmark; budgeted policies at
the primary budget (b = 2) plus the fixed ones; 95% cluster-bootstrap intervals; primary evaluator
(LLM judge, E1). CoVe is shown on the items where it was run (192 RAGTruth items).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz import style  # noqa: E402

SHOW = {"base": ("base model", "black"), "cove_all": ("Chain-of-Verification", "purple"),
        "repairdel": ("claim repair + delete", "orange"), "restart_fb": ("restart", "neutral"),
        "backtrack_fb": ("backtrack", "sky"), "care@restart": ("controller, restart only", "error"),
        "care": ("controller (CARE)", "accent"), "care@oracle": ("oracle boundary (reference)", "correct")}


def main() -> None:
    import matplotlib.pyplot as plt

    style.setup()
    S = pd.read_csv("results/recovery/final/summary.csv")
    S = S[(S.scope == "all") & (S.budget.isin([0, 2]))]
    fig, axes = plt.subplots(1, 2, figsize=style.DOUBLE)
    for ax, bench, title in zip(axes, ["ragtruth", "hotpot"], ["RAGTruth (450 sources)", "HotpotQA, controlled evidence (300)"]):
        sub = S[S.benchmark == bench].set_index("policy")
        for pol, (name, col) in SHOW.items():
            if pol not in sub.index:
                continue
            r = sub.loc[pol]
            ax.errorbar(r.cost_total, r.hall_E1, yerr=[[r.hall_E1 - r.hall_E1_lo], [r.hall_E1_hi - r.hall_E1]], fmt="o",
                        ms=4, color=style.PALETTE[col], label=name + (f" (n={int(r.n)})" if r.n != sub.loc["base"].n else ""))
        ax.set_xlabel("tokens per item (generator-equivalent)")
        ax.set_title(title)
    axes[0].set_ylabel("hallucinated outputs (LLM judge)")
    axes[0].legend(fontsize=6, loc="lower right")
    style.save(fig, "fig24_recovery_tradeoff", "results/figures")


if __name__ == "__main__":
    main()
