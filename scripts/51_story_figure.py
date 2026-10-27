"""Stage 51 (CPU): the one-figure summary of the whole study (Fig. 0) and the ChokeBench figure (Fig. 25).

Reads only committed result tables:
  results/tables/decomp_B_controls_auroc.csv                (Part 1)
  results/commitment/tables_*/commitment_categories.csv      (Part 2)
  results/chokebench/summary.csv                             (stage 49)
  results/recovery/final/evaluator_robustness.json           (stage 50)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a8a7a2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlelocation": "left"})
MODELS = [("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "_allbatches"), ("Qwen2-1.5B-Instruct", "Qwen2-1.5B", "_allbatches"),
          ("Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-7B", ""), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B", "")]
LOC_NAMES = {"human_first_error": "Human first error", "prm_largest_drop": "PRM: largest drop",
             "prm_first_below_0.5": "PRM: first step < 0.5", "gen_max_token_nll": "Generator: max NLL",
             "uniform_random": "Random step", "oracle_decisive_split": "Oracle choke point"}


def panel_difficulty(ax) -> None:
    b = pd.read_csv(ROOT / "results/tables/decomp_B_controls_auroc.csv").set_index("detector")
    rows = [("question-only", "Question"), ("question+generator", "Question + model"), ("prefix 25%", "25% of reasoning"),
            ("Q+G+prefix 25%", "Q + model + 25%"), ("Q+G+prefix 100%", "Q + model + 100%")]
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab) in zip(y, rows):
        r = b.loc[k]
        col = BLUE if k == "question+generator" else (ORANGE if "prefix" in k else GREY)
        ax.barh(yi, r.auroc_primary - 0.5, left=0.5, height=0.6, color=col)
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=INK, lw=1)
        ax.text(max(r.ci_hi, 0.6) + 0.004, yi, f"{r.auroc_primary:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(b.loc["question+generator", "auroc_primary"], color=BLUE, lw=0.8, ls="--")
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlim(0.5, 0.67)
    ax.set_xlabel("AUROC for a wrong final answer")
    ax.set_title("1. Early warning mostly reads difficulty")


def panel_categories(ax) -> None:
    cats = [("capability-limited", "Decided before step 1", GREY), ("latent-commitment", "Before the labelled error", YELLOW),
            ("commits-at-error", "At the labelled error", BLUE), ("recoverable-error", "Still open after it", AQUA)]
    y = np.arange(len(MODELS))[::-1]
    for yi, (gen, lab, suf) in zip(y, MODELS):
        c = pd.read_csv(ROOT / f"results/commitment/tables_{gen}{suf}/commitment_categories.csv")
        c = c[c.delta == 0.125].set_index("category")
        left = 0.0
        for k, name, col in cats:
            s = float(c.loc[k, "share"])
            ax.barh(yi, s, left=left, height=0.6, color=col, edgecolor="white", linewidth=1.5,
                    label=name if yi == y[0] else None)
            if s > 0.09:
                ax.text(left + s / 2, yi, f"{s:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if col in (BLUE, GREY) else INK)
            left += s
    ax.set_yticks(y, [lab for _, lab, _ in MODELS])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of failed traces (resampled, δ = 1/8)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=7.5)
    ax.set_title("2. Where failures are decided")


def panel_localise(ax, S: pd.DataFrame) -> None:
    locs = ["human_first_error", "prm_largest_drop", "prm_first_below_0.5", "gen_max_token_nll", "uniform_random"]
    P = S[S.generator == "pooled"].set_index("localiser")
    y = np.arange(len(locs))[::-1]
    h = 0.36
    for yi, k in zip(y, locs):
        r = P.loc[k]
        ax.barh(yi + h / 2, r.acc_human, height=h, color=GREY, label="finds the human-labelled error" if yi == y[0] else None)
        ax.barh(yi - h / 2, r.acc_decisive, height=h, color=BLUE, label="finds the choke point" if yi == y[0] else None)
        ax.plot([r.acc_decisive_lo, r.acc_decisive_hi], [yi - h / 2] * 2, color=INK, lw=1)
        ax.text(r.acc_decisive_hi + 0.015, yi - h / 2, f"{r.acc_decisive:.0%}", va="center", fontsize=7.5)
        if k != "human_first_error":
            ax.text(r.acc_human + 0.015, yi + h / 2, f"{r.acc_human:.0%}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [LOC_NAMES[k] for k in locs])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of open failed traces (choke point: posterior)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("3. Step localisers find errors, not choke points")


def panel_gain(ax, S: pd.DataFrame) -> None:
    locs = ["oracle_decisive_split", "human_first_error", "prm_first_below_0.5", "prm_largest_drop", "gen_max_token_nll",
            "uniform_random"]
    P = S[S.generator == "pooled"].set_index("localiser")
    y = np.arange(len(locs))[::-1]
    for yi, k in zip(y, locs):
        r = P.loc[k]
        col = AQUA if r.gain_b4_free > 0 else ORANGE
        ax.barh(yi, r.gain_b4_free, height=0.6, color=col)
        ax.plot([r.gain_b4_free_lo, r.gain_b4_free_hi], [yi, yi], color=INK, lw=1)
        x = r.gain_b4_free_hi + 0.006 if r.gain_b4_free > 0 else r.gain_b4_free_lo - 0.006
        ax.text(x, yi, f"{r.gain_b4_free:+.3f}", va="center", ha="left" if r.gain_b4_free > 0 else "right", fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [LOC_NAMES[k] for k in locs])
    ax.set_xlim(-0.26, 0.09)
    ax.set_xlabel("P(solved) − restart, budget = 4 restarts, finding the step is free")
    ax.set_title("4. No deployable resampling point beats starting over")


def panel_rag(ax) -> None:
    rob = json.loads((ROOT / "results/recovery/final/evaluator_robustness.json").read_text())
    base = rob["policies_test_split"]["base@0"]
    evs = [("E1", "LLM judge (E1)", False), ("E1cal", "E1, calibrated", False), ("E2", "LettuceDetect-large", True),
           ("dual", "E1 ∧ E2 (dual)", True), ("D", "gate detector D", True)]
    y = np.arange(len(evs))[::-1]
    for yi, (k, lab, same) in zip(y, evs):
        m, lo, hi = base[k + "_minus_restart"]
        ax.barh(yi, m, height=0.6, color=ORANGE if same else BLUE)
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1)
        ax.text(hi + 0.01, yi, f"{m:+.3f}", va="center", fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [lab for _, lab, _ in evs])
    ax.set_xlim(-0.08, 0.56)
    ax.set_xlabel("hallucination reduction: base − restart-until-D-passes")
    ax.text(0.47, y[2] + 0.5, "share LettuceDetect's\ntraining data with D", fontsize=7.5, color=ORANGE, va="bottom", ha="center")
    ax.text(0.16, (y[0] + y[1]) / 2, "independent of D", fontsize=7.5, color=BLUE, va="center")
    ax.set_title("5. RAG: gains are visible only to the gate's relatives")


def main() -> None:
    S = pd.read_csv(ROOT / "results/chokebench/summary.csv")
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=1.6, left=0.09, right=0.98, top=0.9, bottom=0.12)
    panel_difficulty(fig.add_subplot(gs[0, 0:2]))
    panel_categories(fig.add_subplot(gs[0, 2:4]))
    panel_localise(fig.add_subplot(gs[0, 4:6]), S)
    panel_gain(fig.add_subplot(gs[1, 0:3]), S)
    panel_rag(fig.add_subplot(gs[1, 3:6]))
    fig.suptitle("Errors are not choke points, and restart is hard to beat", x=0.09, ha="left",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig0_story.{ext}", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.75, "left": 0.15, "right": 0.97,
                                                                     "bottom": 0.2, "top": 0.88})
    panel_localise(axs[0], S)
    panel_gain(axs[1], S)
    axs[0].set_title("Localisation accuracy")
    axs[1].set_title("Value of the named step at a matched budget")
    axs[1].set_xlabel("P(solved) − restart, 4× budget, step found free")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig25_chokebench.{ext}", dpi=200)
    print("wrote fig0_story and fig25_chokebench")


if __name__ == "__main__":
    main()
