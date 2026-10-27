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
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "xtick.labelsize": 9, "ytick.labelsize": 9.5, "axes.labelsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlepad": 10,
                     "legend.fontsize": 8.5, "legend.frameon": False})
MODELS = [("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "_allbatches"), ("Qwen2-1.5B-Instruct", "Qwen2-1.5B", "_allbatches"),
          ("Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-7B", ""), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B", "")]
LOC_NAMES = {"human_first_error": "Human first error", "prm_largest_drop": "PRM: largest drop",
             "prm_first_below_0.5": "PRM: first score < 0.5", "gen_max_token_nll": "Model uncertainty (NLL)",
             "uniform_random": "Random step", "oracle_decisive_split": "Oracle choke point",
             "learned_localiser": "Learned, tuned"}


def _value(ax, x, y, text, right=True, color=INK) -> None:
    """Value label placed just past the end of an error bar, so text never sits on a line."""
    ax.annotate(text, (x, y), xytext=(4 if right else -4, 0), textcoords="offset points",
                ha="left" if right else "right", va="center", fontsize=8.5, color=color)


def _bar_legend(ax, items, ncol=2) -> None:
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=name) for name, c in items], loc="upper left",
              bbox_to_anchor=(0.0, -0.2), ncol=ncol, handlelength=1.2, columnspacing=1.2)


def panel_difficulty(ax) -> None:
    b = pd.read_csv(ROOT / "results/tables/decomp_B_controls_auroc.csv").set_index("detector")
    rows = [("question-only", "Question text"), ("question+generator", "Question + model"),
            ("prefix 25%", "First 25% of steps"), ("Q+G+prefix 25%", "Q + model + 25%"),
            ("Q+G+prefix 100%", "Q + model + all steps")]
    y = np.arange(len(rows))[::-1]
    ref = b.loc["question+generator", "auroc_primary"]
    ax.axvline(ref, color=BLUE, lw=0.9, ls=(0, (3, 3)), zorder=0)
    for yi, (k, lab) in zip(y, rows):
        r = b.loc[k]
        col = ORANGE if "prefix" in k else (BLUE if k == "question+generator" else GREY)
        ax.barh(yi, r.auroc_primary - 0.5, left=0.5, height=0.62, color=col)
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=INK, lw=1.1)
        _value(ax, max(r.ci_hi, ref + 0.002), yi, f"{r.auroc_primary:.3f}")  # keep labels clear of the reference line
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlim(0.5, 0.68)
    ax.set_xticks([0.50, 0.55, 0.60, 0.65])
    ax.set_xlabel("AUROC for predicting a wrong answer (0.5 = chance)")
    ax.set_title("1. Early warning mostly reads difficulty")
    _bar_legend(ax, [("does not read the reasoning", GREY), ("baseline: question + model", BLUE),
                     ("reads the reasoning", ORANGE)], ncol=2)


def panel_categories(ax) -> None:
    cats = [("capability-limited", "decided before step 1", GREY), ("latent-commitment", "before the labelled error", YELLOW),
            ("commits-at-error", "at the labelled error", BLUE), ("recoverable-error", "still open after it", AQUA)]
    y = np.arange(len(MODELS))[::-1]
    for yi, (gen, lab, suf) in zip(y, MODELS):
        c = pd.read_csv(ROOT / f"results/commitment/tables_{gen}{suf}/commitment_categories.csv")
        c = c[c.delta == 0.125].set_index("category")
        left = 0.0
        for k, _, col in cats:
            share = float(c.loc[k, "share"])
            ax.barh(yi, share, left=left, height=0.62, color=col, edgecolor="white", linewidth=2)
            if share >= 0.06:
                ax.text(left + share / 2, yi, f"{share:.0%}", ha="center", va="center", fontsize=8.5 if share >= 0.1 else 7,
                        color="white" if col in (BLUE, GREY) else INK, fontweight="bold")
            left += share
    ax.set_yticks(y, [lab for _, lab, _ in MODELS])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of failed solutions, measured by resampling")
    ax.set_title("2. Where failures are decided")
    _bar_legend(ax, [(n, col) for _, n, col in cats], ncol=2)


def panel_localise(ax, S: pd.DataFrame) -> None:
    locs = ["human_first_error", "prm_largest_drop", "prm_first_below_0.5", "learned_localiser", "gen_max_token_nll",
            "uniform_random"]
    P = S[S.generator == "pooled"].set_index("localiser")
    y = np.arange(len(locs))[::-1] * 1.25
    h = 0.42
    for yi, k in zip(y, locs):
        r = P.loc[k]
        ax.barh(yi + h / 2, r.acc_human, height=h, color=GREY)
        ax.barh(yi - h / 2, r.acc_decisive, height=h, color=BLUE)
        ax.plot([r.acc_decisive_lo, r.acc_decisive_hi], [yi - h / 2] * 2, color=INK, lw=1.1)
        _value(ax, r.acc_human, yi + h / 2, f"{r.acc_human:.0%}", color=MUTED)
        _value(ax, r.acc_decisive_hi, yi - h / 2, f"{r.acc_decisive:.0%}")
    ax.set_yticks(y, [LOC_NAMES[k] for k in locs])
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of open failures where the named step is ...")
    ax.set_title("3. Verifiers find errors, not choke points")
    _bar_legend(ax, [("... the human-labelled error", GREY), ("... the choke point", BLUE)], ncol=2)


def panel_gain(ax, S: pd.DataFrame) -> None:
    locs = ["oracle_decisive_split", "human_first_error", "prm_first_below_0.5", "learned_localiser", "prm_largest_drop",
            "gen_max_token_nll", "uniform_random"]
    P = S[S.generator == "pooled"].set_index("localiser")
    y = np.arange(len(locs))[::-1]
    for yi, k in zip(y, locs):
        r = P.loc[k]
        pos = r.gain_b4_free > 0
        ax.barh(yi, r.gain_b4_free, height=0.62, color=AQUA if pos else ORANGE)
        ax.plot([r.gain_b4_free_lo, r.gain_b4_free_hi], [yi, yi], color=INK, lw=1.1)
        _value(ax, r.gain_b4_free_hi if pos else r.gain_b4_free_lo, yi, f"{r.gain_b4_free:+.3f}", right=pos)
    ax.axvline(0, color=INK, lw=0.9)
    ax.text(0.003, y[-1] - 0.75, "restart", fontsize=8.5, color=MUTED, ha="left", va="center")
    ax.set_yticks(y, [LOC_NAMES[k] for k in locs])
    ax.set_ylim(y[-1] - 1.0, y[0] + 0.6)
    ax.set_xlim(-0.27, 0.1)
    ax.set_xlabel("change in P(solved) vs restart, same budget (4 restarts), step found for free")
    ax.set_title("4. No practical restart point beats starting over")
    _bar_legend(ax, [("worse than restart", ORANGE), ("better (ceiling, not deployable)", AQUA)], ncol=2)


def panel_rag(ax) -> None:
    rob = json.loads((ROOT / "results/recovery/final/evaluator_robustness.json").read_text())
    base = rob["policies_test_split"]["base@0"]
    evs = [("E1", "LLM judge (E1)", False), ("E1cal", "LLM judge, calibrated", False),
           ("E2", "LettuceDetect-large (E2)", True), ("dual", "E1 and E2 together", True), ("D", "Gate detector D itself", True)]
    y = np.arange(len(evs))[::-1]
    for yi, (k, lab, same) in zip(y, evs):
        m, lo, hi = base[k + "_minus_restart"]
        ax.barh(yi, m, height=0.62, color=ORANGE if same else BLUE)
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.1)
        _value(ax, hi, yi, f"{100 * m:+.1f} pts")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y, [lab for _, lab, _ in evs])
    ax.set_xlim(-0.08, 0.6)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", "10", "20", "30", "40", "50"])
    ax.set_xlabel("fewer hallucinated answers after restart-until-D-passes (percentage points)")
    ax.set_title("5. RAG: the measured gain depends on who grades it")
    _bar_legend(ax, [("grader independent of D", BLUE), ("grader shares D's training data", ORANGE)], ncol=2)


def fig_tuning() -> None:
    """Fig. 26: nested hyperparameter selection of the learned localiser (inner-CV score per configuration)."""
    rep = json.loads((ROOT / "results/chokebench/learned_tuning.json").read_text())
    grid = rep["grid"]
    lab = [(f"{g['features']}: logreg C={g['C']}" if g["model"] == "logreg"
            else f"{g['features']}: boost d={g['max_depth']} lr={g['learning_rate']}") for g in grid]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(grid))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for c, (held, r) in zip(cols, rep["outer"].items()):
        sel = grid.index(r["selected"])
        ax.plot(x, r["inner_scores"], "-o", color=c, lw=1.5, ms=4,
                label=f"held out {held.replace('-Instruct', '')}: test {r['test_acc_decisive']:.2f}")
        ax.plot(sel, r["inner_scores"][sel], "*", color=c, ms=14, mec=INK, mew=0.6)
    ax.set_xticks(x, lab, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("inner-CV score\n(P(named step = choke point))")
    ax.set_title("Learned localiser: nested hyperparameter search (★ = selected configuration)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig26_learned_tuning.{ext}", dpi=200)
    plt.close(fig)


def fig_mitigation() -> None:
    """Fig. 27: what mitigation buys. Left: failed maths solutions recovered by restart vs budget.
    Right: RAG hallucination vs cost for each policy, under the D-family grader E2 and the
    independent judge E1, with the detect-restart-abstain law's human-scale prediction."""
    m = json.loads((ROOT / "results/recovery/final/mitigation_law.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    for (gen, d), c in zip(sorted(m["maths_restart_recovery"].items()), [BLUE, ORANGE, AQUA, YELLOW]):
        xs = [1, 2, 4, 8]
        ys = [d[str(x)][0] for x in xs]
        a.errorbar(xs, ys, yerr=[[ys[i] - d[str(x)][1] for i, x in enumerate(xs)], [d[str(x)][2] - ys[i] for i, x in enumerate(xs)]],
                   fmt="-o", color=c, lw=2, ms=6, capsize=0, elinewidth=1, label=gen.replace("-Instruct", ""))
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    a.set_ylim(0, 0.85)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
    a.set_xlabel("budget (restart attempts, FLOP-matched)")
    a.set_ylabel("failed solutions recovered")
    a.set_title("Maths: restarting recovers most failures")
    a.legend(loc="lower right")
    order = ["base", "delete", "restart_fb", "backtrack_fb", "repairdel", "care", "abstain_flag", "cove_all"]
    label = {"base": "No mitigation", "delete": "Delete flagged sentences", "restart_fb": "Detect, restart, abstain (DRA)",
             "backtrack_fb": "Detect, backtrack, abstain", "repairdel": "Repair flagged claims", "care": "CARE controller",
             "abstain_flag": "Abstain on flagged QA", "cove_all": "CoVe, raw output (n=192)"}
    cost = {c["policy"]: c for c in m["cost"]}
    law = m["law"]
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        c = cost[k]
        b.plot([c["hall_E2"], c["hall_E1"]], [yi, yi], color=GREY, lw=1.5, zorder=1)
        b.scatter(c["hall_E1"], yi, color=BLUE, marker="s", s=48, zorder=3)
        b.scatter(c["hall_E2"], yi, color=ORANGE, marker="o", s=48, zorder=3)
        if k in ("base", "restart_fb"):
            v, lo, hi = law["h"] if k == "base" else law["pi"]
            b.errorbar([v], [yi + 0.28], xerr=[[v - lo], [hi - v]], fmt="D", color=AQUA, ms=7, capsize=2, zorder=4)
    b.set_yticks(y, [f"{label[k]}  ({cost[k]['cost_vs_base']:.2f}x)" for k in order])
    b.set_xlim(0, 1.02)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    b.set_xlabel("answers with a hallucination   (cost relative to no mitigation in brackets)")
    b.set_title("RAG: hallucination after each mitigation")
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], color=BLUE, marker="s", ls="", label="independent LLM judge (E1)"),
                      Line2D([], [], color=ORANGE, marker="o", ls="", label="grader trained like the detector (E2)"),
                      Line2D([], [], color=AQUA, marker="D", ls="", label="people (no mitigation) / law's prediction (DRA)")],
             loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig27_mitigation.{ext}", dpi=200)
    plt.close(fig)


def fig_law() -> None:
    """Fig. 28: two checks of Proposition 3. Left: from budget 1 to 4, coverage rises and delivered
    hallucination barely moves. Right: the predicted operating curve (hallucination of delivered
    answers vs draws per answered input) for three gates, from human-labelled drafts."""
    t = json.loads((ROOT / "results/recovery/final/law_tests.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.0), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})
    r = t["budget_invariance"]["restart_fb"]
    items = [("coverage", r["coverage_diff"], AQUA), ("delivered hallucination\n(independent judge E1)", r["hall_E1_delivered_diff"], BLUE),
             ("delivered hallucination\n(LettuceDetect-large E2)", r["hall_E2_delivered_diff"], ORANGE)]
    y = np.arange(len(items))[::-1]
    for yi, (lab, d, c) in zip(y, items):
        a.barh(yi, 100 * d[0], height=0.55, color=c)
        a.plot([100 * d[1], 100 * d[2]], [yi, yi], color=INK, lw=1.1)
        _value(a, 100 * d[2], yi, f"{100 * d[0]:+.1f} pts")
    a.axvline(0, color=INK, lw=0.9)
    a.set_yticks(y, [lab for lab, _, _ in items])
    a.set_xlim(-1, 4.5)
    a.set_xlabel("change from budget 1 to budget 4 (percentage points, 95% CI)")
    a.set_title("More retries buy coverage, not faithfulness")
    cols = {"D (LettuceDetect-base)": BLUE, "E2 (LettuceDetect-large)": ORANGE, "D or E2 flags": AQUA}
    for g, c in cols.items():
        pts = [x for x in t["operating_curve"] if x["gate"] == g]
        xs = [x["draws_per_answer"] for x in pts]
        b.plot(xs, [100 * x["pi"] for x in pts], "-o", color=c, lw=2, ms=5, label=f"gate: {g}")
        b.fill_between(xs, [100 * x["pi_lo"] for x in pts], [100 * x["pi_hi"] for x in pts], color=c, alpha=0.12, lw=0)
    run = next(x for x in t["operating_curve"] if x["gate"].startswith("D (") and abs(x["tau"] - 0.5) < 1e-9)
    b.scatter([run["draws_per_answer"]], [100 * run["pi"]], s=160, facecolor="none", edgecolor=INK, lw=1.5, zorder=5)
    b.annotate("the setting used in our runs", (run["draws_per_answer"], 100 * run["pi"]), xytext=(18, 22),
               textcoords="offset points", fontsize=8.5, arrowprops={"arrowstyle": "-", "color": MUTED})
    b.axhline(100 * t["human_base_rate"], color=GREY, ls=(0, (4, 3)), lw=1.2)
    b.text(4.45, 100 * t["human_base_rate"] - 2.5, "no mitigation", ha="right", fontsize=8.5, color=MUTED)
    b.set_ylim(0, 55)
    b.set_xlim(1.2, 4.6)
    b.set_xlabel("generations per answered input (cost)")
    b.set_ylabel("hallucinated delivered answers (%)")
    b.set_title("Pick a hallucination rate, pay its price (predicted, human labels)")
    b.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig28_law.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    S = pd.read_csv(ROOT / "results/chokebench/summary.csv")
    fig = plt.figure(figsize=(17, 10.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.3, wspace=0.08, hspace=0.12)
    gs = fig.add_gridspec(2, 6)
    panel_difficulty(fig.add_subplot(gs[0, 0:2]))
    panel_categories(fig.add_subplot(gs[0, 2:4]))
    panel_localise(fig.add_subplot(gs[0, 4:6]), S)
    panel_gain(fig.add_subplot(gs[1, 0:3]), S)
    panel_rag(fig.add_subplot(gs[1, 3:6]))
    fig.suptitle("Errors are not choke points, and restart is hard to beat", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig0_story.{ext}", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.2, wspace=0.08)
    panel_localise(axs[0], S)
    panel_gain(axs[1], S)
    axs[0].set_title("Localisation accuracy")
    axs[1].set_title("Value of the named step at a matched budget")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig25_chokebench.{ext}", dpi=200)
    plt.close(fig)
    fig_tuning()
    fig_mitigation()
    fig_law()
    print("wrote fig28_law, fig27_mitigation, fig0_story, fig25_chokebench, fig26_learned_tuning")


if __name__ == "__main__":
    main()
