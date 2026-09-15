"""
WHAT: One function per numbered figure (F1-F14), PNG + PDF via style.save.

WHY:  Brief §7: every figure must be regenerable by re-running one command.
      No plotting logic lives in scripts; scripts call these functions.

HOW:  Each `plot_fN(df_or_table, outdir)` builds the matplotlib figure,
      labels every axis with units, and saves both formats. Functions degrade
      to a labelled "insufficient data" panel instead of raising, so a weak or
      negative result still produces an honest figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .style import PALETTE, save, setup

setup()


def _empty_panel(title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.text(0.5, 0.5, "insufficient data (NOT_RUN)", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_f1(df: pd.DataFrame, outdir) -> list:
    """F1: class balance (counts) by dataset and tier. Justifies AUROC."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ct = df.groupby(["subset", "has_error"]).size().unstack(fill_value=0)
    ct.columns = ["Correct trace" if c is False else "Erroneous trace"
                  for c in ct.columns]
    colours = [PALETTE["correct"], PALETTE["error"]]
    ct.plot.bar(ax=ax, color=colours[: len(ct.columns)], edgecolor="black", lw=0.5)
    ax.set_xlabel("Subset")
    ax.set_ylabel("Number of traces (count)")
    ax.legend(frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return save(fig, "fig1_class_balance", outdir)


def plot_f2(df: pd.DataFrame, outdir) -> list:
    """F2: trace-length distribution overall + split by outcome."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["n_steps"], bins=40, color=PALETTE["accent"], edgecolor="white")
    ax[0].axvline(df["n_steps"].median(), color="red", ls="--",
                  label=f"median = {df['n_steps'].median():.0f} steps")
    ax[0].set_xlabel("Steps per trace (count)")
    ax[0].set_ylabel("Number of traces (count)")
    ax[0].legend(frameon=False)
    for label, grp, colour in [
        ("Correct", df[~df["has_error"]], PALETTE["correct"]),
        ("Erroneous", df[df["has_error"]], PALETTE["error"]),
    ]:
        ax[1].hist(grp["n_steps"], bins=30, alpha=0.6, label=label,
                   color=colour, density=True)
    ax[1].set_xlabel("Steps per trace (count)")
    ax[1].set_ylabel("Density (1/steps)")
    ax[1].legend(frameon=False)
    plt.tight_layout()
    return save(fig, "fig2_chain_length", outdir)


def plot_f3(df: pd.DataFrame, outdir) -> list:
    """F3: absolute first-error index histogram (raw onset picture)."""
    import matplotlib.pyplot as plt

    erroneous = df[df["has_error"] & df["first_error_idx"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if erroneous.empty:
        fig, ax = _empty_panel("Absolute index of first erroneous step",
                               "Step index (1-indexed)", "Number of traces (count)")
    else:
        ax.hist(erroneous["first_error_idx"], bins=40, color=PALETTE["error"],
                edgecolor="white")
        ax.axvline(erroneous["first_error_idx"].median(), color="black", ls="--",
                   label=f"median = {erroneous['first_error_idx'].median():.0f}")
        ax.set_xlabel("Step index of first error (1-indexed)")
        ax.set_ylabel("Number of traces (count)")
        ax.legend(frameon=False)
    plt.tight_layout()
    return save(fig, "fig3_absolute_error_index", outdir)


def plot_f4(df: pd.DataFrame, outdir) -> list:
    """F4 (HEADLINE): relative onset histogram + CDF. Proves concentration."""
    import matplotlib.pyplot as plt

    erroneous = df[df["has_error"] & df["relative_onset"].notna()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    if erroneous.empty:
        plt.close(fig)
        fig, ax0 = plt.subplots(figsize=(7, 4.2))
        ax0.text(0.5, 0.5, "insufficient data (NOT_RUN)", ha="center")
        ax0.set_xlabel("Relative onset rho (fraction of chain)")
        ax0.set_ylabel("Number of traces (count)")
        plt.tight_layout()
        return save(fig, "fig4_relative_error_position_HEADLINE", outdir)
    rho = erroneous["relative_onset"]
    ax[0].hist(rho, bins=25, color=PALETTE["error"], edgecolor="white")
    ax[0].axvline(rho.mean(), color="black", ls="--", label=f"mean = {rho.mean():.3f}")
    ax[0].axvline(rho.median(), color=PALETTE["accent"], ls=":",
                  label=f"median = {rho.median():.3f}")
    ax[0].set_xlabel("Relative onset rho (0 = start, 1 = end)")
    ax[0].set_ylabel("Number of traces (count)")
    ax[0].legend(frameon=False)
    cumulative = np.sort(rho.dropna().to_numpy())
    ax[1].plot(cumulative, np.arange(1, len(cumulative) + 1) / len(cumulative),
               color=PALETTE["accent"], lw=2.2)
    for quantile in (0.25, 0.5, 0.75):
        value = np.quantile(cumulative, quantile)
        ax[1].axvline(value, ls=":", alpha=0.6)
        ax[1].text(value, 0.04, f"{quantile:.0%}\n@{value:.2f}", fontsize=8, ha="center")
    ax[1].set_xlabel("Relative onset rho (fraction of chain)")
    ax[1].set_ylabel("Cumulative fraction of errors")
    plt.tight_layout()
    return save(fig, "fig4_relative_error_position_HEADLINE", outdir)


def plot_f5(df: pd.DataFrame, outdir) -> list:
    """F5: onset by difficulty tier (box + violin)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    erroneous = df[df["has_error"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    if erroneous.empty:
        plt.close(fig)
        fig, ax0 = plt.subplots(figsize=(7, 4.2))
        ax0.text(0.5, 0.5, "insufficient data (NOT_RUN)", ha="center")
        ax0.set_xlabel("Subset")
        ax0.set_ylabel("Relative onset rho")
        plt.tight_layout()
        return save(fig, "fig5_difficulty_breakdown", outdir)
    order = sorted(df["subset"].unique())
    sns.boxplot(data=erroneous, x="subset", y="relative_onset", order=order,
                ax=ax[0], palette="Blues", hue="subset", legend=False,
                fliersize=1.5)
    ax[0].set_xlabel("Subset")
    ax[0].set_ylabel("Relative onset rho (fraction)")
    sns.violinplot(data=erroneous, x="subset", y="first_error_idx", order=order,
                   ax=ax[1], palette="Oranges", hue="subset", legend=False, cut=0)
    ax[1].set_xlabel("Subset")
    ax[1].set_ylabel("First-error step index (1-indexed)")
    plt.tight_layout()
    return save(fig, "fig5_difficulty_breakdown", outdir)


def plot_f6(y_true, scores_dict: dict, outdir) -> list:
    """F6: ROC curves, all representations overlaid (detector comparison)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    colours = [PALETTE["accent"], PALETTE["error"], PALETTE["purple"], PALETTE["correct"]]
    for i, (name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, lw=2.2, color=colours[i % len(colours)],
                label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    return save(fig, "fig6_roc_all_representations", outdir)


def plot_f7(curve_df: pd.DataFrame, outdir) -> list:
    """F7 (CORE RESULT): AUROC vs observed prefix fraction, overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"lexical": "s", "surface": "o", "uncertainty": "D", "hidden-state": "^"}
    colours = {"lexical": PALETTE["accent"], "surface": PALETTE["correct"],
               "uncertainty": PALETTE["purple"], "hidden-state": PALETTE["error"]}
    for name, group in curve_df.groupby("representation"):
        group = group.sort_values("fraction_observed")
        ax.plot(100 * group["fraction_observed"], group["auroc"],
                marker=markers.get(name, "o"), ls="-", lw=2.2, ms=7,
                color=colours.get(name, PALETTE["neutral"]), label=name)
        if "ci_lo" in group and "ci_hi" in group:
            ax.fill_between(100 * group["fraction_observed"], group["ci_lo"],
                            group["ci_hi"], alpha=0.15,
                            color=colours.get(name, PALETTE["neutral"]))
    ax.axhline(0.5, color="k", ls="--", lw=1, label="Random (0.50)")
    ax.axhline(0.7, color="gray", ls=":", lw=1, label="Useful (0.70)")
    ax.set_xlabel("Observed prefix of chain (%)")
    ax.set_ylabel("AUROC for eventual failure")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False)
    plt.tight_layout()
    return save(fig, "fig7_auroc_vs_prefix_KEY", outdir)


def plot_f8(heatmap_df: pd.DataFrame, outdir) -> list:
    """F8: layer x depth AUROC heatmap (where the signal lives)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no activation data (NOT_RUN)", ha="center")
        ax.set_xlabel("Depth bin")
        ax.set_ylabel("Layer")
    else:
        pivot = heatmap_df.pivot(index="layer", columns="depth_bin", values="auroc")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0.5, vmax=1.0, ax=ax)
        ax.set_xlabel("Depth bin (fraction of chain)")
        ax.set_ylabel("Model layer")
    plt.tight_layout()
    return save(fig, "fig8_layer_depth_heatmap", outdir)


def plot_f9(lead_df: pd.DataFrame, outdir) -> list:
    """F9: detection lead time Delta distribution + threshold tradeoff."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    deltas = lead_df["delta"].dropna() if not lead_df.empty else pd.Series([], dtype=float)
    if deltas.empty:
        ax[0].text(0.5, 0.5, "no lead-time data (NOT_RUN)", ha="center")
    else:
        ax[0].hist(deltas, bins=30, color=PALETTE["accent"], edgecolor="white")
        ax[0].axvline(0, color="red", ls="--", label="error moment (Delta=0)")
        ax[0].legend(frameon=False)
    ax[0].set_xlabel("Lead time Delta = e - k* (steps)")
    ax[0].set_ylabel("Number of traces (count)")
    if not lead_df.empty:
        summary = lead_df.groupby("threshold")["warned_before_error"].mean()
        ax[1].plot(summary.index.astype(float), summary.values, "o-",
                   color=PALETTE["error"], lw=2.2)
    ax[1].set_xlabel("Alarm threshold tau")
    ax[1].set_ylabel("Fraction warned before error P(Delta>0)")
    plt.tight_layout()
    return save(fig, "fig9_lead_time", outdir)


def plot_f10(rel_df: pd.DataFrame, ece: float, outdir) -> list:
    """F10: reliability diagram + ECE."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if rel_df.empty:
        ax.text(0.5, 0.5, "no calibration data (NOT_RUN)", ha="center")
    else:
        midpoints = (rel_df["lo"] + rel_df["hi"]) / 2
        ax.plot(midpoints, rel_df["acc"], "o-", color=PALETTE["accent"],
                label="Detector", lw=2.2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, title=f"ECE = {ece:.3f}")
    ax.set_xlabel("Mean predicted risk (confidence)")
    ax.set_ylabel("Empirical failure rate (accuracy)")
    plt.tight_layout()
    return save(fig, "fig10_reliability", outdir)


def plot_f11(policy_df: pd.DataFrame, outdir) -> list:
    """F11: recovery vs corruption scatter per policy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if policy_df.empty:
        ax.text(0.5, 0.5, "no intervention data (NOT_RUN)", ha="center")
    else:
        for _, row in policy_df.iterrows():
            ax.scatter(row["corruption_rate"], row["recovery_rate"],
                       s=90, label=str(row["threshold"]))
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Break-even")
        ax.legend(frameon=False, title="Policy")
    ax.set_xlabel("Corruption rate P(broken | would succeed)")
    ax.set_ylabel("Recovery rate P(fixed | would fail)")
    plt.tight_layout()
    return save(fig, "fig11_recovery_vs_corruption", outdir)


def plot_f12(policy_df: pd.DataFrame, outdir) -> list:
    """F12: accuracy vs added-token cost frontier."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    if policy_df.empty:
        ax.text(0.5, 0.5, "no intervention data (NOT_RUN)", ha="center")
    else:
        ax.scatter(policy_df["extra_tokens_per_question"],
                   policy_df["net_benefit"], s=90, color=PALETTE["accent"])
        for _, row in policy_df.iterrows():
            ax.annotate(str(row["threshold"]),
                        (row["extra_tokens_per_question"], row["net_benefit"]))
    ax.set_xlabel("Extra tokens per question (count)")
    ax.set_ylabel("Net benefit (recovery - corruption)")
    plt.tight_layout()
    return save(fig, "fig12_cost_frontier", outdir)


def plot_f13(transfer_df: pd.DataFrame, outdir) -> list:
    """F13: cross-domain transfer bar chart (AUROC drop math -> other)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.4))
    if transfer_df.empty:
        ax.text(0.5, 0.5, "no transfer data (NOT_RUN)", ha="center")
        ax.set_ylabel("AUROC")
    else:
        ax.bar(transfer_df["evaluation_domain"], transfer_df["auroc"],
               color=PALETTE["accent"], edgecolor="black", lw=0.5)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Evaluation domain")
        ax.set_ylabel("AUROC (train on math)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return save(fig, "fig13_transfer", outdir)


def plot_f14(states_2d: np.ndarray, labels: np.ndarray, depths: np.ndarray,
             outdir) -> list:
    """F14: PCA of activations at 25/50/75% depth (qualitative separation)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, centre in zip(axes, (0.25, 0.50, 0.75)):
        mask = np.abs(depths - centre) < 0.13
        ax.set_title(f"Depth ~{centre:.0%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if mask.sum() < 10:
            ax.text(0.5, 0.5, "insufficient data", ha="center")
            continue
        pcs = PCA(n_components=2, random_state=42).fit_transform(states_2d[mask])
        for value, colour, name in [(0, PALETTE["correct"], "Succeed"),
                                    (1, PALETTE["error"], "Fail")]:
            sel = labels[mask] == value
            ax.scatter(pcs[sel, 0], pcs[sel, 1], s=8, alpha=0.6,
                       color=colour, label=name)
        ax.legend(frameon=False, markerscale=3)
    plt.tight_layout()
    return save(fig, "fig14_activation_pca", outdir)
