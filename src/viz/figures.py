"""
WHAT: One function per numbered figure (F1-F14), PNG + PDF via style.save.

WHY:  Brief §6-7: every figure must be regenerable by re-running one command,
      so no plotting logic lives in scripts and no image is edited by hand.

HOW:  Each `plot_fN(...)` takes already-computed tables (never raw models),
      labels every axis with units, and saves both formats. When a stage did
      not run (e.g. no GPU for F8/F14) the function writes an explicit
      "NOT RUN" panel stating why, instead of raising or drawing fake data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .style import DOUBLE, PALETTE, SINGLE, SINGLE_TALL, save, setup

setup()

REPRESENTATION_STYLE = {
    # name: (colour, marker, linestyle)
    "lexical": (PALETTE["accent"], "s", "-"),
    "surface": (PALETTE["correct"], "o", "-"),
    "length-only": (PALETTE["neutral"], "^", "--"),
    "surface-MLP": (PALETTE["purple"], "D", ":"),
    "surface-GRU": (PALETTE["orange"], "v", ":"),
    "uncertainty": (PALETTE["sky"], "P", "-"),
    "hidden-state": (PALETTE["error"], "*", "-"),
}


def _not_run_figure(name: str, outdir, reason: str, xlabel: str, ylabel: str,
                    size=SINGLE) -> list:
    """Explicit NOT RUN panel: the figure exists, and says plainly why it is empty."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, f"NOT RUN\n{reason}", ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color=PALETTE["neutral"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return save(fig, name, outdir)


def plot_f1(df: pd.DataFrame, target: str, outdir, extra_df: pd.DataFrame | None = None) -> list:
    """F1: class balance of the detector target by dataset and tier (justifies AUROC)."""
    import matplotlib.pyplot as plt

    panels = [("ProcessBench tier", df)]
    if extra_df is not None and not extra_df.empty:
        panels.append(("BIG-Bench Mistake task", extra_df))
    fig, axes = plt.subplots(1, len(panels), figsize=DOUBLE if len(panels) > 1 else SINGLE,
                             gridspec_kw={"width_ratios": [4, 5]} if len(panels) > 1 else None)
    axes = np.atleast_1d(axes)
    for ax, (label, frame) in zip(axes, panels):
        counts = frame.groupby(["subset", target]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=[False, True], fill_value=0)
        x = np.arange(len(counts))
        ax.bar(x - 0.2, counts[False], 0.4, color=PALETTE["correct"], label="Final answer correct")
        ax.bar(x + 0.2, counts[True], 0.4, color=PALETTE["error"], label="Final answer incorrect")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in counts.index], fontsize=6.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Traces (count)")
        ax.set_ylim(0, 1.3 * counts.to_numpy().max())  # headroom so the legend never covers a bar
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    return save(fig, "fig1_class_balance", outdir)


def plot_f2(df: pd.DataFrame, target: str, outdir) -> list:
    """F2: trace length overall and split by outcome (tests the length confound)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=DOUBLE)
    bins = np.arange(1.5, df["n_steps"].max() + 1.5, 1)
    ax[0].hist(df["n_steps"], bins=bins, color=PALETTE["accent"], edgecolor="white", lw=0.3)
    ax[0].axvline(df["n_steps"].median(), color=PALETTE["black"], ls="--", lw=1,
                  label=f"median = {df['n_steps'].median():.0f} steps")
    ax[0].set_xlabel("Steps per trace, n (count)")
    ax[0].set_ylabel("Traces (count)")
    ax[0].legend(frameon=False)
    for label, value, colour in [("Final answer correct", False, PALETTE["correct"]),
                                 ("Final answer incorrect", True, PALETTE["error"])]:
        group = df[df[target] == value]
        ax[1].hist(group["n_steps"], bins=bins, alpha=0.55, density=True, color=colour,
                   label=f"{label} (median {group['n_steps'].median():.0f})")
    ax[1].set_xlabel("Steps per trace, n (count)")
    ax[1].set_ylabel("Fraction of traces per step")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    return save(fig, "fig2_chain_length", outdir)


def plot_f3(df: pd.DataFrame, outdir) -> list:
    """F3: absolute first-error index e (raw onset picture)."""
    import matplotlib.pyplot as plt

    erroneous = df[df["has_error"]]
    if erroneous.empty:
        return _not_run_figure("fig3_absolute_error_index", outdir, "no annotated errors",
                               "First-error step e", "Traces (count)")
    fig, ax = plt.subplots(figsize=SINGLE)
    bins = np.arange(0.5, erroneous["first_error_idx"].max() + 1.5, 1)
    ax.hist(erroneous["first_error_idx"], bins=bins, color=PALETTE["error"], edgecolor="white", lw=0.3)
    ax.axvline(erroneous["first_error_idx"].median(), color=PALETTE["black"], ls="--", lw=1,
               label=f"median e = {erroneous['first_error_idx'].median():.0f}")
    ax.set_xlabel("First erroneous step, e (1-indexed)")
    ax.set_ylabel("Traces (count)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, "fig3_absolute_error_index", outdir)


def plot_f4(df: pd.DataFrame, outdir, ks_d: float | None = None, ks_p: float | None = None) -> list:
    """F4 (HEADLINE): relative onset rho - histogram and CDF, each against two references.

    References: Uniform(0,1) (the brief's KS null) and the discrete null in which
    the error step is uniform over each trace's own n steps. The second is the
    fair comparison, because rho = e/n can only take the values 1/n, ..., 1.
    """
    import matplotlib.pyplot as plt

    from src.analysis.statistics import _null_cdf_uniform_steps

    erroneous = df[df["has_error"]]
    rho = erroneous["relative_onset"].dropna().to_numpy()
    if len(rho) == 0:
        return _not_run_figure("fig4_relative_error_position_HEADLINE", outdir,
                               "no annotated errors", "Relative onset rho", "Traces (count)", DOUBLE)
    n_steps = erroneous["n_steps"].to_numpy(dtype=float)
    edges = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(1, 2, figsize=DOUBLE)
    ax[0].hist(rho, bins=edges, color=PALETTE["error"], edgecolor="white", lw=0.3, label="observed")
    # Expected count per bin under the discrete null (same bins, so lattice effects cancel).
    null_at_edges = _null_cdf_uniform_steps(edges - 1e-9, n_steps)
    null_at_edges[-1] = 1.0
    expected = np.diff(null_at_edges) * len(rho)
    ax[0].stairs(expected, edges, color=PALETTE["black"], lw=1.1, label="expected if e ~ Uniform{1..n}")
    ax[0].axvline(rho.mean(), color=PALETTE["accent"], ls="--", lw=1, label=f"observed mean = {rho.mean():.3f}")
    ax[0].set_xlabel(r"Relative onset $\rho = e/n$ (0 = start, 1 = end of chain)")
    ax[0].set_ylabel("Erroneous traces (count)")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.45 * max(expected.max(), np.histogram(rho, edges)[0].max()))
    ax[0].legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2)

    sorted_rho = np.sort(rho)
    grid = np.linspace(0, 1, 1001)
    ax[1].step(sorted_rho, np.arange(1, len(sorted_rho) + 1) / len(sorted_rho), where="post",
               color=PALETTE["error"], lw=1.6, label=r"observed $\rho$")
    ax[1].plot(grid, _null_cdf_uniform_steps(grid, n_steps), color=PALETTE["black"], lw=1.1,
               label="null: e ~ Uniform{1..n}")
    ax[1].plot([0, 1], [0, 1], color=PALETTE["neutral"], ls="--", lw=1, label="Uniform(0,1)")
    for quantile in (0.25, 0.5, 0.75):
        value = float(np.quantile(sorted_rho, quantile))
        ax[1].plot([value, value], [0, quantile], color=PALETTE["accent"], ls=":", lw=0.8)
        ax[1].text(value - 0.01, quantile + 0.02, f"{quantile:.0%} of errors by {value:.2f}",
                   fontsize=6, ha="right")
    title = "" if ks_d is None else f"KS vs Uniform(0,1): D = {ks_d:.3f}"
    ax[1].legend(frameon=False, loc="lower right", title=title, title_fontsize=6.5, fontsize=6.5)
    ax[1].set_xlabel(r"Relative onset $\rho$ (fraction of chain)")
    ax[1].set_ylabel("Cumulative fraction of errors")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.02)
    fig.tight_layout()
    return save(fig, "fig4_relative_error_position_HEADLINE", outdir)


def plot_f5(df: pd.DataFrame, outdir) -> list:
    """F5: onset by difficulty tier - box plot of rho and violin of e."""
    import matplotlib.pyplot as plt

    erroneous = df[df["has_error"]]
    if erroneous.empty:
        return _not_run_figure("fig5_difficulty_breakdown", outdir, "no annotated errors",
                               "Tier", "Relative onset rho", DOUBLE)
    order = sorted(erroneous["subset"].unique())
    rho_groups = [erroneous.loc[erroneous["subset"] == s, "relative_onset"].dropna() for s in order]
    e_groups = [erroneous.loc[erroneous["subset"] == s, "first_error_idx"].dropna() for s in order]
    fig, ax = plt.subplots(1, 2, figsize=DOUBLE)
    ax[0].boxplot(rho_groups, tick_labels=order, widths=0.5, showfliers=True,
                  flierprops={"markersize": 1.5},
                  medianprops={"color": PALETTE["error"]})
    ax[0].set_xlabel("ProcessBench tier")
    ax[0].set_ylabel(r"Relative onset $\rho$ (fraction)")
    parts = ax[1].violinplot(e_groups, showmedians=True, widths=0.7)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["orange"])
        body.set_alpha(0.6)
    ax[1].set_xticks(np.arange(1, len(order) + 1))
    ax[1].set_xticklabels(order)
    ax[1].set_xlabel("ProcessBench tier")
    ax[1].set_ylabel("First erroneous step, e (1-indexed)")
    fig.tight_layout()
    return save(fig, "fig5_difficulty_breakdown", outdir)


def plot_f6(y_true: np.ndarray, scores_by_representation: dict, outdir) -> list:
    """F6: ROC curves at the full prefix, all representations overlaid."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    for name, scores in scores_by_representation.items():
        colour, _, linestyle = REPRESENTATION_STYLE.get(name, (PALETTE["neutral"], "o", "-"))
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color=colour, ls=linestyle,
                label=f"{name} (AUROC {roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["black"], ls="--", lw=0.8, label="chance / majority (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save(fig, "fig6_roc_all_representations", outdir)


def plot_f7(curve_df: pd.DataFrame, outdir, name: str = "fig7_auroc_vs_prefix_KEY",
            representations: tuple = ("lexical", "surface", "length-only")) -> list:
    """F7 (CORE RESULT): AUROC vs observed prefix fraction with bootstrap 95% CIs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=SINGLE)
    for representation in representations:
        group = curve_df[curve_df["representation"] == representation].sort_values("fraction_observed")
        if group.empty:
            continue
        colour, marker, linestyle = REPRESENTATION_STYLE[representation]
        x = 100 * group["fraction_observed"]
        ax.fill_between(x, group["ci_lo"], group["ci_hi"], color=colour, alpha=0.15, lw=0)
        ax.plot(x, group["auroc"], color=colour, marker=marker, ms=3.5, ls=linestyle,
                label=representation)
    ax.axhline(0.5, color=PALETTE["black"], ls="--", lw=0.8, label="chance (0.50)")
    ax.set_xlabel("Observed prefix of the chain, k/n (%)")
    ax.set_ylabel("AUROC (held-out questions)")
    ax.set_xlim(0, 102)
    low = min(0.45, float(curve_df["ci_lo"].min()) - 0.02)
    high = max(0.75, float(curve_df["ci_hi"].max()) + 0.02)
    ax.set_ylim(low, min(1.0, high))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    return save(fig, name, outdir)


def plot_f8(heatmap_df: pd.DataFrame, outdir, reason: str = "") -> list:
    """F8: layer x depth-bin AUROC heatmap (where in the network the signal lives)."""
    import matplotlib.pyplot as plt

    usable = heatmap_df[heatmap_df.get("status", pd.Series(dtype=str)) == "OK"] \
        if not heatmap_df.empty else heatmap_df
    if usable.empty:
        return _not_run_figure("fig8_layer_depth_heatmap", outdir,
                               reason or "no activation tensors (GPU stage 06 did not run)",
                               "Depth bin (fraction of chain)", "Model layer")
    pivot = heatmap_df.pivot(index="layer", columns="depth_bin", values="auroc")
    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), value in np.ndenumerate(pivot.to_numpy(dtype=float)):
        ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.2f}", ha="center", va="center",
                fontsize=6.5, color="white" if not np.isnan(value) and value < 0.8 else "black")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Depth bin (fraction of chain)")
    ax.set_ylabel("Model layer")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    return save(fig, "fig8_layer_depth_heatmap", outdir)


def plot_f9(lead_df: pd.DataFrame, summary_df: pd.DataFrame, false_alarm_df: pd.DataFrame,
            histogram_threshold: float, outdir) -> list:
    """F9: lead-time distribution at one tau + threshold-vs-lead-time tradeoff."""
    import matplotlib.pyplot as plt

    if lead_df.empty:
        return _not_run_figure("fig9_lead_time", outdir, "no annotated test traces",
                               "Lead time Delta (steps)", "Traces (count)", DOUBLE)
    fig, ax = plt.subplots(1, 2, figsize=DOUBLE)
    at_threshold = lead_df[np.isclose(lead_df["threshold"], histogram_threshold)]
    deltas = at_threshold["delta"].dropna()
    never = int((~at_threshold["alarmed"]).sum())
    bins = np.arange(deltas.min() - 0.5, deltas.max() + 1.5, 1) if len(deltas) else 10
    ax[0].hist(deltas, bins=bins, color=PALETTE["accent"], edgecolor="white", lw=0.3)
    ax[0].axvline(0, color=PALETTE["error"], ls="--", lw=1, label=r"alarm at the error ($\Delta$=0)")
    ax[0].set_xlabel(r"Lead time $\Delta = e - k^\ast$ (steps; >0 = warned early)")
    ax[0].set_ylabel("Erroneous test traces (count)")
    ax[0].legend(frameon=False, title=f"tau = {histogram_threshold:g}; never alarmed: {never}",
                 title_fontsize=7)
    ax[1].plot(summary_df["threshold"], summary_df["frac_warned_before_error"], "o-",
               color=PALETTE["correct"], ms=3.5, label=r"P($\Delta$>0), erroneous traces")
    ax[1].plot(summary_df["threshold"], summary_df["alarm_rate"], "s--", color=PALETTE["accent"],
               ms=3.5, label="alarm rate, erroneous traces")
    ax[1].plot(false_alarm_df["threshold"], false_alarm_df["false_alarm_rate_error_free"], "^:",
               color=PALETTE["error"], ms=3.5, label="alarm rate, error-free traces")
    ax[1].set_xlabel(r"Alarm threshold $\tau$")
    ax[1].set_ylabel("Fraction of test traces")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    return save(fig, "fig9_lead_time", outdir)


def plot_f10(reliability_by_detector: dict, ece_by_detector: dict, outdir) -> list:
    """F10: reliability diagram (per detector) with ECE in the legend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    ax.plot([0, 1], [0, 1], color=PALETTE["black"], ls="--", lw=0.8, label="perfect calibration")
    for name, table in reliability_by_detector.items():
        colour, marker, linestyle = REPRESENTATION_STYLE.get(name, (PALETTE["neutral"], "o", "-"))
        ax.plot(table["conf"], table["acc"], color=colour, marker=marker, ms=3.5, ls=linestyle,
                label=f"{name} (ECE {ece_by_detector[name]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk in bin")
    ax.set_ylabel("Observed fraction incorrect in bin")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, "fig10_reliability", outdir)


def plot_f11(policy_df: pd.DataFrame, outdir) -> list:
    """F11: recovery vs corruption per policy (points above the diagonal gain)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(SINGLE_TALL[0], 4.4))
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], ls=":", lw=0.8, label="recovery = corruption")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
    for i, (_, row) in enumerate(policy_df.iterrows()):
        ax.scatter(row["corruption_rate"], row["recovery_rate"], s=22, marker=markers[i % len(markers)],
                   label=row["policy"], zorder=3)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Corruption rate P(broken | would succeed)")
    ax.set_ylabel("Recovery rate P(fixed | would fail)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    return save(fig, "fig11_recovery_vs_corruption", outdir)


def plot_f12(policy_df: pd.DataFrame, outdir) -> list:
    """F12: accuracy change vs added-token cost for every method."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(SINGLE_TALL[0], 4.4))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
    for i, (_, row) in enumerate(policy_df.iterrows()):
        ax.scatter(row["extra_tokens_per_question"] + 1.0, 100 * row["net_accuracy_change"], s=22,
                   marker=markers[i % len(markers)], label=row["policy"], zorder=3)
    ax.axhline(0, color=PALETTE["black"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Extra tokens per question + 1 (log scale)")
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    return save(fig, "fig12_cost_frontier", outdir)


def plot_f13(transfer_df: pd.DataFrame, outdir) -> list:
    """F13: cross-domain transfer - AUROC per evaluation domain and representation."""
    import matplotlib.pyplot as plt

    if transfer_df.empty:
        return _not_run_figure("fig13_transfer", outdir, "no target-domain data",
                               "Evaluation domain", "AUROC", DOUBLE)
    domains = list(dict.fromkeys(transfer_df["evaluation_domain"]))
    representations = list(dict.fromkeys(transfer_df["representation"]))
    width = 0.8 / len(representations)
    fig, ax = plt.subplots(figsize=DOUBLE)
    x = np.arange(len(domains))
    for i, representation in enumerate(representations):
        group = transfer_df[transfer_df["representation"] == representation].set_index("evaluation_domain")
        group = group.reindex(domains)
        colour = REPRESENTATION_STYLE.get(representation, (PALETTE["neutral"],))[0]
        errors = np.vstack([group["auroc"] - group["ci_lo"], group["ci_hi"] - group["auroc"]])
        ax.bar(x + (i - (len(representations) - 1) / 2) * width, group["auroc"], width,
               yerr=errors, capsize=1.5, color=colour, label=representation,
               error_kw={"lw": 0.6})
    ax.axhline(0.5, color=PALETTE["black"], ls="--", lw=0.8, label="chance (0.50)")
    ax.set_xticks(x)
    labels = [d.replace("BBM: ", "BBM\n").replace(" (", "\n(").replace("_", " ") for d in domains]
    ax.set_xticklabels([label.replace("tracking shuffled objects", "tracking shuffled\nobjects")
                        for label in labels], fontsize=6)
    ax.set_ylabel("AUROC (trained on ProcessBench)")
    # Full [0, 1] range: a below-chance transfer (inverted ranking) must stay visible.
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    return save(fig, "fig13_transfer", outdir)


def plot_f14(states: np.ndarray | None, labels: np.ndarray | None, depths: np.ndarray | None,
             outdir, reason: str = "") -> list:
    """F14: PCA of activations at ~25/50/75% depth (qualitative separation)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if states is None or len(states) == 0:
        return _not_run_figure("fig14_activation_pca", outdir,
                               reason or "no activation tensors (GPU stage 06 did not run)",
                               "PC1", "PC2", DOUBLE)
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE)
    for ax, centre in zip(axes, (0.25, 0.50, 0.75)):
        mask = np.abs(depths - centre) < 0.125
        ax.set_xlabel(f"PC1 (states at {centre:.0%} +/- 12.5% depth)")
        ax.set_ylabel("PC2")
        if mask.sum() < 10:
            ax.text(0.5, 0.5, "too few states", ha="center", transform=ax.transAxes)
            continue
        components = PCA(n_components=2, random_state=0).fit_transform(states[mask])
        for value, colour, name in [(0, PALETTE["correct"], "trajectory succeeds"),
                                    (1, PALETTE["error"], "trajectory fails")]:
            selected = labels[mask] == value
            ax.scatter(components[selected, 0], components[selected, 1], s=3, alpha=0.5,
                       color=colour, label=name)
    axes[0].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    return save(fig, "fig14_activation_pca", outdir)
