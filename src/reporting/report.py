"""
WHAT: Builds docs/RESULTS.md and results/paper_values.json from saved tables.

WHY:  Brief §8: every table and headline number in one place, with a mapping
      from each `\\RES{}` placeholder in main.tex to the exact value and the
      file it came from. Generating it (instead of typing it) is what keeps the
      paper, the tables and the report identical to the last decimal.

HOW:  Reads only files written by stages 02-09; computes nothing new except
      formatting. A missing table becomes NOT_RUN. Interpretive sentences are
      chosen by explicit rules on the numbers (e.g. p < 0.05), and the rules are
      printed next to the sentences so a reader can audit them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIGURES = [
    ("F1", "fig1_class_balance", "Class balance by dataset and tier"),
    ("F2", "fig2_chain_length", "Trace length, split by outcome"),
    ("F3", "fig3_absolute_error_index", "Absolute first-error index"),
    ("F4", "fig4_relative_error_position_HEADLINE", "Relative onset: histogram + CDF (paper Fig. 2)"),
    ("F5", "fig5_difficulty_breakdown", "Onset by difficulty tier"),
    ("F6", "fig6_roc_all_representations", "ROC curves, all representations"),
    ("F7", "fig7_auroc_vs_prefix_KEY", "AUROC vs observed prefix (paper Fig. 3)"),
    ("F7b", "fig7b_auroc_vs_prefix_process_error", "AUROC vs prefix, process-error target"),
    ("F8", "fig8_layer_depth_heatmap", "Layer x depth AUROC heatmap"),
    ("F9", "fig9_lead_time", "Lead time distribution + threshold tradeoff"),
    ("F10", "fig10_reliability", "Reliability diagram + ECE"),
    ("F11", "fig11_recovery_vs_corruption", "Recovery vs corruption per policy"),
    ("F12", "fig12_cost_frontier", "Accuracy change vs token cost"),
    ("F13", "fig13_transfer", "Cross-domain transfer"),
    ("F14", "fig14_activation_pca", "PCA of activations"),
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt_p(p: float) -> str:
    """p-values: scientific below 0.001, three decimals otherwise."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def _fmt_mc_p(p: float, n_simulations: int) -> str:
    """Monte Carlo p-value; at the resolution floor 1/(1+N) report it as a bound."""
    floor = 1.0 / (1 + n_simulations)
    return f"<= {floor:.3f} (Monte Carlo floor, {n_simulations} simulations)" if p <= floor + 1e-12 else f"{p:.3f}"


def _latex_p(p: float) -> str:
    """p-value as LaTeX math, e.g. 3.2\\times10^{-12}."""
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"{mantissa}\\times10^{{{int(exponent)}}}"
    return f"{p:.3f}"


def _curve_point(curve: pd.DataFrame, target: str, representation: str, fraction: float) -> pd.Series | None:
    rows = curve[(curve["target"] == target) & (curve["representation"] == representation)
                 & np.isclose(curve["fraction_observed"], fraction)]
    return rows.iloc[0] if not rows.empty else None


def build_paper_values(cfg) -> dict:
    """Every placeholder value for main.tex, each with its source file and field."""
    tables = Path(cfg.outputs.tables_dir)
    target = cfg.detector.target
    desc = _read_csv(tables / "table1_descriptive.csv")
    stats = _read_json(tables / "table_stats_tests.json")
    curve = _read_csv(tables / "table2_prefix_auroc.csv")
    placeholders, notes = {}, []

    def put(key: str, value, source: str, where: str, raw=None) -> None:
        placeholders[key] = {"value": value, "raw": raw, "source": source, "main_tex": where}

    if desc is not None:
        pooled = desc.set_index("subset").loc["ALL"]
        src = "results/tables/table1_descriptive.csv, row ALL"
        put("N_traces", f"{int(pooled['n_traces'])}", f"{src}, n_traces",
            "Abstract; Contributions item 2; Sec. V-A Datasets", int(pooled["n_traces"]))
        put("pct_erroneous", f"{pooled['pct_erroneous']:.1f}", f"{src}, pct_erroneous",
            "Sec. V-D Metrics (process-error rate)", float(pooled["pct_erroneous"]))
        put("pct_answer_incorrect", f"{pooled['pct_answer_incorrect']:.1f}", f"{src}, pct_answer_incorrect",
            "Sec. V-D Metrics (detector target rate)", float(pooled["pct_answer_incorrect"]))
        put("mean_rho", f"{pooled['mean_relative_onset']:.3f}", f"{src}, mean_relative_onset",
            "Abstract; Sec. VI-A", float(pooled["mean_relative_onset"]))
        put("mean_rho_ci", f"[{pooled['relative_onset_ci_lo']:.3f}, {pooled['relative_onset_ci_hi']:.3f}]",
            f"{src}, relative_onset_ci_lo/hi", "Sec. VI-A")
        put("expected_rho_uniform_step", f"{pooled['expected_mean_rho_uniform_step']:.3f}",
            f"{src}, expected_mean_rho_uniform_step", "Sec. VI-A (reference value)")
        put("median_rho", f"{pooled['median_relative_onset']:.3f}", f"{src}, median_relative_onset", "Sec. VI-A")
        put("pct_first_half", f"{pooled['pct_errors_first_half']:.1f}", f"{src}, pct_errors_first_half",
            "Sec. VI-A", float(pooled["pct_errors_first_half"]))
        put("mean_steps_remaining", f"{pooled['mean_steps_remaining']:.2f}", f"{src}, mean_steps_remaining",
            "Abstract; Sec. VI-A; Sec. VII-A", float(pooled["mean_steps_remaining"]))
        put("pct_final_step", f"{pooled['pct_errors_final_step']:.1f}", f"{src}, pct_errors_final_step",
            "Sec. VI-A", float(pooled["pct_errors_final_step"]))
        put("expected_pct_final_step", f"{pooled['expected_pct_final_step_uniform']:.1f}",
            f"{src}, expected_pct_final_step_uniform", "Sec. VI-A", float(pooled["expected_pct_final_step_uniform"]))
        put("mean_err_idx", f"{pooled['mean_err_idx']:.2f}", f"{src}, mean_err_idx", "Sec. VI-A")
        put("median_steps", f"{pooled['median_steps']:.0f}", f"{src}, median_steps", "Table I")
        onset_pct = int(round(100 * float(pooled["mean_relative_onset"])))
        put("motivation_onset_pct", f"{onset_pct}", f"round(100 x mean_rho) from {src}",
            "Sec. I-A Motivation (first percentage)", onset_pct)
        put("motivation_remaining_pct", f"{100 - onset_pct}", "100 - motivation_onset_pct",
            "Sec. I-A Motivation (second percentage)", 100 - onset_pct)
        for _, row in desc.iterrows():
            put(f"table1_{row['subset']}",
                f"{int(row['n_traces'])} & {row['pct_erroneous']:.1f} & {row['median_steps']:.0f} & "
                f"{row['mean_relative_onset']:.3f}",
                f"results/tables/table1_descriptive.csv, row {row['subset']}", "Table I")
    if stats is not None:
        ks, ksd, kw = stats["ks_uniform"], stats["ks_discrete_uniform_step_null"], stats["kruskal_tiers"]
        put("KS_D", f"{ks['statistic_D']:.3f}", "results/tables/table_stats_tests.json, ks_uniform.statistic_D",
            "Abstract; Sec. VI-A", ks["statistic_D"])
        put("KS_p", _latex_p(ks["p_value"]), "results/tables/table_stats_tests.json, ks_uniform.p_value",
            "Abstract; Sec. VI-A", ks["p_value"])
        put("KS_rejects_uniform", "yes" if ks["p_value"] < 0.05 else "no",
            "rule: p < 0.05", "Abstract wording 'not uniformly distributed'")
        floor = 1.0 / (1 + ksd["n_simulations"])
        put("KS_discrete_null_p",
            rf"\le {floor:.3f}" if ksd["p_value"] <= floor + 1e-12 else _latex_p(ksd["p_value"]),
            "results/tables/table_stats_tests.json, ks_discrete_uniform_step_null.p_value",
            "Sec. VI-A (robustness sentence)", ksd["p_value"])
        put("KS_discrete_null_D", f"{ksd['observed_D_vs_null']:.3f}",
            "results/tables/table_stats_tests.json, ks_discrete_uniform_step_null.observed_D_vs_null", "Sec. VI-A")
        put("KS_discrete_null_uniformD_q95", f"{ksd['null_D_uniform_q95']:.3f}",
            "results/tables/table_stats_tests.json, ks_discrete_uniform_step_null.null_D_uniform_q95",
            "Sec. VI-A (KS-to-uniform expected under the discrete null)")
        put("KS_discrete_null_mean_rho", f"{ksd['null_mean_rho']:.3f}",
            "results/tables/table_stats_tests.json, ks_discrete_uniform_step_null.null_mean_rho", "Sec. VI-A")
        put("KS_discrete_null_mean_p",
            rf"\le {floor:.3f}" if ksd["p_value_mean"] <= floor + 1e-12 else _latex_p(ksd["p_value_mean"]),
            "results/tables/table_stats_tests.json, ks_discrete_uniform_step_null.p_value_mean", "Sec. VI-A")
        put("KS_discrete_null_direction", ksd["direction"], "sign of observed - null mean rho", "Sec. VI-A")
        put("KW_H", f"{kw['H']:.2f}", "results/tables/table_stats_tests.json, kruskal_tiers.H", "Sec. VI-C", kw["H"])
        put("KW_p", _latex_p(kw["p_value"]), "results/tables/table_stats_tests.json, kruskal_tiers.p_value",
            "Sec. VI-C", kw["p_value"])
        put("KW_eta2", f"{kw['eta_squared']:.3f}", "results/tables/table_stats_tests.json, kruskal_tiers.eta_squared",
            "Sec. VI-C", kw["eta_squared"])
        put("KW_does_shift", "does" if kw["p_value"] < 0.05 else "does not", "rule: p < 0.05", "Sec. VI-C")
    if curve is not None:
        src = "results/tables/table2_prefix_auroc.csv"
        for representation, fraction, key in [("lexical", 0.25, "AUROC_lexical_25"),
                                               ("lexical", 1.0, "AUROC_lexical_100"),
                                               ("surface", 1.0, "AUROC_surface_100"),
                                               ("surface", 0.25, "AUROC_surface_25"),
                                               ("length-only", 0.25, "AUROC_length_25"),
                                               ("length-only", 1.0, "AUROC_length_100")]:
            point = _curve_point(curve, target, representation, fraction)
            if point is not None:
                put(key, f"{point['auroc']:.3f}",
                    f"{src}, target={target}, representation={representation}, fraction={fraction}",
                    "Abstract; Sec. VI-B" if representation == "lexical" else "Sec. VI-B",
                    float(point["auroc"]))
                put(key + "_ci", f"[{point['ci_lo']:.3f}, {point['ci_hi']:.3f}]", f"{src} ci_lo/ci_hi", "Sec. VI-B")
    put("prefix_grid", ", ".join(f"{100 * f:g}" for f in cfg.prefix_grid), "configs/default.yaml prefix_grid",
        "Contributions item 3")
    put("n_prefix_fractions", f"{len(cfg.prefix_grid)}", "configs/default.yaml prefix_grid (count)",
        "Contributions item 3")

    # ---- numbers quoted in Sec. V-VII beyond the original placeholders
    quality = _read_json(tables / "processed_traces_quality_report.json")
    if quality is not None:
        put("n_questions", f"{quality['n_questions']}", "processed_traces_quality_report.json, n_questions",
            "Sec. V-A Datasets", quality["n_questions"])
        put("dropped_by_p5", f"{quality['dropped_by_p5']}", "processed_traces_quality_report.json, dropped_by_p5",
            "Sec. V-A Datasets", quality["dropped_by_p5"])
    processed = Path(cfg.data.processed_dir) / "processed_traces.jsonl"
    if processed.exists():
        frame = pd.read_json(processed, lines=True)
        put("n_generators", f"{frame['generator'].nunique()}", "processed_traces.jsonl, distinct generator",
            "Sec. V-A; Sec. VIII", int(frame["generator"].nunique()))
    if desc is not None:
        pooled = desc.set_index("subset").loc["ALL"]
        put("n_erroneous", f"{int(pooled['n_erroneous'])}", "table1_descriptive.csv, row ALL, n_erroneous",
            "Fig. 2 caption", int(pooled["n_erroneous"]))
    desc_t = _read_csv(tables / "table1b_descriptive_transfer.csv")
    if desc_t is not None:
        tasks = desc_t[desc_t["subset"] != "ALL"]
        put("bbm_n_traces", f"{int(desc_t.set_index('subset').loc['ALL', 'n_traces'])}",
            "table1b_descriptive_transfer.csv, row ALL, n_traces", "Sec. V-A Datasets")
        put("bbm_n_tasks", f"{len(tasks)}", "table1b_descriptive_transfer.csv (rows)", "Sec. V-A Datasets")
        put("bbm_max_pct_incorrect", f"{tasks['pct_answer_incorrect'].max():.1f}",
            "table1b_descriptive_transfer.csv, max pct_answer_incorrect over tasks", "Sec. V-D Metrics")
        put("bbm_n_tasks_at_max_pct", f"{int(np.isclose(tasks['pct_answer_incorrect'], tasks['pct_answer_incorrect'].max()).sum())}",
            "table1b_descriptive_transfer.csv (count of tasks at that rate)", "Sec. V-D Metrics")
    if stats is not None:
        ksd = stats["ks_discrete_uniform_step_null"]
        put("KS_discrete_null_uniformD_mean", f"{ksd['null_D_uniform_mean']:.3f}",
            "table_stats_tests.json, ks_discrete_uniform_step_null.null_D_uniform_mean", "Sec. VI-A")
        put("KS_discrete_null_D95", f"{ksd['null_D_vs_null_q95']:.3f}",
            "table_stats_tests.json, ks_discrete_uniform_step_null.null_D_vs_null_q95", "Sec. VI-A")
        put("n_simulations", f"{ksd['n_simulations']}", "table_stats_tests.json, n_simulations", "Sec. VI-A")
    if curve is not None:
        target_curve = curve[curve["target"] == target]
        put("n_test", f"{int(target_curve['n_test'].iloc[0])}", "table2_prefix_auroc.csv, n_test",
            "Sec. V-C", int(target_curve["n_test"].iloc[0]))
        lexical = target_curve[target_curve["representation"] == "lexical"].set_index("fraction_observed")
        length = target_curve[target_curve["representation"] == "length-only"].set_index("fraction_observed")
        overlap = [f for f in lexical.index if lexical.loc[f, "ci_lo"] <= length.loc[f, "ci_hi"]]
        put("lexical_length_ci_overlap", ", ".join(f"{100 * f:g}" for f in overlap),
            "table2_prefix_auroc.csv, fractions where lexical ci_lo <= length-only ci_hi", "Sec. VI-B")
        point = _curve_point(curve, cfg.detector.secondary_target, "lexical", 1.0)
        if point is not None:
            put("AUROC_lexical_100_process_error", f"{point['auroc']:.3f}",
                f"table2_prefix_auroc.csv, target={cfg.detector.secondary_target}, lexical, fraction=1.0", "Sec. VI-B")
    lead = _read_csv(tables / "table_lead_summary.csv")
    if lead is not None and (lead["threshold"] == 0.5).any():
        row = lead[lead["threshold"] == 0.5].iloc[0]
        src = "table_lead_summary.csv, threshold=0.5"
        put("lead_alarm_rate_050", f"{100 * row['alarm_rate']:.1f}", f"{src}, alarm_rate", "Sec. VI-D")
        put("lead_warned_050", f"{100 * row['frac_warned_before_error']:.1f}", f"{src}, frac_warned_before_error",
            "Sec. VI-D")
        put("lead_median_delta_050", f"{row['median_delta']:.0f}", f"{src}, median_delta", "Sec. VI-D")
        put("lead_false_alarm_050", f"{100 * row['false_alarm_rate_error_free']:.1f}",
            f"{src}, false_alarm_rate_error_free", "Sec. VI-D; Sec. VII-C")
        put("lead_n_error_free", f"{int(row['n_error_free'])}", f"{src}, n_error_free", "Sec. VI-D")
        put("lead_n_erroneous", f"{int(row['n_traces'])}", f"{src}, n_traces", "Sec. VI-D")
    ece = _read_csv(tables / "table_calibration_ece.csv")
    if ece is not None:
        for detector_row in ece.itertuples():
            put(f"ECE_{detector_row.detector}", f"{detector_row.ece:.3f}",
                f"table_calibration_ece.csv, detector={detector_row.detector}", "Sec. VI-D")
    transfer = _read_csv(tables / "table_transfer.csv")
    if transfer is not None:
        lex = transfer[(transfer["representation"] == "lexical")
                       & transfer["evaluation_domain"].str.startswith("BBM: ")
                       & (transfer["evaluation_domain"] != "BBM: all tasks")]
        put("transfer_lexical_min", f"{lex['auroc'].min():.3f}", "table_transfer.csv, lexical, min over BBM tasks",
            "Sec. VI-D")
        put("transfer_lexical_max", f"{lex['auroc'].max():.3f}", "table_transfer.csv, lexical, max over BBM tasks",
            "Sec. VI-D")
        put("transfer_lexical_min_task", lex.loc[lex["auroc"].idxmin(), "evaluation_domain"].replace("BBM: ", ""),
            "table_transfer.csv, task with the minimum", "Sec. VI-D")
    notes.append("Model/temperature/samples/layer placeholders are CONFIG values (configs/default.yaml "
                 "probes.*), not results; the experiment that uses them (E4) did not run in this build.")
    return {"target": target, "placeholders": placeholders, "notes": notes}


def _pivot_curve(curve: pd.DataFrame, target: str) -> str:
    frame = curve[curve["target"] == target].copy()
    frame["cell"] = frame.apply(lambda r: f"{r['auroc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)
    frame["prefix"] = frame["fraction_observed"].map(lambda f: f"{100 * f:g}%")
    pivot = frame.pivot(index="prefix", columns="representation", values="cell")
    order = sorted(pivot.index, key=lambda s: float(s.rstrip("%")))
    return pivot.loc[order].reset_index().to_markdown(index=False)


def render_results_markdown(cfg, values: dict, stage_status: dict) -> str:
    """Full RESULTS.md text."""
    tables, figures_dir = Path(cfg.outputs.tables_dir), Path(cfg.outputs.figures_dir)
    target = cfg.detector.target
    p = values["placeholders"]
    lines = ["# Results", "",
             "> Auto-generated by `scripts/10_build_report.py` from the files in `results/tables/`. "
             "Every number was computed at runtime from data. Stages that did not run read `NOT_RUN` "
             "with the reason - never a guessed value. Re-generate with `make report`.", ""]
    if cfg.data.synthetic:
        lines += ["> **SMOKE-TEST BUILD** - all numbers come from labelled SYNTHETIC data "
                  "(`configs/quick.yaml`). Do not paste them anywhere.", ""]

    # ---------------- E1
    desc = _read_csv(tables / "table1_descriptive.csv")
    lines += ["## E1 - Where failures originate", ""]
    if desc is None:
        lines += ["NOT_RUN", ""]
    else:
        lines += [f"- Traces analysed: **{p['N_traces']['value']}** "
                  f"({p['pct_erroneous']['value']}% contain a process error; "
                  f"{p['pct_answer_incorrect']['value']}% have an incorrect final answer - the detector target y)",
                  f"- Mean relative onset rho: **{p['mean_rho']['value']}** 95% CI {p['mean_rho_ci']['value']} "
                  f"(median {p['median_rho']['value']}); reference value if the error step were uniform over each "
                  f"trace's own steps: {p['expected_rho_uniform_step']['value']}",
                  f"- First errors in the first half of the chain (rho <= 0.5): **{p['pct_first_half']['value']}%**",
                  f"- First error at the final step (e = n): {p['pct_final_step']['value']}% observed vs "
                  f"{p['expected_pct_final_step']['value']}% expected if e were uniform over steps",
                  f"- Mean first-error step e: {p['mean_err_idx']['value']}; "
                  f"mean steps remaining after it (n - e): **{p['mean_steps_remaining']['value']}**", "",
                  "Table I (ProcessBench, `table1_descriptive.csv`):", "",
                  desc[["subset", "n_traces", "pct_erroneous", "pct_answer_incorrect", "median_steps",
                        "mean_err_idx", "mean_relative_onset", "median_relative_onset",
                        "expected_mean_rho_uniform_step", "pct_errors_first_half",
                        "pct_errors_final_step", "expected_pct_final_step_uniform",
                        "mean_steps_remaining"]].round(3).to_markdown(index=False), ""]
        desc_t = _read_csv(tables / "table1b_descriptive_transfer.csv")
        if desc_t is not None:
            lines += ["Supplementary - BIG-Bench Mistake (`table1b_descriptive_transfer.csv`):", "",
                      desc_t[["subset", "n_traces", "pct_erroneous", "pct_answer_incorrect", "median_steps",
                              "mean_relative_onset", "expected_mean_rho_uniform_step",
                              "mean_steps_remaining"]].round(3).to_markdown(index=False), ""]

    # ---------------- E2
    stats = _read_json(tables / "table_stats_tests.json")
    lines += ["## E2 - Statistical tests (effect size beside every p-value)", ""]
    if stats is None:
        lines += ["NOT_RUN", ""]
    else:
        ks, ksd, kw = stats["ks_uniform"], stats["ks_discrete_uniform_step_null"], stats["kruskal_tiers"]
        mw = stats["mann_whitney_length_process_error"]
        mw_y = stats[f"mann_whitney_length_{target}"]
        lines += [
            f"- KS, rho vs Uniform(0,1): D = {ks['statistic_D']:.3f}, p = {_fmt_p(ks['p_value'])} (n = {ks['n']}). "
            f"Rule p < 0.05 -> {'uniformity rejected' if ks['p_value'] < 0.05 else 'uniformity NOT rejected'}.",
            f"- Robustness, discrete null e ~ Uniform{{1..n}} (Monte Carlo, {ksd['n_simulations']} simulations, "
            f"each trace keeps its own n). Under this null the KS distance to Uniform(0,1) is itself "
            f"{ksd['null_D_uniform_mean']:.3f} on average (95th percentile {ksd['null_D_uniform_q95']:.3f}) versus "
            f"the observed {ksd['observed_D_uniform']:.3f}, so short-trace discreteness alone produces a "
            f"Uniform(0,1) rejection of that size. Goodness of fit to the discrete null: D = "
            f"{ksd['observed_D_vs_null']:.3f} (null 95th percentile {ksd['null_D_vs_null_q95']:.3f}), "
            f"p {_fmt_mc_p(ksd['p_value'], ksd['n_simulations'])}. Mean rho {ksd['observed_mean_rho']:.3f} vs null "
            f"{ksd['null_mean_rho']:.3f}: {ksd['direction']}, p {_fmt_mc_p(ksd['p_value_mean'], ksd['n_simulations'])}. "
            + ("Rule p < 0.05 -> the onset distribution differs from errors spread evenly over steps."
               if ksd["p_value"] < 0.05 else
               "**Rule p >= 0.05 -> onset is NOT distinguishable from errors spread evenly over steps; "
               "the Uniform(0,1) rejection above must not be read as concentration.**"),
            f"- Mann-Whitney U on length, process-error vs error-free traces: p = {_fmt_p(mw['p_value'])}, "
            f"rank-biserial r = {mw['rank_biserial_r']:+.3f} (median n {mw['median_len_erroneous']:.0f} vs "
            f"{mw['median_len_correct']:.0f}; r > 0 means erroneous traces are longer)",
            f"- Mann-Whitney U on length, incorrect vs correct final answer: p = {_fmt_p(mw_y['p_value'])}, "
            f"r = {mw_y['rank_biserial_r']:+.3f} (median n {mw_y['median_len_erroneous']:.0f} vs "
            f"{mw_y['median_len_correct']:.0f})",
            f"- Kruskal-Wallis, rho across the {kw.get('k_groups', 'n/a')} tiers: H = {kw['H']:.2f}, "
            f"p = {_fmt_p(kw['p_value'])}, eta^2 = {kw['eta_squared']:.3f}. Rule p < 0.05 -> onset "
            f"{'does' if kw['p_value'] < 0.05 else 'does not'} shift with difficulty; eta^2 gives the size.", ""]

    # ---------------- E3
    curve = _read_csv(tables / "table2_prefix_auroc.csv")
    lines += ["## E3 - Predictability curve (AUROC [bootstrap 95% CI], held-out questions)", ""]
    if curve is None:
        lines += ["NOT_RUN", ""]
    else:
        n_test = int(curve["n_test"].iloc[0])
        lines += [f"Target y = `{target}` (final answer incorrect). Test set: {n_test} traces from questions "
                  "never seen in training (grouped split, disjointness asserted).", "",
                  _pivot_curve(curve, target), "",
                  f"Secondary target `{cfg.detector.secondary_target}` (a process error exists):", "",
                  _pivot_curve(curve, cfg.detector.secondary_target), ""]
        status = _read_csv(tables / "table2_representation_status.csv")
        if status is not None:
            lines += ["Representation status:", "", status.to_markdown(index=False), ""]
        lexical = curve[(curve["target"] == target) & (curve["representation"] == "lexical")].set_index("fraction_observed")
        length = curve[(curve["target"] == target) & (curve["representation"] == "length-only")].set_index("fraction_observed")
        margins = (lexical["auroc"] - length["auroc"]).dropna()
        overlap = [f for f in margins.index if lexical.loc[f, "ci_lo"] <= length.loc[f, "ci_hi"]]
        lines += ["Honest reading (rule-based):", "",
                  f"- Lexical minus length-only AUROC ranges from {margins.min():+.3f} to {margins.max():+.3f} "
                  "across prefixes. Because k = ceil(f n), the prefix length reveals n at every f, so length-only "
                  "is the confound floor the paper must clear.",
                  f"- Prefixes where the lexical CI overlaps the length-only CI: "
                  f"{', '.join(f'{100 * f:g}%' for f in overlap) if overlap else 'none'}.",
                  "- Uncertainty and hidden-state representations are NOT_RUN, so this build cannot say "
                  "whether internal states add anything beyond text.", ""]

    # ---------------- E4
    heat = _read_csv(tables / "table3_probe_heatmap.csv")
    lines += ["## E4 - Activation probing (layer x depth)", ""]
    if heat is None or heat.empty:
        lines += [f"NOT_RUN - stage 06 status `{stage_status.get('06_extract_activations', 'NOT_RUN')}`, "
                  f"stage 07 status `{stage_status.get('07_train_probes', 'NOT_RUN')}`. No CUDA GPU on the build "
                  "machine. F8 and F14 are explicit NOT RUN panels. Run `make activations probes` on a GPU.", ""]
    else:
        lines += [heat.to_markdown(index=False), ""]

    # ---------------- E5
    lead = _read_csv(tables / "table_lead_summary.csv")
    series = _read_json(tables / "stepwise_risk_series.json")
    lines += ["## E5 - Detection lead time (Delta = e - k*)", ""]
    if lead is None:
        lines += ["NOT_RUN", ""]
    else:
        lines += [f"Detector: `{series['representation']}` per-step logistic regressions (one per step count k, "
                  f"fitted up to k = {series['fitted_up_to_k']}, reused beyond), predicting `{series['target']}`, "
                  "evaluated on held-out test traces only. `frac_warned_before_error` counts never-alarming traces "
                  "as not warned.", "", lead.round(3).to_markdown(index=False), ""]
        specificity = lead.dropna(subset=["false_alarm_rate_error_free"])
        specificity = specificity[specificity["alarm_rate"] > 0]
        lines += ["Rule-based reading (an alarm is only informative if it is rarer on error-free traces):", ""]
        for row in specificity.itertuples():
            lines.append(f"- tau = {row.threshold:g}: alarms on {100 * row.alarm_rate:.1f}% of erroneous vs "
                         f"{100 * row.false_alarm_rate_error_free:.1f}% of error-free test traces "
                         f"(difference {100 * (row.alarm_rate - row.false_alarm_rate_error_free):+.1f} points); "
                         f"P(Delta > 0) = {row.frac_warned_before_error:.3f}.")
        lines += ["", "A positive Delta from an alarm that fires on most error-free traces too is not evidence of "
                  "useful early warning: most first errors occur after step 1, so an indiscriminate early alarm "
                  "produces Delta > 0 by construction.", ""]

    # ---------------- E6
    ece = _read_csv(tables / "table_calibration_ece.csv")
    lines += ["## E6 - Calibration (full prefix, test split, 10 equal-width bins)", ""]
    if ece is None:
        lines += ["NOT_RUN", ""]
    else:
        lines += [ece.round(4).to_markdown(index=False), "",
                  "Length-only produces rank scores, not probabilities, so it has no ECE. A detector with "
                  "high ECE cannot be thresholded at a nominal risk level without recalibration.", ""]

    # ---------------- E7
    transfer = _read_csv(tables / "table_transfer.csv")
    lines += ["## E7 - Cross-domain transfer (train: ProcessBench maths; evaluate: BIG-Bench Mistake)", ""]
    if transfer is None:
        lines += ["NOT_RUN", ""]
    else:
        lines += [transfer.round(3).to_markdown(index=False), ""]
        loto = _read_csv(tables / "table_transfer_leave_one_tier_out.csv")
        if loto is not None:
            lines += ["Within-maths, leave one ProcessBench tier out:", "", loto.round(3).to_markdown(index=False), ""]
        status = _read_csv(tables / "table_transfer_status.csv")
        if status is not None:
            lines += [status.to_markdown(index=False), ""]
        inverted = transfer[transfer["ci_hi"] < 0.5]
        above_in_domain = transfer[(transfer["ci_lo"] > transfer["in_domain_auroc"])
                                   & ~transfer["evaluation_domain"].str.contains("in-domain")]
        lines += ["Rule-based reading:", "",
                  "- Below chance with the whole 95% CI under 0.5 (the transferred ranking is inverted): "
                  + (", ".join(f"{r.representation} on {r.evaluation_domain} ({r.auroc:.3f})"
                               for r in inverted.itertuples()) or "none"),
                  "- CI entirely above the in-domain AUROC (the target task is easier for this feature set, "
                  "most plausibly because of within-task length differences, not transfer of a reasoning signal): "
                  + (", ".join(f"{r.representation} on {r.evaluation_domain} ({r.auroc:.3f})"
                               for r in above_in_domain.itertuples()) or "none"),
                  "- The pooled 'BBM: all tasks' AUROC mixes tasks with different base rates and lengths ("
                  + "; ".join(f"{r.subset}: {r.pct_answer_incorrect:.0f}% incorrect, median {r.median_steps:.0f} steps"
                              for r in _read_csv(tables / "table1b_descriptive_transfer.csv").itertuples()
                              if r.subset != "ALL")
                  + "), so it partly measures between-task differences and should not be quoted alone.", ""]
        lines += ["Note: BIG-Bench Mistake positive rates differ sharply by task (the dataset over-samples incorrect "
                  "answers for four of five tasks), so AUROC - which is base-rate independent - is the comparable "
                  "quantity, not accuracy.", ""]

    # ---------------- E8
    policy = _read_csv(tables / "table_intervention.csv")
    assumptions = _read_json(tables / "table_intervention_assumptions.json")
    lines += ["## E8 - Intervention simulation (offline)", ""]
    if policy is None:
        lines += ["NOT_RUN", ""]
    else:
        lines += ["> **SIMULATED, NOT MEASURED.** Which traces alarm, at which step, which would fail, and regeneration "
                  "token counts are measured from data. What a verification or regeneration then *does* is ASSUMED "
                  f"(`intervention.assumed` in the config): `{json.dumps(assumptions)}`. Self-consistency rows use an "
                  "independent-samples model, which is optimistic because real samples are correlated. None of these "
                  "numbers is evidence about a real model; they show how the policy's trade-off depends on the "
                  "detector's alarms under the stated assumptions.", "",
                  policy.round(4).to_markdown(index=False), ""]

    # ---------------- Datasets
    verification = _read_json(tables / "dataset_identifier_verification.json")
    lines += ["## Dataset identifier verification", ""]
    lines += [pd.DataFrame(verification).to_markdown(index=False) if verification else "NOT_RUN (stage 01)", ""]

    # ---------------- Figures
    lines += ["## Figure inventory", "", "| ID | File | Exists (PNG / PDF) | Content |", "|---|---|---|---|"]
    for fig_id, name, caption in FIGURES:
        png, pdf = (figures_dir / f"{name}.png").exists(), (figures_dir / f"{name}.pdf").exists()
        not_run = fig_id in ("F8", "F14") and (heat is None or heat.empty)
        lines.append(f"| {fig_id} | `{name}` | {'yes' if png else 'no'} / {'yes' if pdf else 'no'} | "
                     f"{caption}{' - **NOT RUN panel** (no GPU)' if not_run else ''} |")
    lines.append("")

    # ---------------- Paper mapping
    lines += ["## Paper mapping (`\\RES{}` placeholder in main.tex -> value -> source)", "",
              "| Key | Value | Where in main.tex | Source |", "|---|---|---|---|"]
    for key, entry in p.items():
        lines.append(f"| {key} | `{entry['value']}` | {entry['main_tex']} | {entry['source']} |")
    lines += ["", *[f"- {note}" for note in values["notes"]], ""]

    lines += ["## Stage status (from results/manifest.json)", "", "| Stage | Status |", "|---|---|"]
    for stage in ("01_download_data", "02_preprocess", "03_run_eda", "04_run_statistics", "05_train_baselines",
                  "06_extract_activations", "07_train_probes", "08_lead_time_analysis",
                  "09_simulate_intervention", "10_build_report"):
        lines.append(f"| {stage} | {stage_status.get(stage, 'NOT_RUN')} |")
    return "\n".join(lines) + "\n"
