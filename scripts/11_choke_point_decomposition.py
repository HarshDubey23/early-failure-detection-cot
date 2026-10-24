"""Stage 11: error-vs-choke-point decomposition and difficulty controls (CPU, real data).

Parts (all on ProcessBench, all grouped by question id):
  A  failure decomposition: P(error) and P(fatal | error) by tier, onset, steps left, generator
  B  difficulty controls: question-only, generator/tier-only, prefix, and their combination;
     increment of the prefix over question+generator; within-problem concordance
  C  what the failure detector tracks: clean vs recovered vs fatal
  D  fatality prediction at the moment of the first error
  E  explicit self-correction language after the first error
Thin CLI wrapper over src/analysis/decomposition.py. Writes results/tables/decomp_*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import decomposition as dec
from src.analysis.statistics import auroc, auroc_with_ci, paired_auroc_difference
from src.config import load_config
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything

N_FOLDS = 5
TFIDF = {"max_features": 20000, "ngram_range": (1, 2), "min_df": 2}


def cluster_rate_ci(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> tuple:
    """Mean of a 0/1 array with a question-level cluster bootstrap 95% CI."""
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({"g": groups, "v": values})
    sums = frame.groupby("g")["v"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = sums[rng.integers(0, len(sums), len(sums))]
        boot.append(pick[:, 0].sum() / pick[:, 1].sum())
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def part_a(df: pd.DataFrame, n_boot: int, seed: int) -> tuple:
    """Fatality of first errors, stratified."""
    from scipy import stats

    err = df[df.has_error == 1].copy()
    g = err["question_id"].to_numpy()
    overall = cluster_rate_ci(err["answer_incorrect"].to_numpy(), g, n_boot, seed)
    rows = [{"stratum": "all", "level": "all", "n_errors": len(err),
             "fatality": overall[0], "ci_lo": overall[1], "ci_hi": overall[2]}]

    def add(stratum: str, key: pd.Series) -> None:
        for level, sub in err.groupby(key, observed=True):
            m, lo, hi = cluster_rate_ci(sub["answer_incorrect"].to_numpy(),
                                        sub["question_id"].to_numpy(), n_boot, seed)
            rows.append({"stratum": stratum, "level": str(level), "n_errors": len(sub),
                         "fatality": m, "ci_lo": lo, "ci_hi": hi})

    add("tier", err["subset"])
    add("relative_onset", pd.cut(err["rho"], [0, 0.25, 0.5, 0.75, 0.999, 1.0],
                                 labels=["(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "last step"]))
    add("steps_after_error", pd.cut(err["steps_after_error"], [-1, 0, 1, 2, 3, 5, 1000],
                                    labels=["0", "1", "2", "3", "4-5", "6+"]))
    add("absolute_error_step", pd.cut(err["e"], [0, 1, 2, 3, 5, 1000], labels=["1", "2", "3", "4-5", "6+"]))
    add("generator", err["generator"])
    table = pd.DataFrame(rows)

    fatal, rec = err[err.answer_incorrect == 1], err[err.answer_incorrect == 0]
    tests = {}
    for col in ("rho", "e", "steps_after_error", "n_steps"):
        r = stats.mannwhitneyu(fatal[col], rec[col], alternative="two-sided")
        tests[col] = {"U": float(r.statistic), "p_value": float(r.pvalue),
                      "rank_biserial_fatal_vs_recovered": float(2 * r.statistic / (len(fatal) * len(rec)) - 1),
                      "median_fatal": float(fatal[col].median()), "median_recovered": float(rec[col].median()),
                      "mean_fatal": float(fatal[col].mean()), "mean_recovered": float(rec[col].mean())}

    # Adjusted association: logistic regression of fatality on onset and steps left,
    # controlling tier and generator; question-cluster bootstrap for the coefficients.
    from sklearn.linear_model import LogisticRegression

    X = pd.get_dummies(err[["subset", "generator"]], drop_first=True).astype(float)
    X["rho"] = err["rho"].to_numpy()
    X["log1p_steps_after"] = np.log1p(err["steps_after_error"].to_numpy())
    cols = list(X.columns)
    Xv, yv = X.to_numpy(), err["answer_incorrect"].to_numpy()
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-12
    Z = (Xv - mu) / sd
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(Z, yv)
    coef = fit.coef_[0] / sd
    frame = pd.DataFrame({"g": g, "i": np.arange(len(g))})
    groups_rows = [sub.i.to_numpy() for _, sub in frame.groupby("g")]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(min(n_boot, 500)):
        idx = np.concatenate([groups_rows[i] for i in rng.integers(0, len(groups_rows), len(groups_rows))])
        if len(np.unique(yv[idx])) < 2:
            continue
        b = LogisticRegression(C=1e6, max_iter=5000).fit(Z[idx], yv[idx]).coef_[0] / sd
        boot.append(b)
    boot = np.asarray(boot)
    regression = {}
    for name in ("rho", "log1p_steps_after"):
        j = cols.index(name)
        regression[name] = {"coef": float(coef[j]), "ci_lo": float(np.quantile(boot[:, j], 0.025)),
                            "ci_hi": float(np.quantile(boot[:, j], 0.975)),
                            "odds_ratio": float(np.exp(coef[j]))}
    return table, {"mann_whitney": tests, "adjusted_logit": regression,
                   "n_fatal": int(len(fatal)), "n_recovered": int(len(rec))}


def specs_for(df: pd.DataFrame) -> dict:
    """Every feature recipe used in parts B-D."""
    steps = df["steps"].tolist()
    spec = {
        "Q": dec.FeatureSpec("Q", "text", df["question"].tolist(), TFIDF),
        "G": dec.FeatureSpec("G", "onehot", list(zip(df["generator"], df["subset"]))),
    }
    for f in (0.25, 0.5, 1.0):
        texts = [" ".join(dec.prefix_by_fraction(s, f)) for s in steps]
        spec[f"P{int(f * 100)}"] = dec.FeatureSpec(f"P{int(f * 100)}", "text", texts, TFIDF)
        spec[f"L{int(f * 100)}"] = dec.FeatureSpec(
            f"L{int(f * 100)}", "numeric", [len(dec.prefix_by_fraction(s, f)) for s in steps])
    # Error-anchored views (only meaningful for erroneous traces; others get placeholders).
    e = df["e"].fillna(1).astype(int).to_numpy()
    n = df["n_steps"].to_numpy()
    spec["ERRSTEP"] = dec.FeatureSpec("ERRSTEP", "text", [s[k - 1] for s, k in zip(steps, e)], TFIDF)
    spec["PRE_E"] = dec.FeatureSpec("PRE_E", "text", [" ".join(s[:k]) for s, k in zip(steps, e)], TFIDF)
    spec["POST1"] = dec.FeatureSpec("POST1", "text", [" ".join(s[:min(k + 1, len(s))]) for s, k in zip(steps, e)], TFIDF)
    spec["POST2"] = dec.FeatureSpec("POST2", "text", [" ".join(s[:min(k + 2, len(s))]) for s, k in zip(steps, e)], TFIDF)
    spec["AFTER"] = dec.FeatureSpec("AFTER", "text", [" ".join(s[k:]) or "<none>" for s, k in zip(steps, e)], TFIDF)
    spec["EPOS"] = dec.FeatureSpec("EPOS", "numeric", np.c_[e, np.log(e)])
    spec["EPOS_OFFLINE"] = dec.FeatureSpec("EPOS_OFFLINE", "numeric",
                                           np.c_[e, np.log(e), n, e / n, np.log1p(n - e)])
    return spec


B_DETECTORS = {
    "question-only": ["Q"], "generator+tier": ["G"], "question+generator": ["Q", "G"],
    "prefix 25%": ["P25"], "prefix 50%": ["P50"], "prefix 100%": ["P100"],
    "Q+G+prefix 25%": ["Q", "G", "P25"], "Q+G+prefix 50%": ["Q", "G", "P50"],
    "Q+G+prefix 100%": ["Q", "G", "P100"],
    "length 25%": ["L25"], "length 100%": ["L100"],
}

D_DETECTORS = {
    "error position (online)": ["EPOS"],
    "Q+G": ["Q", "G"],
    "Q+G+position": ["Q", "G", "EPOS"],
    "Q+G+position+error step": ["Q", "G", "EPOS", "ERRSTEP"],
    "Q+G+position+prefix to error": ["Q", "G", "EPOS", "PRE_E"],
    "Q+G+position+1 step after": ["Q", "G", "EPOS", "POST1"],
    "Q+G+position+2 steps after": ["Q", "G", "EPOS", "POST2"],
    "position+length (offline)": ["EPOS_OFFLINE"],
    "Q+G+offline position+after-error text": ["Q", "G", "EPOS_OFFLINE", "AFTER"],
}


def run_detectors(detectors: dict, spec: dict, y: np.ndarray, groups: np.ndarray,
                  seeds: list, rows: np.ndarray | None, log) -> dict:
    """Out-of-fold scores for every detector and seed: {seed: {name: scores}}."""
    out = {}
    for seed in seeds:
        folds = dec.grouped_folds(groups, N_FOLDS, seed)
        out[seed] = {}
        for name, keys in detectors.items():
            out[seed][name] = dec.cross_fit_scores([spec[k] for k in keys], y, folds, seed=seed, rows=rows)
        log.info("    seed %s done", seed)
    return out


def summarise(scores: dict, y: np.ndarray, groups: np.ndarray, rows: np.ndarray,
              n_boot: int, primary_seed: int) -> pd.DataFrame:
    """AUROC per detector: mean/SD over seeds and the primary seed's cluster-bootstrap CI."""
    names = list(next(iter(scores.values())))
    table = []
    for name in names:
        per_seed = [auroc(y[rows], s[name][rows]) for s in scores.values()]
        ci = auroc_with_ci(y[rows], scores[primary_seed][name][rows], n_bootstrap=n_boot,
                           seed=primary_seed, groups=groups[rows])
        table.append({"detector": name, "auroc_seed_mean": float(np.mean(per_seed)),
                      "auroc_seed_sd": float(np.std(per_seed, ddof=1)),
                      "auroc_primary": ci["auroc"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                      "n": int(len(rows)), "positive_rate": float(y[rows].mean())})
    return pd.DataFrame(table)


def paired(scores: dict, y, groups, rows, pairs: list, n_boot: int, primary_seed: int) -> pd.DataFrame:
    out = []
    for a, b in pairs:
        r = paired_auroc_difference(y[rows], scores[primary_seed][a][rows], scores[primary_seed][b][rows],
                                    n_bootstrap=n_boot, seed=primary_seed, groups=groups[rows])
        signs = [np.sign(auroc(y[rows], s[a][rows]) - auroc(y[rows], s[b][rows])) for s in scores.values()]
        diffs = [auroc(y[rows], s[a][rows]) - auroc(y[rows], s[b][rows]) for s in scores.values()]
        out.append({"a": a, "b": b, **r, "diff_seed_mean": float(np.mean(diffs)),
                    "seeds_a_better": int(np.sum(np.array(signs) > 0)), "n_seeds": len(signs)})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="11: choke-point decomposition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("11_choke_point_decomposition", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables = Path(cfg.outputs.tables_dir)
    n_boot = args.bootstrap or cfg.metrics.bootstrap_resamples
    seeds = args.seeds or list(cfg.robustness.split_seeds)
    primary = cfg.seed

    df = dec.add_outcome_columns(read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl"))
    df = df.reset_index(drop=True)
    groups = df["question_id"].to_numpy()
    y = df["answer_incorrect"].to_numpy()
    all_rows = np.arange(len(df))
    summary = [f"traces={len(df)}  groups: " + ", ".join(f"{k}={v}" for k, v in df.group.value_counts().items())]
    results: dict = {"counts": df.group.value_counts().to_dict()}

    # ---- A
    log.info("A: fatality of first errors")
    table_a, tests_a = part_a(df, n_boot, primary)
    atomic_write_dataframe(table_a, tables / "decomp_A_fatality.csv")
    results["A"] = tests_a
    fa = table_a.iloc[0]
    summary.append(f"A  P(fatal | error) = {fa.fatality:.3f} [{fa.ci_lo:.3f}, {fa.ci_hi:.3f}] "
                   f"(n={int(fa.n_errors)})")

    # ---- B
    log.info("B: difficulty controls (%d seeds x %d folds)", len(seeds), N_FOLDS)
    spec = specs_for(df)
    scores_b = run_detectors(B_DETECTORS, spec, y, groups, seeds, None, log)
    table_b = summarise(scores_b, y, groups, all_rows, n_boot, primary)
    atomic_write_dataframe(table_b, tables / "decomp_B_controls_auroc.csv")
    pairs_b = [("Q+G+prefix 25%", "question+generator"), ("Q+G+prefix 50%", "question+generator"),
               ("Q+G+prefix 100%", "question+generator"), ("prefix 25%", "question+generator"),
               ("prefix 100%", "question+generator"), ("prefix 25%", "length 25%"),
               ("prefix 100%", "length 100%"), ("question-only", "length 100%")]
    table_b2 = paired(scores_b, y, groups, all_rows, pairs_b, n_boot, primary)
    atomic_write_dataframe(table_b2, tables / "decomp_B_increments.csv")
    wp_pairs = dec.within_problem_pairs(groups, y)
    wp_names = ["generator+tier", "question+generator", "prefix 25%", "prefix 100%",
                "Q+G+prefix 25%", "Q+G+prefix 100%", "length 100%"]
    table_b3 = dec.concordance_ci({k: scores_b[primary][k] for k in wp_names}, wp_pairs, groups,
                                  n_boot, primary, reference="generator+tier")
    per_seed_wp = {k: [dec.concordance(s[k], wp_pairs) for s in scores_b.values()] for k in wp_names}
    table_b3["seed_mean"] = [float(np.mean(per_seed_wp[k])) for k in table_b3.detector]
    table_b3["seed_sd"] = [float(np.std(per_seed_wp[k], ddof=1)) for k in table_b3.detector]
    atomic_write_dataframe(table_b3, tables / "decomp_B_within_problem.csv")
    for _, r in table_b.iterrows():
        summary.append(f"B  {r.detector:<22} AUROC {r.auroc_primary:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]"
                       f"  seeds {r.auroc_seed_mean:.3f}±{r.auroc_seed_sd:.3f}")
    for _, r in table_b2.iterrows():
        summary.append(f"B  {r.a} - {r.b}: {r.difference:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] p={r.p_value:.3f}"
                       f" (a better in {r.seeds_a_better}/{r.n_seeds} seeds)")
    for _, r in table_b3.iterrows():
        summary.append(f"B  within-problem {r.detector:<20} {r.within_problem_concordance:.3f} "
                       f"[{r.ci_lo:.3f},{r.ci_hi:.3f}] ({r.n_problems} problems, {r.n_pairs} pairs)")

    # ---- C
    log.info("C: what the failure detector tracks")
    y_err = df["has_error"].to_numpy()
    scores_c = run_detectors({"prefix 100% (trained on error)": ["P100"],
                              "prefix 25% (trained on error)": ["P25"]}, spec, y_err, groups, seeds, None, log)
    group_arr = df["group"].to_numpy()
    rows_c = []
    contrasts = [("recovered", "clean"), ("fatal", "recovered"), ("fatal", "clean")]
    sources = [(scores_b, "prefix 100%"), (scores_b, "prefix 25%"), (scores_b, "Q+G+prefix 100%"),
               (scores_c, "prefix 100% (trained on error)"), (scores_c, "prefix 25% (trained on error)")]
    for src, name in sources:
        for pos, neg in contrasts:
            rows = np.flatnonzero(np.isin(group_arr, [pos, neg]))
            yy = (group_arr[rows] == pos).astype(int)
            ci = auroc_with_ci(yy, src[primary][name][rows], n_bootstrap=n_boot, seed=primary, groups=groups[rows])
            per_seed = [auroc(yy, s[name][rows]) for s in src.values()]
            rows_c.append({"detector": name, "positive": pos, "negative": neg, **ci,
                           "seed_mean": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed, ddof=1))})
    table_c = pd.DataFrame(rows_c)
    atomic_write_dataframe(table_c, tables / "decomp_C_three_way.csv")
    for _, r in table_c.iterrows():
        summary.append(f"C  {r.detector:<32} {r.positive:>9} vs {r.negative:<9} {r.auroc:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]")

    # ---- D
    log.info("D: fatality at the moment of the first error")
    err_rows = np.flatnonzero(df["has_error"].to_numpy() == 1)
    scores_d = run_detectors(D_DETECTORS, spec, y, groups, seeds, err_rows, log)
    table_d = summarise(scores_d, y, groups, err_rows, n_boot, primary)
    atomic_write_dataframe(table_d, tables / "decomp_D_fatality_auroc.csv")
    pairs_d = [("Q+G+position+error step", "Q+G+position"), ("Q+G+position+prefix to error", "Q+G+position"),
               ("Q+G+position+1 step after", "Q+G+position+prefix to error"),
               ("Q+G+position+2 steps after", "Q+G+position+prefix to error"),
               ("Q+G+position", "Q+G")]
    table_d2 = paired(scores_d, y, groups, err_rows, pairs_d, n_boot, primary)
    atomic_write_dataframe(table_d2, tables / "decomp_D_increments.csv")
    for _, r in table_d.iterrows():
        summary.append(f"D  {r.detector:<40} AUROC {r.auroc_primary:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]"
                       f"  seeds {r.auroc_seed_mean:.3f}±{r.auroc_seed_sd:.3f}")
    for _, r in table_d2.iterrows():
        summary.append(f"D  {r.a} - {r.b}: {r.difference:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] p={r.p_value:.3f}")

    # ---- E
    log.info("E: self-correction language")
    from scipy.stats import fisher_exact

    e_idx = df["e"].fillna(0).astype(int).to_numpy()
    after = [any(dec.has_correction_marker(t) for t in s[k:]) for s, k in zip(df["steps"], e_idx)]
    at_err = [dec.has_correction_marker(s[k - 1]) if k > 0 else False for s, k in zip(df["steps"], e_idx)]
    anywhere = [any(dec.has_correction_marker(t) for t in s) for s in df["steps"]]
    df["marker_after_error"], df["marker_in_error_step"], df["marker_anywhere"] = after, at_err, anywhere
    rows_e = []
    for grp in ("clean", "recovered", "fatal"):
        sub = df[df.group == grp]
        col = "marker_anywhere" if grp == "clean" else "marker_after_error"
        m, lo, hi = cluster_rate_ci(sub[col].astype(float).to_numpy(), sub["question_id"].to_numpy(), n_boot, primary)
        rows_e.append({"group": grp, "measure": col, "rate": m, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
    rec, fat = df[df.group == "recovered"], df[df.group == "fatal"]
    odds, p = fisher_exact([[rec.marker_after_error.sum(), (~rec.marker_after_error).sum()],
                            [fat.marker_after_error.sum(), (~fat.marker_after_error).sum()]])
    table_e = pd.DataFrame(rows_e)
    atomic_write_dataframe(table_e, tables / "decomp_E_correction_markers.csv")
    results["E"] = {"fisher_odds_recovered_vs_fatal": float(odds), "fisher_p": float(p),
                    "recovered_with_marker_after_error": int(rec.marker_after_error.sum()),
                    "n_recovered": int(len(rec))}
    for _, r in table_e.iterrows():
        summary.append(f"E  {r.group:<9} {r.measure:<20} {r.rate:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] (n={r.n})")
    summary.append(f"E  Fisher recovered vs fatal (marker after error): OR={odds:.2f}, p={p:.2e}")

    # Sample of recovered traces for manual reading (seeded, deterministic).
    sample = rec.sample(n=min(40, len(rec)), random_state=primary)
    lines = ["# Recovered traces: first error annotated, final answer correct (seeded sample)\n"]
    for _, r in sample.iterrows():
        k = int(r.e)
        lines.append(f"## {r.trace_id}  (tier={r.subset}, generator={r.generator}, e={k}, n={r.n_steps})\n")
        lines.append(f"**Question:** {r.question[:400]}\n")
        for j, step in enumerate(r.steps, start=1):
            tag = "ERROR" if j == k else ("after" if j > k else "before")
            if j >= k - 1:
                lines.append(f"- s{j} [{tag}]: {step[:700]}")
        lines.append("")
    atomic_write_text(tables / "decomp_recovered_sample.md", "\n".join(lines))

    atomic_write_text(tables / "decomp_summary.json", json.dumps(results, indent=2, default=float))
    manifest.log("11_choke_point_decomposition", args.config, "OK",
                 {"seeds": seeds, "n_folds": N_FOLDS, "bootstrap": n_boot})
    for line in summary:
        log.info(line)
    print_summary_block("11_choke_point_decomposition", summary)


if __name__ == "__main__":
    main()
