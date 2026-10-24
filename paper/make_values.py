"""Write paper/values.tex: every number quoted in the paper, read from results tables.

The paper never types a result by hand; it uses these macros. Re-run after any experiment:
    python paper/make_values.py
GPU macros are emitted only for runs whose tables exist (results/commitment/tables_<generator>).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
C = ROOT / "results" / "commitment"
macros: dict = {}


def put(name: str, value, fmt: str = "{:.3f}") -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


# ---- stage 11 (CPU) --------------------------------------------------------
a = pd.read_csv(T / "decomp_A_fatality.csv")
row = lambda s, l: a[(a.stratum == s) & (a.level == l)].iloc[0]  # noqa: E731
r = row("all", "all")
put("fatAll", r.fatality); put("fatAllLo", r.ci_lo); put("fatAllHi", r.ci_hi)
put("nErr", int(r.n_errors), "{:,}")
put("nonFatalPct", pct(1 - r.fatality), "{}")
for tier, name in [("gsm8k", "Gsm"), ("math", "Math"), ("olympiadbench", "Oly"), ("omnimath", "Omni")]:
    t = row("tier", tier)
    put(f"fat{name}", t.fatality); put(f"fat{name}Lo", t.ci_lo); put(f"fat{name}Hi", t.ci_hi)
put("nonFatalOmniPct", pct(1 - row("tier", "omnimath").fatality), "{}")
onset = a[a.stratum == "relative_onset"]
put("fatOnsetMin", onset[onset.level != "last step"].fatality.min())
put("fatOnsetMax", onset[onset.level != "last step"].fatality.max())
put("fatLast", row("relative_onset", "last step").fatality)
s = json.loads((T / "decomp_summary.json").read_text())
put("mwRhoP", s["A"]["mann_whitney"]["rho"]["p_value"], "{:.2f}")
put("mwRhoR", s["A"]["mann_whitney"]["rho"]["rank_biserial_fatal_vs_recovered"], "{:.2f}")
put("nRecovered", s["A"]["n_recovered"], "{:,}")
put("nFatal", s["A"]["n_fatal"], "{:,}")

b = pd.read_csv(T / "decomp_B_controls_auroc.csv").set_index("detector")
for det, name in [("question-only", "Q"), ("generator+tier", "G"), ("question+generator", "QG"),
                  ("prefix 25%", "PA"), ("prefix 50%", "PB"), ("prefix 100%", "PC"),
                  ("Q+G+prefix 25%", "QGPA"), ("Q+G+prefix 50%", "QGPB"), ("Q+G+prefix 100%", "QGPC"),
                  ("length 100%", "LenC")]:
    put(f"auc{name}", b.loc[det, "auroc_primary"]); put(f"auc{name}Lo", b.loc[det, "ci_lo"])
    put(f"auc{name}Hi", b.loc[det, "ci_hi"]); put(f"auc{name}Sd", b.loc[det, "auroc_seed_sd"])
inc = pd.read_csv(T / "decomp_B_increments.csv")
for a_, b_, name in [("Q+G+prefix 25%", "question+generator", "IncA"), ("Q+G+prefix 50%", "question+generator", "IncB"),
                     ("Q+G+prefix 100%", "question+generator", "IncC"), ("prefix 25%", "question+generator", "PAvsQG")]:
    r = inc[(inc.a == a_) & (inc.b == b_)].iloc[0]
    put(f"d{name}", r.difference, "{:+.3f}"); put(f"d{name}Lo", r.ci_lo, "{:+.3f}")
    put(f"d{name}Hi", r.ci_hi, "{:+.3f}"); put(f"d{name}P", r.p_value, "{:.3f}")
wp = pd.read_csv(T / "decomp_B_within_problem.csv").set_index("detector")
put("wpProblems", int(wp.iloc[0].n_problems), "{:,}"); put("wpPairs", int(wp.iloc[0].n_pairs), "{:,}")
for det, name in [("generator+tier", "G"), ("prefix 25%", "PA"), ("prefix 100%", "PC"), ("Q+G+prefix 100%", "QGPC")]:
    put(f"wp{name}", wp.loc[det, "within_problem_concordance"])
    put(f"wp{name}Lo", wp.loc[det, "ci_lo"]); put(f"wp{name}Hi", wp.loc[det, "ci_hi"])
c3 = pd.read_csv(T / "decomp_C_three_way.csv")
g = lambda d, p, n: c3[(c3.detector == d) & (c3.positive == p) & (c3.negative == n)].iloc[0]  # noqa: E731
for d, name in [("prefix 100%", "Out"), ("Q+G+prefix 100%", "OutQG"), ("prefix 100% (trained on error)", "Err")]:
    for p_, n_, tag in [("recovered", "clean", "RC"), ("fatal", "recovered", "FR"), ("fatal", "clean", "FC")]:
        r = g(d, p_, n_)
        put(f"tw{name}{tag}", r.auroc); put(f"tw{name}{tag}Lo", r.ci_lo); put(f"tw{name}{tag}Hi", r.ci_hi)
d4 = pd.read_csv(T / "decomp_D_fatality_auroc.csv").set_index("detector")
put("fatPredQG", d4.loc["Q+G", "auroc_primary"]); put("fatPredPos", d4.loc["error position (online)", "auroc_primary"])
put("fatPredErrStep", d4.loc["Q+G+position+error step", "auroc_primary"])
d4i = pd.read_csv(T / "decomp_D_increments.csv")
put("fatPredMaxInc", d4i.difference.abs().max(), "{:.3f}")
e5 = pd.read_csv(T / "decomp_E_correction_markers.csv").set_index("group")
put("markRec", pct(e5.loc["recovered", "rate"]), "{}"); put("markFat", pct(e5.loc["fatal", "rate"]), "{}")
put("markP", s["E"]["fisher_p"], "{:.2f}")
lt = pd.read_csv(T / "table2d_length_only_tie_fix.csv").set_index("fraction")
put("lenFixA", lt.loc[0.25, "lexical_minus_length_fixed"], "{:+.3f}"); put("lenFixAP", lt.loc[0.25, "p_value"], "{:.3f}")

# ---- stage 13 (GPU) --------------------------------------------------------
KEYS = {"Qwen2.5-1.5B-Instruct": "QwenSmall", "Qwen2-1.5B-Instruct": "QwenTwo",
        "Qwen2.5-Math-7B-Instruct": "QwenMath", "Qwen2.5-7B-Instruct": "QwenSeven", "Llama-3.1-8B-Instruct": "Llama",
        # variants: new independent batches only, all batches, pooled new batches, threshold sensitivity
        "Qwen2.5-1.5B-Instruct_newbatches": "QwenSmallNew", "Qwen2-1.5B-Instruct_newbatches": "QwenTwoNew",
        "Qwen2.5-1.5B-Instruct_allbatches": "QwenSmallAll", "Qwen2-1.5B-Instruct_allbatches": "QwenTwoAll",
        "pooled_1p5b_newbatches": "PooledNew"}
for tables in sorted(C.glob("tables_*")):
    key = KEYS.get(tables.name.replace("tables_", ""))
    if key is None:
        continue
    summ = json.loads((tables / "summary.json").read_text())
    put(f"{key}N", summ["n_traces"], "{:,}")
    fe = summ["first_error_is_decisive"]
    put(f"{key}Dec", fe["share"][0]); put(f"{key}DecLo", fe["share"][1]); put(f"{key}DecHi", fe["share"][2])
    put(f"{key}DecChance", fe["chance"]); put(f"{key}DecN", fe["n"], "{}")
    cats = pd.read_csv(tables / "commitment_categories.csv")
    cats = cats[(cats.delta - 0.125).abs() < 1e-9].set_index("category")
    for cat, name in [("capability-limited", "Cap"), ("latent-commitment", "Latent"), ("commits-at-error", "AtErr"),
                      ("recoverable-error", "Recov"), ("never-commits", "Never")]:
        put(f"{key}{name}", cats.loc[cat, "share"]); put(f"{key}{name}Lo", cats.loc[cat, "ci_lo"])
        put(f"{key}{name}Hi", cats.loc[cat, "ci_hi"])
    nl = 1 - cats.loc["capability-limited", "share"]
    for cat, name in [("latent-commitment", "Latent"), ("commits-at-error", "AtErr"), ("recoverable-error", "Recov")]:
        put(f"{key}{name}NL", 100 * cats.loc[cat, "share"] / nl, "{:.0f}")  # % of fatal traces that are not capability-limited
    put(f"{key}CapPct", 100 * cats.loc["capability-limited", "share"], "{:.0f}")
    if "backtrack" in summ:
        for k_, name in [("bt_restart", "BtRestart"), ("bt_before_error", "BtErr"), ("bt_before_decisive", "BtDec"),
                         ("bt_best_prefix_oracle", "BtOracle")]:
            put(f"{key}{name}", summ["backtrack"][k_][0])
        for k_, name in [("bt_restart", "BtRestart"), ("bt_before_error", "BtErr"), ("bt_before_decisive", "BtDec"),
                         ("bt_best_prefix_oracle", "BtOracle")]:
            put(f"{key}{name}Lo", summ["backtrack"][k_][1]); put(f"{key}{name}Hi", summ["backtrack"][k_][2])
        for js, name in [("backtrack_decisive_minus_error", "BtDiff"), ("backtrack_restart_minus_error", "BtRestErr"),
                         ("backtrack_decisive_minus_restart", "BtDecRest"),
                         ("backtrack_all_fatal_decisive_minus_error", "BtDiffAll")]:
            dd = summ[js]
            put(f"{key}{name}", dd[0], "{:+.3f}"); put(f"{key}{name}Lo", dd[1], "{:+.3f}"); put(f"{key}{name}Hi", dd[2], "{:+.3f}")
        put(f"{key}BtN", summ["backtrack_n"], "{:.0f}")
        tests = summ.get("backtrack_tests", {})
        for js, name in [("dec_minus_err", "BtDiff"), ("restart_minus_err", "BtRestErr"), ("dec_minus_restart", "BtDecRest")]:
            if js in tests:
                pv = tests[js]["p"]
                put(f"{key}{name}P", "<0.001" if pv < 0.001 else f"{pv:.3f}", "{}")
                put(f"{key}{name}NinetyLo", tests[js]["ci90"][0], "{:+.3f}")
                put(f"{key}{name}NinetyHi", tests[js]["ci90"][1], "{:+.3f}")
        if "backtrack_tost_0.05" in summ:
            put(f"{key}Tost", summ["backtrack_tost_0.05"], "{}")
        tok = summ["backtrack_tokens"]
        put(f"{key}TokRest", tok["tok_restart"], "{:.0f}"); put(f"{key}TokErr", tok["tok_before_error"], "{:.0f}")
        put(f"{key}TokDec", tok["tok_before_decisive"], "{:.0f}")
        put(f"{key}TokSave", 100 * (1 - tok["tok_before_decisive"] / tok["tok_restart"]), "{:.0f}")
    put(f"{key}DecNull", fe["position_matched_null"])
    if "minus_null" in fe:
        put(f"{key}DecMinusNull", fe["minus_null"][0], "{:+.3f}"); put(f"{key}DecMinusNullLo", fe["minus_null"][1], "{:+.3f}")
        put(f"{key}DecMinusNullHi", fe["minus_null"][2], "{:+.3f}")
        put(f"{key}DecMinusNullP", "<0.001" if fe["minus_null_p"] < 0.001 else f"{fe['minus_null_p']:.3f}", "{}")
    put(f"{key}DecLast", fe["share_error_at_last_step"])
    put(f"{key}DecExcl", fe["share_excluding_last_step"][0])
    put(f"{key}NFatal", summ["groups"].get("fatal", 0), "{}"); put(f"{key}NClean", summ["groups"].get("clean", 0), "{}")
    put(f"{key}NRecov", summ["groups"].get("recovered", 0), "{}")
    pt = pd.read_csv(tables / "per_trace.csv")
    for grp, gname in [("fatal", "Fat"), ("recovered", "Rec")]:
        sub = pt[pt.group == grp]
        put(f"{key}Drop{gname}", sub.drop_at_error.mean(), "{:+.3f}")
    put(f"{key}Floor", pt[pt.group == "clean"].max_drop.mean(), "{:+.3f}")
    put(f"{key}Prefixes", int(pt.n.sum()), "{:,}")
    put(f"{key}M", int(pt.m.iloc[0]), "{}")
    put(f"{key}Rollouts", int((pt.n * pt.m).sum()), "{:,}")
    ev = pd.read_csv(tables / "event_study.csv")
    for grp, gname in [("fatal", "Fat"), ("recovered", "Rec"), ("clean", "Clean")]:
        sub = ev[ev.group == grp].set_index("r")
        for r_, rname in [(-1, "Before"), (0, "After")]:
            if r_ in sub.index:
                put(f"{key}Ev{gname}{rname}", sub.loc[r_, "mean_v"])

# ---- stage 16 robustness ----------------------------------------------------------
for rob in sorted(C.glob("robustness_*")):
    key = KEYS.get(rob.name.replace("robustness_", ""))
    if key is None or not (rob / "summary.json").exists():
        continue
    r = json.loads((rob / "summary.json").read_text())
    put(f"{key}SplitR", r["split_half"]["pearson_r"]); put(f"{key}SplitSB", r["split_half"]["spearman_brown_full_m"])
    if "decisive_posterior" in r:
        dp = r["decisive_posterior"]
        put(f"{key}PostDec", dp["posterior_mean_share_d_eq_e"][0]); put(f"{key}PostDecLo", dp["posterior_mean_share_d_eq_e"][1])
        put(f"{key}PostDecHi", dp["posterior_mean_share_d_eq_e"][2])
        put(f"{key}PostModal", 100 * dp["share_modal_ge_0.5"], "{:.0f}"); put(f"{key}PostModalHigh", 100 * dp["share_modal_ge_0.8"], "{:.0f}")
    if "decisive_agreement_between_halves" in r:
        da = r["decisive_agreement_between_halves"]
        put(f"{key}Agree", da["agreement"][0]); put(f"{key}AgreeNull", da["position_matched_null"])
    for row in r.get("definition_sensitivity", []):
        dn = {"largest_drop": "Arg", "point_of_no_return": "Commit", "first_drop_above_clean_floor": "Floor"}[row["definition"]]
        dl = {0.0625: "A", 0.125: "B", 0.25: "C"}[round(row["delta"], 4)]
        put(f"{key}Def{dn}{dl}", row["share_first_error"])

tot = 0
for tables in sorted(C.glob("tables_*")):
    if KEYS.get(tables.name.replace("tables_", "")) in ("QwenSmall", "QwenTwo", "QwenMath", "Llama"):
        pt = pd.read_csv(tables / "per_trace.csv"); tot += int((pt.n * pt.m).sum())
put("TotalRollouts", tot, "{:,}")
out = ROOT / "paper" / "values.tex"
out.write_text("% generated by paper/make_values.py -- do not edit\n"
               + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(f"wrote {len(macros)} macros -> {out}")
