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
coding = pd.read_csv(T / "decomp_recovered_coding.csv").category.value_counts()
put("codeN", int(coding.sum()), "{}")
for cat, name in [("flawed_reasoning_correct_answer", "Flawed"), ("low_entropy_or_qualitative_answer", "LowEnt"),
                  ("error_not_load_bearing", "NotLoad"), ("annotation_questionable", "Quest")]:
    put(f"code{name}", int(coding.get(cat, 0)), "{}")
put("codeRepair", int(coding.get("silent_repair", 0) + coding.get("explicit_self_correction", 0)), "{}")
lt = pd.read_csv(T / "table2d_length_only_tie_fix.csv").set_index("fraction")
put("lenFixA", lt.loc[0.25, "lexical_minus_length_fixed"], "{:+.3f}"); put("lenFixAP", lt.loc[0.25, "p_value"], "{:.3f}")

# ---- stage 13 (GPU) --------------------------------------------------------
KEYS = {"Qwen2.5-1.5B-Instruct": "QwenSmall", "Qwen2-1.5B-Instruct": "QwenTwo",
        "Qwen2.5-Math-7B-Instruct": "QwenMath", "Qwen2.5-7B-Instruct": "QwenSeven"}
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
    if "backtrack" in summ:
        for k_, name in [("bt_restart", "BtRestart"), ("bt_before_error", "BtErr"), ("bt_before_decisive", "BtDec"),
                         ("bt_best_prefix_oracle", "BtOracle")]:
            put(f"{key}{name}", summ["backtrack"][k_][0])
        dd = summ["backtrack_decisive_minus_error"]
        put(f"{key}BtDiff", dd[0], "{:+.3f}"); put(f"{key}BtDiffLo", dd[1], "{:+.3f}"); put(f"{key}BtDiffHi", dd[2], "{:+.3f}")
    ev = pd.read_csv(tables / "event_study.csv")
    for grp, gname in [("fatal", "Fat"), ("recovered", "Rec"), ("clean", "Clean")]:
        sub = ev[ev.group == grp].set_index("r")
        for r_, rname in [(-1, "Before"), (0, "After")]:
            if r_ in sub.index:
                put(f"{key}Ev{gname}{rname}", sub.loc[r_, "mean_v"])

out = ROOT / "paper" / "values.tex"
out.write_text("% generated by paper/make_values.py -- do not edit\n"
               + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(f"wrote {len(macros)} macros -> {out}")
