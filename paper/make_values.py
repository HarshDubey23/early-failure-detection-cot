"""Write paper/values.tex: every number quoted in the paper, read from results tables.

The paper never types a result by hand; it uses these macros. Re-run after any experiment:
    python paper/make_values.py
GPU macros are emitted only for runs whose tables exist (results/commitment/tables_<generator>).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
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
        "pooled_1p5b_newbatches": "PooledNew", "Llama-3.1-8B-Instruct_droptrunc": "LlamaTrunc",
        "Qwen2.5-Math-7B-Instruct_droptrunc": "QwenMathTrunc",
        "Qwen2.5-1.5B-Instruct_allbatches_delta0.0625": "QwenSmallAllDA", "Qwen2.5-1.5B-Instruct_allbatches_delta0.25": "QwenSmallAllDC",
        "Qwen2-1.5B-Instruct_allbatches_delta0.0625": "QwenTwoAllDA", "Qwen2-1.5B-Instruct_allbatches_delta0.25": "QwenTwoAllDC",
        "Qwen2.5-Math-7B-Instruct_delta0.0625": "QwenMathDA", "Qwen2.5-Math-7B-Instruct_delta0.25": "QwenMathDC",
        "Llama-3.1-8B-Instruct_delta0.0625": "LlamaDA", "Llama-3.1-8B-Instruct_delta0.25": "LlamaDC"}
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
                put(f"{key}{name}P", "$<$0.001" if pv < 0.001 else f"{pv:.3f}", "{}")
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
        put(f"{key}DecMinusNullP", "$<$0.001" if fe["minus_null_p"] < 0.001 else f"{fe['minus_null_p']:.3f}", "{}")
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

# ---- stage 17 policies, stage 19 PRM check, stage 20 answer curves --------------------
POL = {"P0_restart": "Restart", "P1_prm_first_error": "PrmFirst", "P2_prm_largest_drop": "PrmDrop",
       "P3_probe_search": "Probe", "P4_learned": "Learned", "P5_human_first_error": "Human", "P6_oracle": "Oracle"}
pol_file = C / "policies" / "policy_summary.csv"
if pol_file.exists():
    PS = pd.read_csv(pol_file)
    for (gen, b, acct, pop, pol), r in PS.set_index(["generator", "budget", "accounting", "population", "policy"]).iterrows():
        key = KEYS.get(gen)
        if key is None or acct not in ("flops", "flops_shared_verifier") or pop != "all_fatal" or b not in (1, 4, 8):
            continue
        tag = {1: "One", 4: "Four", 8: "Eight"}[b] + ("Sh" if acct == "flops_shared_verifier" else "")
        put(f"{key}Pol{POL[pol]}{tag}", r.success)
        put(f"{key}Pol{POL[pol]}{tag}Diff", r.minus_restart, "{:+.3f}")
        put(f"{key}Pol{POL[pol]}{tag}Lo", r.diff_lo, "{:+.3f}")
        put(f"{key}Pol{POL[pol]}{tag}Hi", r.diff_hi, "{:+.3f}")
        put(f"{key}Pol{POL[pol]}{tag}N", r.attempts, "{:.1f}")
prm_file = C / "prm_check.json"
if prm_file.exists():
    pc = json.loads(prm_file.read_text())
    put("prmFOne", 100 * pc["mean_f1_over_subsets"], "{:.1f}")
    for sub, name in [("gsm8k", "Gsm"), ("math", "Math"), ("olympiadbench", "Olym"), ("omnimath", "Omni")]:
        if sub in pc:
            put(f"prmFOne{name}", 100 * pc[sub]["f1"], "{:.1f}")
    put("prmN", pc["all"]["n"], "{:,}")
ans_file = C / "answer_curves" / "summary.json"
if ans_file.exists():
    for gen, d in json.loads(ans_file.read_text()).items():
        key = KEYS.get(gen)
        if key is None:
            continue
        if "recovered_answer_already_correct_at_error" in d:
            v = d["recovered_answer_already_correct_at_error"]
            put(f"{key}AnsRecFixed", v[0]); put(f"{key}AnsRecFixedLo", v[1]); put(f"{key}AnsRecFixedHi", v[2])
            put(f"{key}AnsRecN", d["recovered_n"], "{}")
        for c in ["never_right", "wrong_before_error", "turns_wrong_at_error", "turns_wrong_later"]:
            if f"fatal_{c}" in d:
                nm = "".join(w.capitalize() for w in c.split("_"))
                put(f"{key}AnsFat{nm}", d[f"fatal_{c}"][0])

# ---- Llama confirmatory tests: Holm adjustment over C1-C3 -----------------------------
lt = C / "tables_Llama-3.1-8B-Instruct" / "summary.json"
if lt.exists():
    ls = json.loads(lt.read_text())
    raw = {"COne": ls["backtrack_tests"]["restart_minus_err"]["p"], "CTwo": ls["backtrack_tests"]["dec_minus_err"]["p"],
           "CThree": max(ls["first_error_is_decisive"]["minus_null_p"], 1 / 2000)}
    order = sorted(raw, key=raw.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * raw[k]))
        adj[k] = running
    for k, v in adj.items():
        put(f"LlamaHolm{k}", "$<$0.002" if v < 0.002 else f"{v:.2f}", "{}")

# ---- truncation and answer formatting of the continuations -----------------------------
def trunc_stats(files, cap=1020):
    tot = at_cap = boxed = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("skipped") or not r.get("m"):
                continue
            tot += r["m"]
            at_cap += sum(1 for t, a in zip(r["new_tokens"], r["answers"]) if t >= cap and a is None)
            boxed += r.get("n_boxed", 0)
    return (100 * at_cap / tot, 100 * boxed / tot) if tot else (float("nan"), float("nan"))


RC0 = ROOT / "results" / "run_c"
for files, key in [(sorted(RC0.glob("rollouts.Llama-3.1-8B-Instruct.*.jsonl")), "Llama"),
                   (sorted(C.glob("Qwen2.5-Math-7B-Instruct_s*.jsonl")), "QwenMath")]:
    if files:
        tr, bx = trunc_stats(files)
        put(f"{key}TruncPct", tr, "{:.0f}"); put(f"{key}BoxedPct", bx, "{:.0f}")

# ---- question coverage --------------------------------------------------------------
pt_file = ROOT / "data" / "processed" / "processed_traces.jsonl"
if pt_file.exists():
    import re as _re
    qs: dict = {}
    for line in open(pt_file, encoding="utf-8"):
        r = json.loads(line)
        qs.setdefault(r["generator"], set()).add(_re.sub(r"\s+", " ", r["question"]).strip().lower())
    qwen = qs.get("Qwen2.5-1.5B-Instruct", set()) | qs.get("Qwen2-1.5B-Instruct", set()) | qs.get("Qwen2.5-Math-7B-Instruct", set())
    ll = qs.get("Llama-3.1-8B-Instruct", set())
    put("QuestionsQwen", len(qwen), "{:,}"); put("QuestionsLlama", len(ll), "{:,}")
    put("QuestionsLlamaNew", len(ll - qwen), "{:,}"); put("QuestionsUnion", len(ll | qwen), "{:,}")
else:  # the dataset text is not committed; fall back to the counts cached when it was present
    qc = json.loads((T / "question_coverage.json").read_text())
    for k in ("QuestionsQwen", "QuestionsLlama", "QuestionsLlamaNew", "QuestionsUnion"):
        put(k, qc[k], "{:,}")

# ---- totals straight from the raw JSONL files ------------------------------------------
def count_rows(files):
    n = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 0 if r.get("skipped") else int(r.get("m", 0))
    return n


RC = ROOT / "results" / "run_c"
roll_files = sorted(C.glob("*_s*.jsonl")) + sorted(RC.glob("rollouts.*.jsonl"))
put("TotalRollouts", count_rows(roll_files), "{:,}")
put("RunBRollouts", count_rows(sorted(C.glob("*_s*.jsonl"))), "{:,}")
put("RunCRollouts", count_rows(sorted(RC.glob("rollouts.*.jsonl"))), "{:,}")
put("ProbeSamples", count_rows(sorted(RC.glob("probe.*.jsonl"))), "{:,}")
prm_rows = [json.loads(x) for x in open(RC / "signals_prm.jsonl")] if (RC / "signals_prm.jsonl").exists() else []
put("PrmTraces", sum(1 for r in prm_rows if r.get("ok")), "{:,}")
# ---- stage 49: ChokeBench --------------------------------------------------------------
cb = ROOT / "results" / "chokebench" / "summary.csv"
if cb.exists():
    S = pd.read_csv(cb)
    P = S[S.generator == "pooled"].set_index("localiser")
    for loc, name in [("human_first_error", "Human"), ("prm_first_below_0.5", "PrmFirst"), ("prm_min", "PrmMin"),
                      ("prm_largest_drop", "PrmDrop"), ("gen_max_token_nll", "Nll"), ("gen_max_token_entropy", "Ent"),
                      ("first_step", "First"), ("last_step", "Last"), ("uniform_random", "Rand"),
                      ("learned_localiser", "Learned"), ("oracle_decisive_split", "Oracle")]:
        r = P.loc[loc]
        for col, cn, f in [("acc_human", "AccH", "{:.2f}"), ("acc_decisive", "AccD", "{:.2f}"),
                           ("gain", "Gain", "{:+.3f}"), ("gain_b4_free", "GainB", "{:+.3f}")]:
            if not np.isnan(r[col]):
                put(f"cb{name}{cn}", r[col], f)
                put(f"cb{name}{cn}Lo", r[col + "_lo"], f)
                put(f"cb{name}{cn}Hi", r[col + "_hi"], f)
        if not np.isnan(r.token_ratio):
            put(f"cb{name}Tok", r.token_ratio, "{:.2f}")
        put(f"cb{name}AccHPct", f"{100 * r.acc_human:.0f}" if not np.isnan(r.acc_human) else "--", "{}")
        put(f"cb{name}AccDPct", f"{100 * r.acc_decisive:.0f}" if not np.isnan(r.acc_decisive) else "--", "{}")
    put("cbN", int(P.loc["first_step", "n_traces"]), "{:,}")
    put("cbNeff", P.loc["first_step", "effective_n"], "{:.0f}")
    for gen, key in [("Qwen2.5-1.5B-Instruct", "QwenSmall"), ("Qwen2-1.5B-Instruct", "QwenTwo"),
                     ("Qwen2.5-Math-7B-Instruct", "QwenMath"), ("Llama-3.1-8B-Instruct", "Llama")]:
        G = S[S.generator == gen].set_index("localiser")
        for loc, name in [("human_first_error", "Human"), ("prm_largest_drop", "PrmDrop"),
                          ("prm_first_below_0.5", "PrmFirst"), ("oracle_decisive_split", "Oracle")]:
            put(f"cb{key}{name}AccD", G.loc[loc, "acc_decisive"], "{:.2f}") if not np.isnan(G.loc[loc, "acc_decisive"]) else None
            put(f"cb{key}{name}GainB", G.loc[loc, "gain_b4_free"], "{:+.3f}")

lt = ROOT / "results" / "chokebench" / "learned_tuning.json"
if lt.exists():
    rep = json.loads(lt.read_text())
    put("ltGrid", len(rep["grid"]), "{}")
    for gen, key in [("Qwen2.5-1.5B-Instruct", "QwenSmall"), ("Qwen2-1.5B-Instruct", "QwenTwo"),
                     ("Qwen2.5-Math-7B-Instruct", "QwenMath"), ("Llama-3.1-8B-Instruct", "Llama")]:
        o = rep["outer"][gen]
        put(f"lt{key}Inner", max(o["inner_scores"]), "{:.2f}")
        put(f"lt{key}Test", o["test_acc_decisive"], "{:.2f}")
        put(f"lt{key}Feat", o["selected"]["features"], "{}")

# ---- Run R (Part 3): preregistered tests, mechanism, evaluator robustness ------------------
RF = ROOT / "results" / "recovery" / "final"
tests = json.loads((RF / "tests.json").read_text())
fam = tests["families"]["ragtruth/llama-2-7b-chat"]


def put_ci(name: str, d: dict, f: str = "{:+.3f}") -> None:
    put(name, d["diff"], f); put(name + "Lo", d["lo"], f); put(name + "Hi", d["hi"], f)


put_ci("rrHZero", fam["sanity_care_vs_base"])
put("rrHZeroP", fam["sanity_care_vs_base"]["p"], "{:.4f}")
put_ci("rrHOne", fam["confirmatory"]["care_vs_restart_fb"])
put_ci("rrHTwo", fam["confirmatory"]["care_vs_backtrack_fb"])
put_ci("rrHFive", fam["confirmatory"]["care_vs_care@restart"])
put_ci("rrHFiveEtwo", fam["E2_robustness"]["care_vs_care@restart"])
put_ci("rrOracle", fam["oracle_headroom"]["oracle_vs_restart_allocation"])
eq = fam["equivalence_care_vs_restart_fb"]
put("rrTostLo", eq["lo90"], "{:+.3f}"); put("rrTostHi", eq["hi90"], "{:+.3f}")
summ = pd.read_csv(RF / "summary.csv")
ss = summ[(summ.benchmark == "ragtruth") & (summ.scope == "all")]
for pol, b, name in [("base", 0, "Base"), ("restart_fb", 2, "Restart"), ("backtrack_fb", 2, "Backtrack"), ("care", 2, "Care")]:
    r = ss[(ss.policy == pol) & (ss.budget == b)].iloc[0]
    put(f"rr{name}EOne", pct(r.hall_E1), "{}"); put(f"rr{name}ETwo", pct(r.hall_E2), "{}")
    put(f"rr{name}Cov", pct(r.covered), "{}"); put(f"rr{name}Tok", r.cost_total, "{:,.0f}")
mech = json.loads((RF / "mechanism" / "summary.json").read_text())
r1 = mech["R1_recovery"]["E1"]
put("mechN", r1["n"], "{}"); put("mechCont", mech["n_continuations"], "{:,}")
put("mechDiff", r1["difference"][0], "{:+.3f}"); put("mechDiffLo", r1["difference"][1], "{:+.3f}")
put("mechDiffHi", r1["difference"][2], "{:+.3f}"); put("mechP", r1["p_signflip"], "{:.3f}")
put("mechSum", r1["by_task"]["Summary"][0], "{:+.3f}")
sn = mech["R2_snowball"]["excess_jump"]
put("mechSnow", sn[0], "{:+.3f}"); put("mechSnowLo", sn[1], "{:+.3f}"); put("mechSnowHi", sn[2], "{:+.3f}")
ev = tests["primary_metric"]["E1_response_level"]
put("evEOneKappa", ev["kappa"]); put("evEOneRate", pct(ev["rate_pred"]), "{}"); put("evHumanRate", pct(ev["rate_true"]), "{}")
cs = json.loads((RF / "cove_sensitivity.json").read_text())
put("coveEcho", pct(cs["share_echoing_verification_block"]), "{}")
put("coveN", cs["n"], "{}")
put("coveDiff", cs["care_minus_cove_before_echo"][0], "{:+.3f}")
put("coveDiffLo", cs["care_minus_cove_before_echo"][1], "{:+.3f}"); put("coveDiffHi", cs["care_minus_cove_before_echo"][2], "{:+.3f}")

ml_file = RF / "mitigation_law.json"
if ml_file.exists():
    ml = json.loads(ml_file.read_text())
    for k, name in [("h", "H"), ("alpha", "Alpha"), ("beta", "Beta"), ("pi", "Pi"), ("reduction_vs_base", "Red"),
                    ("implied_halluc_D_passed_cont", "Inv")]:
        v, lo, hi = ml["law"][k]
        put(f"law{name}", f"{100 * v:.1f}", "{}"); put(f"law{name}Lo", f"{100 * lo:.1f}", "{}")
        put(f"law{name}Hi", f"{100 * hi:.1f}", "{}")
    for c in ml["cost"]:
        key = {"base": "Base", "delete": "Del", "repairdel": "Rep", "restart_fb": "Dra", "backtrack_fb": "Back",
               "care": "Care", "abstain_flag": "Abst", "cove_all": "Cove"}[c["policy"]]
        put(f"cost{key}", c["cost_vs_base"], "{:.2f}"); put(f"tok{key}", c["tokens"], "{:,.0f}")
        put(f"hallEOne{key}", f"{100 * c['hall_E1']:.1f}", "{}"); put(f"hallETwo{key}", f"{100 * c['hall_E2']:.1f}", "{}")
        put(f"cov{key}", f"{100 * c['coverage']:.1f}", "{}")
    rec = ml["maths_restart_recovery"]
    for b, nm in [("4", "Four"), ("8", "Eight"), ("1", "One")]:
        vals = [d[b][0] for d in rec.values()]
        put(f"recMin{nm}", f"{100 * min(vals):.0f}", "{}"); put(f"recMax{nm}", f"{100 * max(vals):.0f}", "{}")

lt_file = RF / "law_tests.json"
if lt_file.exists():
    lt = json.loads(lt_file.read_text())
    r = lt["budget_invariance"]["restart_fb"]
    for key, name in [("coverage_diff", "Cov"), ("hall_E1_delivered_diff", "EOne"), ("hall_E2_delivered_diff", "ETwo")]:
        v, lo, hi, _ = r[key]
        put(f"inv{name}", f"{100 * v:+.1f}", "{}"); put(f"inv{name}Lo", f"{100 * lo:+.1f}", "{}")
        put(f"inv{name}Hi", f"{100 * hi:+.1f}", "{}")
    put("invCovOne", f"{100 * r['coverage_b1']:.1f}", "{}"); put("invCovFour", f"{100 * r['coverage_b4']:.1f}", "{}")
    pick = {(c["gate"].split(" ")[0], round(c["tau"], 1)): c for c in lt["operating_curve"]}
    for (g, t), name in [(("D", 0.5), "DFive"), (("E2", 0.3), "ETwoThree"), (("D", 0.1), "DOne"), (("E2", 0.1), "ETwoOne")]:
        c = pick[(g, t)]
        put(f"oc{name}Pi", f"{100 * c['pi']:.1f}", "{}"); put(f"oc{name}Draws", c["draws_per_answer"], "{:.1f}")
        put(f"oc{name}Cov", f"{100 * c['coverage_two_draws']:.0f}", "{}")
    c = next(c for c in lt["operating_curve"] if c["gate"].startswith("D or") and abs(c["tau"] - 0.1) < 1e-9)
    put("ocUnionPi", f"{100 * c['pi']:.1f}", "{}"); put("ocUnionPiLo", f"{100 * c['pi_lo']:.1f}", "{}")
    put("ocUnionPiHi", f"{100 * c['pi_hi']:.1f}", "{}"); put("ocUnionDraws", c["draws_per_answer"], "{:.1f}")
    for pol, name in [("restart_fb", "Dra"), ("care", "Care")]:     # each loop vs base, same items (exploratory)
        for g, gn in [("hall_E1", "EOne"), ("hall_E2", "ETwo")]:
            v, lo, hi, _ = lt["vs_base_b2"][pol][g]
            put(f"vb{name}{gn}", v, "{:+.3f}"); put(f"vb{name}{gn}Lo", lo, "{:+.3f}"); put(f"vb{name}{gn}Hi", hi, "{:+.3f}")

val_file = ROOT / "results" / "chokebench" / "validity.json"
if val_file.exists():                                               # stage 54: ChokeBench predictive validity
    val = json.loads(val_file.read_text())
    T, S = val["trace_level"], val["system_level"]
    for key, name in [(T["choke_only"]["hit_d"], "valD"), (T["human_only"]["hit_e"], "valE"),
                      (T["both"]["hit_d"], "valBothD"), (T["both"]["hit_e"], "valBothE"), (T["both"]["choke_minus_human"], "valDiff")]:
        put(name, key[0], "{:+.2f}"); put(name + "Lo", key[1], "{:+.2f}"); put(name + "Hi", key[2], "{:+.2f}")
    put("valRhoD", S["mean_rho_chokebench"], "{:+.2f}"); put("valRhoE", S["mean_rho_processbench"], "{:+.2f}")
    put("valRhoDiff", S["mean_rho_diff"][0], "{:+.2f}"); put("valRhoDiffLo", S["mean_rho_diff"][1], "{:+.2f}")
    put("valRhoDiffHi", S["mean_rho_diff"][2], "{:+.2f}"); put("valN", T["n_traces"], "{}")

rob_file = RF / "evaluator_robustness.json"
if rob_file.exists():
    rob = json.loads(rob_file.read_text())
    cal = rob["calibration"]
    put("robFstar", f"{100 * cal['f_star']:.0f}", "{}")
    put("robNDev", cal["n_dev"], "{}"); put("robNTest", cal["n_test"], "{}")
    kt = cal["test"]
    put("robKEOne", next(v for k, v in kt.items() if k.startswith("E1 (any")))
    put("robKEOneCal", next(v for k, v in kt.items() if k.startswith("E1-calibrated")))
    put("robKETwo", next(v for k, v in kt.items() if k.startswith("E2")))
    put("robKDual", next(v for k, v in kt.items() if k.startswith("dual")))
    put("robKD", next(v for k, v in kt.items() if k.startswith("D ")))
    put("robAgreeETwoD", pct(cal["agreement_with_D_on_all_drafts"]["E2"]), "{}")
    put("robAgreeEOneD", pct(cal["agreement_with_D_on_all_drafts"]["E1-calibrated"]), "{}")
    tab = rob["policies_test_split"]
    for key, name in [("base@0", "Base"), ("restart_fb@2", "Restart"), ("care@2", "Care"), ("care@restart@2", "CareR"),
                      ("abstain_flag@0", "Abst"), ("backtrack_fb@2", "Back"), ("cove_flag_clean@0", "Cove"),
                      ("delete@0", "Del")]:
        rec = tab[key]
        put(f"rob{name}Cov", pct(rec["coverage"]), "{}")
        put(f"rob{name}N", rec["n"], "{}")
        if rec["cost"] is not None:
            put(f"rob{name}Tok", rec["cost"], "{:,.0f}")
        for ev_, en in [("E1", "EOne"), ("E1cal", "EOneCal"), ("E2", "ETwo"), ("dual", "Dual"), ("D", "D")]:
            put(f"rob{name}{en}", pct(rec[ev_]), "{}")
            m, lo, hi = rec[ev_ + "_minus_restart"]
            if key != "restart_fb@2":
                put(f"rob{name}{en}Diff", m, "{:+.3f}"); put(f"rob{name}{en}DiffLo", lo, "{:+.3f}")
                put(f"rob{name}{en}DiffHi", hi, "{:+.3f}")

out = ROOT / "paper" / "values.tex"
out.write_text("% generated by paper/make_values.py -- do not edit\n"
               + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(f"wrote {len(macros)} macros -> {out}")
