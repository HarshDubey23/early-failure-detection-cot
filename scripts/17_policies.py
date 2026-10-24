"""Stage 17 (CPU): practical backtracking policies against restart at a matched end-to-end budget.

Implements section 5 of docs/PREREGISTRATION_RUN_C.md. For each fatal trace a policy picks a
restart point k; it then spends whatever is left of the budget on independent attempts from k.

  cost of one restart attempt   C0 = prefill(question) + mean generated tokens from k = 0
  budget                        B  = b * C0, b in {1, 2, 4, 8}
  policy cost                   overhead (verifier passes, probes, oracle rollouts)
                                + prefill(question + first k steps), paid once (prefix cache)
                                + N * mean generated tokens from k
  attempts                      N = floor((B - overhead - prefill) / generated tokens from k)
  success                       P(at least one of N attempts correct), Jeffreys-posterior mean of
                                1 - (1 - p)^N from the half-B continuations of prefix k

All tokens are weighted by model size relative to the generator (FLOPs ~ parameters x tokens),
so a 7B verifier pass over a 1.5B model's trace costs ~5x its token count. Decode-only
accounting (prefill weight 0) is reported as a secondary view. Halves A/B are redrawn in each of
`--n-splits` splits: half A feeds probes, oracle and the learned policy's labels, half B scores.

Policies: P0 restart; P1 PRM first error; P2 PRM largest drop; P3 probe search; P4 learned
(logistic regression on step features, trained on other generators only); P5 human first error
(reference); P6 oracle decisive step from half A (all its rollouts counted).
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
s13 = import_module("13_commitment_analysis")

PARAMS_B = {"Qwen2.5-1.5B-Instruct": 1.54, "Qwen2-1.5B-Instruct": 1.54, "Qwen2.5-Math-7B-Instruct": 7.62,
            "Llama-3.1-8B-Instruct": 8.03, "PRM": 7.62}
POLICIES = ["P0_restart", "P1_prm_first_error", "P2_prm_largest_drop", "P3_probe_search", "P4_learned",
            "P5_human_first_error", "P6_oracle"]
FEATURES = ["prm", "prm_drop", "nll_mean", "nll_max", "ent_mean", "ent_max", "rel_pos", "step_tokens"]


def jeffreys_pass(c: float, n: float, N: np.ndarray) -> np.ndarray:
    """E[1 - (1 - p)^N] for p ~ Beta(c + 1/2, n - c + 1/2); 0 when N < 1."""
    a, b = c + 0.5, n - c + 0.5
    N = np.asarray(N, dtype=float)
    out = 1 - np.exp(betaln(a, b + np.maximum(N, 0)) - betaln(a, b))
    return np.where(N >= 1, out, 0.0)


def load_signals(path_gen: list, path_prm: list) -> tuple:
    gen, prm = {}, {}
    for p in path_gen:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("skipped"):
                gen[r["trace_id"]] = r
    for p in path_prm:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("skipped") and r.get("ok", True):
                prm[r["trace_id"]] = r
    return gen, prm


def step_features(t_gen: dict | None, t_prm: dict | None, n: int) -> np.ndarray:
    """(n, len(FEATURES)) matrix for steps 1..n; missing signals become NaN."""
    X = np.full((n, len(FEATURES)), np.nan)
    if t_prm is not None:
        p = np.asarray(t_prm["prm"], dtype=float)
        X[:, 0] = p
        X[:, 1] = np.concatenate([[1.0], p[:-1]]) - p
    if t_gen is not None:
        for j, f in enumerate(["nll_mean", "nll_max", "ent_mean", "ent_max"], start=2):
            X[:, j] = t_gen[f]
        X[:, 7] = t_gen["step_tokens"]
    X[:, 6] = np.arange(1, n + 1) / n
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="17: policies at a matched budget")
    ap.add_argument("--gens", nargs="+", required=True, help="generator names, matched to --rollout-glob")
    ap.add_argument("--rollouts", nargs="+", required=True, help="files; generator taken from each row")
    ap.add_argument("--signals-gen", nargs="*", default=[])
    ap.add_argument("--signals-prm", nargs="*", default=[])
    ap.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--delta", type=float, default=0.125)
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    curves = s13.load_curves([Path(p) for p in args.rollouts])
    sig_gen, sig_prm = load_signals(args.signals_gen, args.signals_prm)
    text = {}
    for line in open(args.traces, encoding="utf-8"):
        r = json.loads(line)
        if r["trace_id"] in curves:
            text[r["trace_id"]] = r
    fatal = {t: c for t, c in curves.items() if s13.group_of(c["meta"]) == "fatal" and c["meta"]["generator"] in args.gens}

    # token accounting: exact counts from stage 14 when present, else ~3.5 characters per token
    def tokens(tid: str) -> tuple:
        g = sig_gen.get(tid)
        if g is not None:
            return float(g["prompt_tokens"]), np.asarray(g["step_tokens"], dtype=float), True
        tr = text[tid]
        return (len(tr["question"]) / 3.5 + 30, np.array([len(s) / 3.5 for s in tr["steps"]]), False)

    # ---- per split: half-B counts, half-A decisive step, half-A probes ----------------------
    split_data = {tid: [] for tid in fatal}
    for tid, c in fatal.items():
        m = len(c["correct"][0])
        for _ in range(args.n_splits):
            perm = rng.permutation(m)
            a_idx, b_idx = np.sort(perm[: m // 2]), np.sort(perm[m // 2:])
            v_a = np.array([x[a_idx].mean() for x in c["correct"]] + [1.0 if c["observed"] else 0.0])
            split_data[tid].append({
                "cB": np.array([x[b_idx].sum() for x in c["correct"]], dtype=float), "nB": float(len(b_idx)),
                "v0A": v_a[0], "dA": int(np.argmax(v_a[:-1] - v_a[1:])) + 1,
                "probe": np.array([x[a_idx[:2]].sum() for x in c["correct"]], dtype=float),
                "mA": float(len(a_idx))})

    # ---- learned policy (P4): one model per held-out generator, trained on the others ---------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feats = {tid: step_features(sig_gen.get(tid), sig_prm.get(tid), len(c["correct"])) for tid, c in fatal.items()}
    have_signals = any(tid in sig_prm for tid in fatal)

    def train_rows(gens_train: list, s: int) -> tuple:
        X, y = [], []
        for tid, c in fatal.items():
            if c["meta"]["generator"] not in gens_train:
                continue
            sd = split_data[tid][s]
            if sd["v0A"] <= args.delta:
                continue
            F = feats[tid]
            for j in range(F.shape[0]):
                X.append(F[j])
                y.append(1 if j + 1 == sd["dA"] else 0)
        return np.array(X), np.array(y)

    learned_target: dict = {}
    if have_signals:
        for held in args.gens:
            others = [g for g in args.gens if g != held and g != "Llama-3.1-8B-Instruct"] if held != "Llama-3.1-8B-Instruct" \
                else [g for g in args.gens if g != held]
            if not others:
                continue
            for s in range(args.n_splits):
                X, y = train_rows(others, s)
                ok = ~np.isnan(X).any(axis=1)
                if ok.sum() < 20 or y[ok].sum() < 3:
                    continue
                sc = StandardScaler().fit(X[ok])
                clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[ok]), y[ok])
                for tid, c in fatal.items():
                    if c["meta"]["generator"] != held:
                        continue
                    F = feats[tid]
                    if np.isnan(F).any():
                        continue
                    learned_target[(tid, s)] = int(np.argmax(clf.predict_proba(sc.transform(F))[:, 1])) + 1

    # ---- evaluate ---------------------------------------------------------------------------
    rows = []
    for tid, c in fatal.items():
        gen = c["meta"]["generator"]
        size = PARAMS_B.get(gen, 1.0)
        P, st, exact = tokens(tid)
        n = len(st)
        L = P + np.concatenate([[0.0], np.cumsum(st)])[:n]    # prefill tokens of question + first k steps, k = 0..n-1
        g = np.asarray(c["tokens"][:n], dtype=float)            # mean generated tokens from prefix k
        trace_tokens = P + st.sum()
        prm_pass = trace_tokens * PARAMS_B["PRM"] / size
        e = int(c["meta"]["first_error_idx"])
        prm = sig_prm.get(tid)
        if prm is not None:
            p = np.asarray(prm["prm"], dtype=float)
            below = np.nonzero(p < 0.5)[0]
            k_p1 = int(below[0]) if len(below) else 0               # step j (1-based) = below+1 -> k = j - 1
            drops = np.concatenate([[1.0], p[:-1]]) - p
            k_p2 = int(np.argmax(drops))
        probes = sorted({max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4)} & set(range(n)))
        for s, sd in enumerate(split_data[tid]):
            # (restart point k, overhead prefill tokens, overhead generated tokens)
            choice = {"P0_restart": (0, 0.0, 0.0), "P5_human_first_error": (e - 1, 0.0, 0.0),
                      "P6_oracle": (sd["dA"] - 1, float(np.sum(L)), float(sd["mA"] * np.sum(g)))}
            if prm is not None:
                choice["P1_prm_first_error"] = (k_p1, prm_pass, 0.0)
                choice["P2_prm_largest_drop"] = (k_p2, prm_pass, 0.0)
            if (tid, s) in learned_target:
                choice["P4_learned"] = (learned_target[(tid, s)] - 1, prm_pass + trace_tokens, 0.0)
            ok = [k for k in probes if sd["probe"][k] > 0]
            choice["P3_probe_search"] = (max(ok) if ok else 0, float(sum(L[k] for k in probes)),
                                         float(sum(2 * g[k] for k in probes)))
            for pol, (k, oh_pre, oh_gen) in choice.items():
                for b in args.budgets:
                    for acct, wpf in [("flops", 1.0), ("decode_only", 0.0)]:
                        C0 = wpf * P + g[0]
                        B = b * C0
                        oh = wpf * oh_pre + oh_gen  # decode-only view drops every prefill, including verifier passes
                        N = np.floor((B - oh - wpf * L[k]) / max(g[k], 1.0))
                        succ = float(jeffreys_pass(sd["cB"][k], sd["nB"], N))
                        rows.append({"trace_id": tid, "generator": gen, "split": s, "policy": pol, "budget": b,
                                     "accounting": acct, "k": k, "attempts": float(max(N, 0)), "success": succ,
                                     "not_limited": bool(sd["v0A"] > args.delta), "exact_tokens": exact})
    df = pd.DataFrame(rows)
    per_trace = (df.groupby(["generator", "trace_id", "policy", "budget", "accounting"])
                 .agg(success=("success", "mean"), attempts=("attempts", "mean"), not_limited=("not_limited", "mean"),
                      exact_tokens=("exact_tokens", "first")).reset_index())
    per_trace.to_csv(out / "policy_per_trace.csv", index=False)

    boot_rng = np.random.default_rng(args.seed + 1)
    summ = []
    for (gen, b, acct), sub in per_trace.groupby(["generator", "budget", "accounting"]):
        for pop, popsub in [("all_fatal", sub), ("not_limited", sub[sub.not_limited >= 0.5])]:
            wide = popsub.pivot_table(index="trace_id", columns="policy", values="success")
            if "P0_restart" not in wide:
                continue
            idx = [boot_rng.integers(0, len(wide), len(wide)) for _ in range(2000)]
            for pol in wide.columns:
                x = wide[pol].to_numpy()
                d = (wide[pol] - wide["P0_restart"]).to_numpy()
                okm = ~np.isnan(d)
                xb = [np.nanmean(x[i]) for i in idx]
                dbs = [np.nanmean(d[i]) for i in idx]
                summ.append({"generator": gen, "budget": b, "accounting": acct, "population": pop, "policy": pol,
                             "n": int(okm.sum()), "success": float(np.nanmean(x)),
                             "success_lo": float(np.nanquantile(xb, 0.025)), "success_hi": float(np.nanquantile(xb, 0.975)),
                             "minus_restart": float(np.nanmean(d)), "diff_lo": float(np.nanquantile(dbs, 0.025)),
                             "diff_hi": float(np.nanquantile(dbs, 0.975)),
                             "attempts": float(popsub[popsub.policy == pol].attempts.mean())})
    S = pd.DataFrame(summ)
    S.to_csv(out / "policy_summary.csv", index=False)
    show = S[(S.accounting == "flops") & (S.population == "all_fatal") & (S.budget == 4)]
    print(show[["generator", "policy", "n", "success", "minus_restart", "diff_lo", "diff_hi", "attempts"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
