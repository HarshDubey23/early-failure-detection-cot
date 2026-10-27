"""Stage 49 (CPU): ChokeBench, a counterfactual benchmark for step-level failure localisation.

ProcessBench grades a verifier on finding the *human-labelled first error*. This script grades the
same signals on a second target that the committed rollouts make measurable: the *decisive step*,
the step where the generator's own chance of success, v_k, falls most. It reuses the committed
on-policy rollouts (results/commitment, results/run_c) and step signals (PRM, generator NLL and
entropy). No GPU, no new model calls.

For every not-capability-limited fatal trace (v_0 > 1/8) and every localiser (a rule that names one
step), it reports
  - acc_human:      the named step is the human first error (the ProcessBench target);
  - acc_decisive:   posterior probability that the named step is the decisive step (Jeffreys
                    posterior on every v_k, 2,000 draws), so sampling noise is not counted as error;
  - gain_vs_restart: success of one attempt resampled from just before the named step, minus one
                    restart attempt (v_{j-1} - v_0). The localisers never look at the rollouts, so
                    scoring them on all continuations is unbiased. The oracle row is sample-split.
  - token_ratio:    generated tokens per attempt from that point, relative to restart.
  - gain_b4_free:   P(solved) at a budget of four restart attempts (FLOP-matched as in stage 17:
                    prefill counted once per resampling point, every generated token counted),
                    minus restart, if finding the step cost nothing. An upper bound on what any
                    verifier using that localiser can gain; stage 17 charges the verifier.

Output: results/chokebench/{per_trace.csv, summary.csv, summary.md, chokebench_v1.jsonl}
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "results" / "commitment"
R = ROOT / "results" / "run_c"
OUT = ROOT / "results" / "chokebench"
DELTA = 0.125
N_POST = 2000
N_SPLIT = 50
N_BOOT = 2000
BUDGET = 4.0


def jeffreys_pass(c: float, n: float, N: float) -> float:
    """E[1 - (1 - p)^N], p ~ Beta(c + 1/2, n - c + 1/2) (same estimator as stage 17)."""
    if N < 1:
        return 0.0
    a, b = c + 0.5, n - c + 0.5
    return float(1 - np.exp(betaln(a, b + N) - betaln(a, b)))

spec = importlib.util.spec_from_file_location("s13", ROOT / "scripts" / "13_commitment_analysis.py")
s13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s13)

GENS = {
    "Qwen2.5-1.5B-Instruct": sorted(C.glob("Qwen2.5-1.5B-Instruct_s*.jsonl")) + sorted(R.glob("rollouts.Qwen2.5-1.5B-Instruct.b*.jsonl")),
    "Qwen2-1.5B-Instruct": sorted(C.glob("Qwen2-1.5B-Instruct_s*.jsonl")) + sorted(R.glob("rollouts.Qwen2-1.5B-Instruct.b*.jsonl")),
    "Qwen2.5-Math-7B-Instruct": sorted(C.glob("Qwen2.5-Math-7B-Instruct_s*.jsonl")),
    "Llama-3.1-8B-Instruct": sorted(R.glob("rollouts.Llama-3.1-8B-Instruct.*.jsonl")),
}
LOCALISERS = ["human_first_error", "prm_first_below_0.5", "prm_min", "prm_largest_drop",
              "gen_max_token_nll", "gen_max_token_entropy", "first_step", "last_step", "uniform_random"]


def load_jsonl(path: Path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.setdefault(r["trace_id"], r)
    return out


def localise(name: str, n: int, e: int, prm, gen) -> int | None:
    """1-based step named by a localiser, or None if it names no step (or lacks its signal)."""
    if name == "human_first_error":
        return e
    if name == "first_step":
        return 1
    if name == "last_step":
        return n
    if name.startswith("prm"):
        if prm is None or len(prm) != n:
            return None
        p = np.asarray(prm, dtype=float)
        if name == "prm_first_below_0.5":
            below = np.nonzero(p < 0.5)[0]
            return int(below[0]) + 1 if len(below) else None
        if name == "prm_min":
            return int(np.argmin(p)) + 1
        return int(np.argmax(np.concatenate([[1.0], p[:-1]]) - p)) + 1   # same rule as policy P2
    if name.startswith("gen"):
        if gen is None or len(gen["nll_max"]) != n:
            return None
        key = "nll_max" if name == "gen_max_token_nll" else "ent_max"
        return int(np.argmax(gen[key])) + 1
    raise ValueError(name)


FEATURE_SETS = {
    "prm": ["prm", "prm_drop", "prm_first_below", "rel_pos"],
    "generator": ["nll_mean", "nll_max", "ent_mean", "ent_max", "log_step_tokens", "rel_pos"],
    "all": ["prm", "prm_drop", "prm_first_below", "nll_mean", "nll_max", "ent_mean", "ent_max", "log_step_tokens", "rel_pos"],
}
ALL_FEATURES = FEATURE_SETS["all"]
GRID = ([{"model": "logreg", "C": C} for C in (0.01, 0.1, 1.0, 10.0)]
        + [{"model": "gboost", "max_depth": dpt, "learning_rate": lr} for dpt in (2, 3) for lr in (0.05, 0.2)])


def step_features(prm, gen, n: int) -> np.ndarray | None:
    """(n, len(ALL_FEATURES)) step features, or None if a signal is missing (same signals as stage 17)."""
    if prm is None or gen is None or len(prm) != n or len(gen["nll_max"]) != n:
        return None
    p = np.asarray(prm, dtype=float)
    first = np.zeros(n)
    below = np.nonzero(p < 0.5)[0]
    if len(below):
        first[below[0]] = 1.0
    cols = {"prm": p, "prm_drop": np.concatenate([[1.0], p[:-1]]) - p, "prm_first_below": first,
            "nll_mean": np.asarray(gen["nll_mean"], float), "nll_max": np.asarray(gen["nll_max"], float),
            "ent_mean": np.asarray(gen["ent_mean"], float), "ent_max": np.asarray(gen["ent_max"], float),
            "log_step_tokens": np.log1p(np.asarray(gen["step_tokens"], float)), "rel_pos": np.arange(1, n + 1) / n}
    return np.column_stack([cols[c] for c in ALL_FEATURES])


def _fit(traces: list, feats: list, hp: dict):
    """Step-level classifier of 'this step is the choke point', trained on posterior soft labels.

    Each step appears twice (label 1 with weight P(d=j), label 0 with weight 1-P(d=j)); weights are
    divided by n so every trace counts once, and multiplied by the trace's split-selection share w.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = [ALL_FEATURES.index(f) for f in feats]
    X = np.vstack([t["F"][:, idx] for t in traces])
    p = np.concatenate([t["post"] for t in traces])
    tw = np.concatenate([np.full(len(t["post"]), t["w"] / len(t["post"])) for t in traces])
    X2, y2, w2 = np.vstack([X, X]), np.r_[np.ones(len(p)), np.zeros(len(p))], np.r_[p * tw, (1 - p) * tw]
    if hp["model"] == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=2000))
        return clf.fit(X2, y2, logisticregression__sample_weight=w2), idx
    clf = HistGradientBoostingClassifier(max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
                                         max_iter=200, random_state=0)
    return clf.fit(X2, y2, sample_weight=w2), idx


def _pick(model, idx, t) -> int:
    """0-based step index the model names for trace t."""
    return int(np.argmax(model.predict_proba(t["F"][:, idx])[:, 1]))


def _score(traces: list, model, idx) -> float:
    """Weighted mean posterior probability that the named step is the choke point."""
    acc = np.array([t["post"][_pick(model, idx, t)] for t in traces])
    w = np.array([t["w"] for t in traces])
    return float((acc * w).sum() / w.sum())


def learned_localiser(learn: list) -> list:
    """Leave-one-generator-out training with nested hyperparameter selection.

    Outer loop: hold out one generator. Inner loop: on the other generators only, leave each one out
    in turn, fit every (feature set, model, hyperparameter) configuration and score it on the
    left-out generator; pick the configuration with the best mean inner score; refit it on all
    training generators; name one step per held-out trace. The held-out generator never influences
    training or selection. Writes results/chokebench/learned_tuning.json.
    """
    gens = sorted({t["base"]["generator"] for t in learn})
    configs = [dict(hp, features=fs) for fs in FEATURE_SETS for hp in GRID]
    rows, report = [], {"grid": configs, "outer": {}}
    for held in gens:
        train = [t for t in learn if t["base"]["generator"] != held]
        inner_gens = sorted({t["base"]["generator"] for t in train})
        scores = []
        for cfg in configs:
            hp = {k: v for k, v in cfg.items() if k != "features"}
            fold = []
            for ig in inner_gens:
                tr = [t for t in train if t["base"]["generator"] != ig]
                va = [t for t in train if t["base"]["generator"] == ig]
                model, idx = _fit(tr, FEATURE_SETS[cfg["features"]], hp)
                fold.append(_score(va, model, idx))
            scores.append(float(np.mean(fold)))
        best = configs[int(np.argmax(scores))]
        hp = {k: v for k, v in best.items() if k != "features"}
        model, idx = _fit(train, FEATURE_SETS[best["features"]], hp)
        test = [t for t in learn if t["base"]["generator"] == held]
        report["outer"][held] = {"inner_scores": scores, "selected": best,
                                 "test_acc_decisive": _score(test, model, idx), "n_test": len(test)}
        for t in test:
            k = _pick(model, idx, t)
            j = k + 1
            rows.append(dict(t["base"], localiser="learned_localiser", j=j, named=True,
                             acc_human=float(j == t["e"]), acc_decisive=float(t["post"][k]),
                             hit_decisive=float(j == t["d"]), gain=float(t["gain"][k]),
                             gain_b4_free=float(t["gain_b4"][k]), token_ratio=float(t["tok_ratio"][k])))
    (OUT / "learned_tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(49)
    prm_all = load_jsonl(R / "signals_prm.jsonl")
    rows, release, learn = [], [], []
    for gen, files in GENS.items():
        sig_gen = load_jsonl(R / f"signals_gen.{gen}.jsonl")
        curves = s13.load_curves(files)
        for tid, c in curves.items():
            if s13.group_of(c["meta"]) != "fatal":
                continue
            n = len(c["correct"])
            e = int(c["meta"]["first_error_idx"])
            cnt = np.array([x.sum() for x in c["correct"]], dtype=float)
            m = np.array([len(x) for x in c["correct"]], dtype=float)
            v = np.concatenate([cnt / m, [0.0]])                       # v_0..v_{n-1}, v_n = 0 (fatal)
            prm = (prm_all.get(tid) or {}).get("prm") if prm_all.get(tid, {}).get("generator") == gen else None
            g = sig_gen.get(tid)
            release.append({"trace_id": tid, "generator": gen, "n_steps": n, "human_first_error": e,
                            "successes": cnt.astype(int).tolist(), "continuations": m.astype(int).tolist(),
                            "mean_continuation_tokens": [round(float(t), 1) for t in c["tokens"][:n]],
                            "prm": prm, "capability_limited": bool(v[0] <= DELTA)})
            # Sample splitting, as in the paper (Sec. 5.4): half A decides whether a trace is "open"
            # (v_0^A > delta) and, for the oracle, picks the decisive step; half B scores every
            # resampling point. Selecting and scoring on the same continuations would make restart
            # look better than it is (v_0 is selected to be high).
            vA0, vB, cB, nB, dA = [], [], [], [], []
            for _ in range(N_SPLIT):
                ia, ib, cb, nb = [], [], [], []
                for x in c["correct"]:
                    perm = rng.permutation(len(x))
                    ia.append(x[perm[: len(x) // 2]].mean())
                    hb = x[perm[len(x) // 2:]]
                    ib.append(hb.mean())
                    cb.append(hb.sum())
                    nb.append(len(hb))
                va = np.array(ia + [0.0])
                vA0.append(va[0])
                vB.append(np.array(ib))
                cB.append(np.array(cb, dtype=float))
                nB.append(np.array(nb, dtype=float))
                dA.append(int(np.argmax(va[:-1] - va[1:])) + 1)
            keep = np.array(vA0) > DELTA
            w = float(keep.mean())                                      # share of splits that include the trace
            if w == 0.0:
                continue
            vB = np.array(vB)[keep]                                     # (splits kept, n)
            cB, nB = np.array(cB)[keep], np.array(nB)[keep]
            dA = np.array(dA)[keep]
            tok = np.asarray(c["tokens"][:n], dtype=float)
            if g is not None and len(g["step_tokens"]) == n:            # exact token counts from stage 14
                P0 = float(g["prompt_tokens"])
                L = P0 + np.concatenate([[0.0], np.cumsum(g["step_tokens"])])[:n]
                B = BUDGET * (P0 + tok[0])
                N_att = np.floor((B - L) / np.maximum(tok, 1.0))         # attempts affordable from each point
            else:
                N_att = None

            def gain_at(k: int) -> float:
                return float(np.mean(vB[:, k] - vB[:, 0]))

            def gain_b4(ks) -> float:
                """Matched-budget gain over restart; ks = resampling point per kept split."""
                if N_att is None:
                    return np.nan
                ks = np.broadcast_to(np.asarray(ks), (len(cB),))
                return float(np.mean([jeffreys_pass(cB[s][k], nB[s][k], N_att[k]) - jeffreys_pass(cB[s][0], nB[s][0], N_att[0])
                                      for s, k in enumerate(ks)]))

            d = int(np.argmax(v[:-1] - v[1:])) + 1
            draws = rng.beta(cnt[:, None] + 0.5, m[:, None] - cnt[:, None] + 0.5, size=(n, N_POST))
            draws = np.vstack([draws, np.zeros((1, N_POST))])
            post = np.bincount(np.argmax(draws[:-1] - draws[1:], axis=0), minlength=n) / N_POST  # P(d = j), j = 1..n
            base = {"trace_id": tid, "generator": gen, "n": n, "e": e, "d": d, "v0": v[0], "w": w,
                    "post_d_eq_e": post[e - 1]}
            for name in LOCALISERS:
                if name == "uniform_random":
                    rows.append(dict(base, localiser=name, j=np.nan, named=True, acc_human=1.0 / n,
                                     acc_decisive=1.0 / n, hit_decisive=1.0 / n,
                                     gain=float(np.mean([gain_at(k) for k in range(n)])),
                                     gain_b4_free=float(np.mean([gain_b4(k) for k in range(n)])),
                                     token_ratio=float(np.mean(tok) / max(tok[0], 1.0))))
                    continue
                j = localise(name, n, e, prm, g)
                if j is None and name.startswith(("prm", "gen")) and (prm is None if name.startswith("prm") else g is None):
                    continue                                            # signal missing for this trace
                named = j is not None
                k = (j - 1) if named else 0                             # no step named -> plain restart
                rows.append(dict(base, localiser=name, j=j if named else np.nan, named=named,
                                 acc_human=float(named and j == e), acc_decisive=float(post[j - 1]) if named else 0.0,
                                 hit_decisive=float(named and j == d), gain=gain_at(k), gain_b4_free=gain_b4(k),
                                 token_ratio=float(tok[k] / max(tok[0], 1.0))))
            F = step_features(prm, g, n)
            if F is not None:
                learn.append({"base": base, "F": F, "post": post, "e": e, "d": d, "w": w,
                              "gain": np.array([gain_at(k) for k in range(n)]),
                              "gain_b4": np.array([gain_b4(k) for k in range(n)]),
                              "tok_ratio": tok / max(tok[0], 1.0)})
            rows.append(dict(base, localiser="oracle_decisive_split", j=np.nan, named=True, acc_human=np.nan,
                             acc_decisive=np.nan, hit_decisive=np.nan,
                             gain=float(np.mean(vB[np.arange(len(dA)), dA - 1] - vB[:, 0])),
                             gain_b4_free=gain_b4(dA - 1), token_ratio=float(np.mean(tok[dA - 1]) / max(tok[0], 1.0))))

    rows += learned_localiser(learn)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_trace.csv", index=False)
    with open(OUT / "chokebench_v1.jsonl", "w", encoding="utf-8") as f:
        for r in release:
            f.write(json.dumps(r) + "\n")

    def boot(x: np.ndarray, w: np.ndarray) -> tuple:
        """Weighted mean (weight = share of splits that select the trace) and trace bootstrap CI."""
        ok = ~np.isnan(x)
        x, w = x[ok], w[ok]
        if len(x) == 0:
            return (np.nan, np.nan, np.nan)
        idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
        bs = (x[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
        return float((x * w).sum() / w.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    summ = []
    for (gen, loc), sub in list(df.groupby(["generator", "localiser"])) + [(("pooled", loc), df[df.localiser == loc])
                                                                         for loc in df.localiser.unique()]:
        rec = {"generator": gen, "localiser": loc, "n_traces": len(sub), "effective_n": float(sub.w.sum()),
               "names_a_step": float(sub.named.mean())}
        for col in ["acc_human", "acc_decisive", "gain", "gain_b4_free", "token_ratio"]:
            mu, lo, hi = boot(sub[col].to_numpy(dtype=float), sub.w.to_numpy(dtype=float))
            rec[col], rec[col + "_lo"], rec[col + "_hi"] = mu, lo, hi
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)

    order = LOCALISERS + ["learned_localiser", "oracle_decisive_split"]
    pooled = S[S.generator == "pooled"].set_index("localiser").reindex(order)
    lines = ["# ChokeBench v1: localising where failures are decided",
             "",
             f"Not-capability-limited fatal traces (v_0 > 1/8), four generators, pooled n = "
             f"{int(pooled.loc['first_step', 'n_traces'])} (fewer where a signal is missing). Traces are weighted by the share of 50 sample splits whose half A finds them open; half B scores every resampling point. "
             "95% trace-bootstrap CIs. Generated by `scripts/49_chokebench.py`.",
             "",
             "| Localiser | finds human first error | finds decisive step (posterior) | gain vs restart, per attempt | gain vs restart at 4x budget, localisation free | tokens per attempt vs restart |",
             "|---|---|---|---|---|---|"]
    for loc, r in pooled.iterrows():
        f = lambda c: "–" if np.isnan(r[c]) else f"{r[c]:.3f} [{r[c+'_lo']:.3f}, {r[c+'_hi']:.3f}]"  # noqa: E731
        g = lambda c: "–" if np.isnan(r[c]) else f"{r[c]:+.3f} [{r[c+'_lo']:+.3f}, {r[c+'_hi']:+.3f}]"  # noqa: E731
        lines.append(f"| {loc} | {f('acc_human')} | {f('acc_decisive')} | {g('gain')} | {g('gain_b4_free')} | "
                     f"{'–' if np.isnan(r['token_ratio']) else format(r['token_ratio'], '.2f')} |")
    lines += ["", "## Per generator (finds decisive step / gain vs restart)", "",
              "| Localiser | " + " | ".join(GENS) + " |", "|---|" + "---|" * len(GENS)]
    for loc in order:
        cells = []
        for gen in GENS:
            r = S[(S.generator == gen) & (S.localiser == loc)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            a = "–" if np.isnan(r.acc_decisive) else f"{r.acc_decisive:.2f}"
            cells.append(f"{a} / {r.gain:+.3f} (n={int(r.n_traces)})")
        lines.append(f"| {loc} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
