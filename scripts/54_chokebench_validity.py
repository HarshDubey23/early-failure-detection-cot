"""Stage 54 (CPU): does a ChokeBench score predict real recovery gains? (predictive validity, exploratory)

A benchmark for step localisers matters only if scoring well on it predicts that resampling from the
named step actually helps. We test this without circularity: in each of 50 random splits of every
prefix's continuations, half A labels the decisive step d_A (largest drop in v) and half B scores
the recovery gain of resampling just before the named step j, g_B = v_B(j-1) - v_B(0). The two
halves are independent given the true value curve, so a hit on d_A cannot mechanically raise g_B.

  T  trace level, within trace: regress g_B on hit(j = d_A) and hit(j = human first error e), with
     trace fixed effects (difficulty removed); cluster bootstrap over traces.
  S  system level, held-out model: rank localisers by their ChokeBench score (accuracy on d_A) or
     their ProcessBench-style score (accuracy on e) on three generators, and correlate that ranking
     with their mean g_B on the fourth (Spearman), for each held-out generator.

    python scripts/54_chokebench_validity.py   ->  results/chokebench/validity.{json,md}
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s49", ROOT / "scripts" / "49_chokebench.py")
s49 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s49)
OUT = ROOT / "results" / "chokebench"
LOCS = [x for x in s49.LOCALISERS if x != "uniform_random"]     # each names one step


def collect(rng) -> pd.DataFrame:
    prm_all = s49.load_jsonl(s49.R / "signals_prm.jsonl")
    rows = []
    for gen, files in s49.GENS.items():
        sig_gen = s49.load_jsonl(s49.R / f"signals_gen.{gen}.jsonl")
        for tid, c in s49.s13.load_curves(files).items():
            if s49.s13.group_of(c["meta"]) != "fatal":
                continue
            n, e = len(c["correct"]), int(c["meta"]["first_error_idx"])
            prm = (prm_all.get(tid) or {}).get("prm") if prm_all.get(tid, {}).get("generator") == gen else None
            named = {L: s49.localise(L, n, e, prm, sig_gen.get(tid)) for L in LOCS}
            acc = {L: [] for L in named if named[L] is not None}
            for _ in range(s49.N_SPLIT):
                va, vb = [], []
                for x in c["correct"]:
                    p = rng.permutation(len(x))
                    va.append(x[p[: len(x) // 2]].mean())
                    vb.append(x[p[len(x) // 2:]].mean())
                va = np.array(va + [0.0])
                if va[0] <= s49.DELTA:                      # half A says capability-limited: not scored
                    continue
                dA = int(np.argmax(va[:-1] - va[1:])) + 1
                for L in acc:
                    j = named[L]
                    acc[L].append((float(j == dA), float(j == e), vb[j - 1] - vb[0]))
            for L, a in acc.items():
                if a:
                    a = np.array(a)
                    rows.append({"trace_id": tid, "generator": gen, "localiser": L, "w": len(a) / s49.N_SPLIT,
                                 "hit_d": a[:, 0].mean(), "hit_e": a[:, 1].mean(), "gain": a[:, 2].mean()})
    return pd.DataFrame(rows)


def within(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Weighted OLS of gain on cols after removing trace means (trace fixed effects)."""
    X = df[cols].to_numpy() - df.groupby("trace_id")[cols].transform("mean").to_numpy()
    y = df["gain"].to_numpy() - df.groupby("trace_id")["gain"].transform("mean").to_numpy()
    w = df["w"].to_numpy()
    return np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]


def trace_level(df: pd.DataFrame, rng) -> dict:
    ids = df["trace_id"].unique()
    groups = {t: g for t, g in df.groupby("trace_id")}
    specs = {"choke_only": ["hit_d"], "human_only": ["hit_e"], "both": ["hit_d", "hit_e"]}
    est = {k: within(df, v) for k, v in specs.items()}
    boots = {k: [] for k in specs}
    for _ in range(s49.N_BOOT):
        b = pd.concat([groups[t].assign(trace_id=f"{t}#{i}") for i, t in enumerate(rng.choice(ids, len(ids)))])
        for k, v in specs.items():
            boots[k].append(within(b, v))
    out = {}
    for k, v in specs.items():
        B = np.array(boots[k])
        out[k] = {c: [float(est[k][i]), float(np.quantile(B[:, i], 0.025)), float(np.quantile(B[:, i], 0.975))]
                  for i, c in enumerate(v)}
    B = np.array(boots["both"])
    diff = B[:, 0] - B[:, 1]
    out["both"]["choke_minus_human"] = [float(est["both"][0] - est["both"][1]),
                                        float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
    out["n_traces"], out["n_rows"] = int(len(ids)), int(len(df))
    return out


def system_level(df: pd.DataFrame) -> dict:
    def agg(d):
        return d.groupby("localiser").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["w"]) for c in ("hit_d", "hit_e", "gain")}),
                                            include_groups=False)
    out = {}
    for G in df["generator"].unique():
        train, test = agg(df[df.generator != G]), agg(df[df.generator == G])
        common = train.index.intersection(test.index)
        # the human-first-error localiser is excluded: its ProcessBench score is 1 by definition
        common = [L for L in common if L != "human_first_error"]
        out[G] = {"n_localisers": len(common),
                  "rho_chokebench": float(spearmanr(train.loc[common, "hit_d"], test.loc[common, "gain"])[0]),
                  "rho_processbench": float(spearmanr(train.loc[common, "hit_e"], test.loc[common, "gain"])[0]),
                  "localisers": {L: {"chokebench_other": float(train.loc[L, "hit_d"]), "processbench_other": float(train.loc[L, "hit_e"]),
                                     "gain_heldout": float(test.loc[L, "gain"])} for L in common}}
    out["mean_rho_chokebench"] = float(np.mean([v["rho_chokebench"] for v in out.values() if isinstance(v, dict)]))
    out["mean_rho_processbench"] = float(np.mean([v["rho_processbench"] for v in out.values() if isinstance(v, dict)]))
    return out


def system_boot(df: pd.DataFrame, rng, n_boot: int = 500) -> list:
    """Trace bootstrap (within generator) of mean rho(ChokeBench) - mean rho(ProcessBench-style)."""
    groups = {t: g for t, g in df.groupby("trace_id")}
    by_gen = df.groupby("generator")["trace_id"].unique()
    diffs = []
    for _ in range(n_boot):
        b = pd.concat([groups[t].assign(trace_id=f"{t}#{i}") for ids in by_gen for i, t in enumerate(rng.choice(ids, len(ids)))])
        S = system_level(b)
        diffs.append(S["mean_rho_chokebench"] - S["mean_rho_processbench"])
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def main() -> None:
    rng = np.random.default_rng(54)
    df = collect(rng)
    S = system_level(df)
    S["mean_rho_diff"] = [S["mean_rho_chokebench"] - S["mean_rho_processbench"]] + system_boot(df, rng)
    res = {"trace_level": trace_level(df, rng), "system_level": S,
           "note": "exploratory; localiser-level ChokeBench accuracies and gains (stage 49) had been seen before this analysis"}
    (OUT / "validity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    T, S = res["trace_level"], res["system_level"]
    f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"  # noqa: E731
    lines = ["# ChokeBench predictive validity (stage 54, exploratory)", "",
             f"{T['n_traces']} open failed traces, {T['n_rows']} (trace, localiser) pairs. Half A of each prefix's continuations "
             "labels the decisive step, half B scores the gain of resampling from the named step (50 splits); trace fixed effects.", "",
             "| Model | gain per attempt when the named step is the decisive step | ... when it is the human first error |", "|---|---|---|",
             f"| decisive step only | {f(T['choke_only']['hit_d'])} | |",
             f"| human first error only | | {f(T['human_only']['hit_e'])} |",
             f"| both | {f(T['both']['hit_d'])} | {f(T['both']['hit_e'])} |", "",
             f"Difference (decisive minus human, joint model): {f(T['both']['choke_minus_human'])}.", "",
             "## Held-out model: do benchmark scores on three generators rank localisers by gain on the fourth?", "",
             "| Held-out generator | localisers | Spearman, ChokeBench score | Spearman, ProcessBench-style score |", "|---|---|---|---|"]
    for G, v in S.items():
        if isinstance(v, dict):
            lines.append(f"| {G} | {v['n_localisers']} | {v['rho_chokebench']:+.2f} | {v['rho_processbench']:+.2f} |")
    lines += [f"| **mean** | | **{S['mean_rho_chokebench']:+.2f}** | **{S['mean_rho_processbench']:+.2f}** |", "",
              f"Mean difference {S['mean_rho_diff'][0]:+.2f} [{S['mean_rho_diff'][1]:+.2f}, {S['mean_rho_diff'][2]:+.2f}] "
              "(trace bootstrap within generator, 500 resamples). Seven localisers per generator, so each single rho is noisy.", "",
              "Reading: the coefficients compare named steps within a trace, not a named step with restart. Naming the "
              "decisive step avoids most of the loss other steps incur; it does not by itself beat restart (stage 49).", "",
              res["note"] + "."]
    (OUT / "validity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
