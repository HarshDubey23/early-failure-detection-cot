"""Stage 13 (CPU): from counterfactual rollouts to choke points.

Reads the JSONL written by stage 12 and answers, per generator:
  1. Where do failing trajectories commit, relative to the annotated first error?
     (capability-limited / latent commitment / at the error / recoverable error)
  2. Is the annotated first error the largest drop in success probability?
     Compared against the drop noise floor measured on error-free traces.
  3. Do "recovered" traces (error, correct answer) actually lose success
     probability at their first error?
  4. Where should a verifier send the model back to? Success of resampling from
     before the first error vs before the decisive step vs restarting, with
     sample splitting so the decisive step is chosen and evaluated on
     different continuations (no winner's curse).

Usage:
  python scripts/13_commitment_analysis.py --rollouts results/commitment/qwen25_math_7b.jsonl \
      --out results/commitment/tables_qwen25_math_7b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import commitment as cm  # noqa: E402


def load_curves(paths: list) -> dict:
    """{trace_id: dict(meta, rows sorted by k)}; appends the observed outcome as k = n.

    Accepts several JSONL files (e.g. one per GPU shard); duplicate (trace, k) rows keep the first.
    """
    by_trace: dict = defaultdict(list)
    seen = set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if (row["trace_id"], row["k"]) in seen:
                continue
            seen.add((row["trace_id"], row["k"]))
            by_trace[row["trace_id"]].append(row)
    curves = {}
    for tid, rows in by_trace.items():
        rows = sorted(rows, key=lambda r: r["k"])
        n = rows[0]["n_steps"]
        ks = [r["k"] for r in rows]
        if ks != list(range(n)):  # incomplete trace (interrupted run): skip
            continue
        correct = [np.asarray(r["correct"], dtype=bool) for r in rows]
        tokens = np.array([np.mean(r["new_tokens"]) for r in rows] + [0.0])
        observed = bool(rows[0]["final_correct"])
        curves[tid] = {"meta": rows[0], "correct": correct, "observed": observed, "tokens": tokens,
                       "backend": rows[0].get("backend", "?")}
    return curves


def v_from(correct: list, observed: bool, idx: np.ndarray | None = None) -> np.ndarray:
    """v_0..v_{n-1} from rollouts (optionally a subset of samples), v_n = observed outcome."""
    vals = [float(np.mean(c if idx is None else c[idx[idx < len(c)]])) for c in correct]
    return np.array(vals + [1.0 if observed else 0.0])


def group_of(meta: dict) -> str:
    e = meta["first_error_idx"]
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return "clean"
    return "recovered" if meta["final_correct"] else "fatal"


def bootstrap_share(flags: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(n_boot)]
    return float(flags.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="13: commitment analysis")
    ap.add_argument("--rollouts", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.0625, 0.125, 0.25])
    ap.add_argument("--primary-delta", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    curves = load_curves([Path(p) for p in args.rollouts])
    backends = {c["backend"] for c in curves.values()}
    if "mock" in backends:
        print("WARNING: rollouts come from the MOCK backend; numbers are not results.")
    rng = np.random.default_rng(args.seed)
    per_trace = []
    for tid, c in curves.items():
        meta = c["meta"]
        grp = group_of(meta)
        v = v_from(c["correct"], c["observed"])
        n = meta["n_steps"]
        e = meta["first_error_idx"]
        e = None if e is None or (isinstance(e, float) and np.isnan(e)) else int(e)
        drops = v[:-1] - v[1:]  # drops[k-1] = drop caused by step k
        # Sample splitting: half A picks the decisive step, half B evaluates resampling from it.
        m = len(c["correct"][0])
        perm = rng.permutation(m)
        a_idx, b_idx = np.sort(perm[: m // 2]), np.sort(perm[m // 2:])
        v_a, v_b = v_from(c["correct"], c["observed"], a_idx), v_from(c["correct"], c["observed"], b_idx)
        d_a = cm.decisive_step(v_a)
        row = {"trace_id": tid, "group": grp, "subset": meta["subset"], "n": n, "e": e, "m": m,
               "v0": v[0], "decisive_step": cm.decisive_step(v), "max_drop": float(np.max(drops)),
               "observed_correct": c["observed"]}
        if e is not None:
            row.update({"v_before_error": v[e - 1], "v_at_error": v[e], "drop_at_error": float(drops[e - 1]),
                        "max_drop_elsewhere": float(np.max(np.delete(drops, e - 1))) if n > 1 else np.nan})
        for delta in args.deltas:
            row[f"commit@{delta}"] = cm.commitment_step(v, delta)
            if grp == "fatal":
                row[f"category@{delta}"] = cm.classify_failing_trace(v, e, delta)
        if grp == "fatal":
            row.update({"bt_restart": v_b[0], "bt_before_error": v_b[e - 1],
                        "bt_before_decisive": v_b[(d_a or 1) - 1],
                        "bt_best_prefix_oracle": float(np.max(v[:-1])),
                        "tok_restart": c["tokens"][0], "tok_before_error": c["tokens"][e - 1],
                        "tok_before_decisive": c["tokens"][(d_a or 1) - 1]})
        per_trace.append(row)
    df = pd.DataFrame(per_trace)
    df.to_csv(out / "per_trace.csv", index=False)

    lines = [f"rollouts: {args.rollouts}", f"backend(s): {sorted(backends)}",
             f"traces: {len(df)} ({df.group.value_counts().to_dict()})"]
    summary = {"n_traces": int(len(df)), "groups": df.group.value_counts().to_dict(), "backend": sorted(backends)}

    # 1. Commitment categories for failing traces
    fatal = df[df.group == "fatal"]
    cat_rows = []
    for delta in args.deltas:
        col = f"category@{delta}"
        for cat in ["capability-limited", "latent-commitment", "commits-at-error", "recoverable-error", "never-commits"]:
            share = bootstrap_share((fatal[col] == cat).to_numpy(), seed=args.seed)
            cat_rows.append({"delta": delta, "category": cat, "share": share[0], "ci_lo": share[1],
                             "ci_hi": share[2], "n_fatal": int(len(fatal))})
    cats = pd.DataFrame(cat_rows)
    cats.to_csv(out / "commitment_categories.csv", index=False)
    for _, r in cats[cats.delta == args.primary_delta].iterrows():
        lines.append(f"[fatal, delta={r.delta}] {r.category:<20} {r.share:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")

    # 2. Is the first error the decisive step? Compared with the noise floor of clean traces.
    clean = df[df.group == "clean"]
    not_limited = fatal[fatal.v0 > args.primary_delta]
    is_decisive = (not_limited.decisive_step == not_limited.e).to_numpy()
    share = bootstrap_share(is_decisive, seed=args.seed)
    chance = float(np.mean(1.0 / not_limited.n)) if len(not_limited) else float("nan")
    lines.append(f"[fatal, not capability-limited, n={len(not_limited)}] first error is the decisive step: "
                 f"{share[0]:.3f} [{share[1]:.3f}, {share[2]:.3f}] (chance if uniform: {chance:.3f})")
    for name, sub in [("fatal", fatal), ("recovered", df[df.group == "recovered"])]:
        if len(sub):
            m_drop = bootstrap_share(sub.drop_at_error.to_numpy(), seed=args.seed)
            lines.append(f"[{name}] mean drop in success prob at the first error: {m_drop[0]:+.3f} "
                         f"[{m_drop[1]:+.3f}, {m_drop[2]:+.3f}]")
    if len(clean):
        floor = bootstrap_share(clean.max_drop.to_numpy(), seed=args.seed)
        lines.append(f"[clean] noise floor, mean of the largest per-trace drop: {floor[0]:+.3f} "
                     f"[{floor[1]:+.3f}, {floor[2]:+.3f}]")
    summary["first_error_is_decisive"] = {"share": share, "chance": chance, "n": int(len(not_limited))}

    # 4. Backtracking (sample-split): where to resample from?
    if len(not_limited):
        bt = {}
        for col in ["bt_restart", "bt_before_error", "bt_before_decisive", "bt_best_prefix_oracle"]:
            bt[col] = bootstrap_share(not_limited[col].to_numpy(), seed=args.seed)
            lines.append(f"[backtrack, fatal not limited] {col:<24} success {bt[col][0]:.3f} "
                         f"[{bt[col][1]:.3f}, {bt[col][2]:.3f}]")
        diff = (not_limited.bt_before_decisive - not_limited.bt_before_error).to_numpy()
        d = bootstrap_share(diff, seed=args.seed)
        lines.append(f"[backtrack] decisive-step minus first-error: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        for col in ["tok_restart", "tok_before_error", "tok_before_decisive"]:
            lines.append(f"[backtrack] mean continuation tokens {col:<20} {not_limited[col].mean():.0f}")
        summary["backtrack"] = bt
        summary["backtrack_decisive_minus_error"] = d

    # 5. Event study: success probability aligned at the annotated first error (r = k - e).
    #    r = -1 is "just before the error", r = 0 "just after it". Clean traces are aligned at a
    #    uniformly drawn step as a placebo, so a drop that is only an artefact of time shows there too.
    ev = []
    for tid, c in curves.items():
        meta = c["meta"]
        grp = group_of(meta)
        v = v_from(c["correct"], c["observed"])
        n = meta["n_steps"]
        e = meta["first_error_idx"]
        e = None if e is None or (isinstance(e, float) and np.isnan(e)) else int(e)
        anchor = e if e is not None else int(rng.integers(1, n + 1))
        for r in range(-3, 4):
            k = anchor + r
            if 0 <= k <= n:
                ev.append({"trace_id": tid, "group": grp, "subset": meta["subset"], "r": r, "v": v[k]})
    ev = pd.DataFrame(ev)
    ev_rows = []
    for (grp, r), sub in ev.groupby(["group", "r"]):
        mean, lo, hi = bootstrap_share(sub.v.to_numpy(), seed=args.seed)
        ev_rows.append({"group": grp, "r": r, "mean_v": mean, "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
    ev_table = pd.DataFrame(ev_rows)
    ev_table.to_csv(out / "event_study.csv", index=False)
    for grp in ["fatal", "recovered", "clean"]:
        sub = ev_table[ev_table.group == grp].set_index("r")
        if len(sub):
            lines.append(f"[event study, {grp}] v at r=-1..+1 relative to the error: "
                         + ", ".join(f"r={r}: {sub.loc[r, 'mean_v']:.3f}" for r in (-1, 0, 1) if r in sub.index))

    # 6. Per-tier view of the headline quantities.
    tier_rows = []
    for tier, sub in df.groupby("subset"):
        f = sub[sub.group == "fatal"]
        nl = f[f.v0 > args.primary_delta]
        tier_rows.append({"tier": tier, "n_traces": len(sub), "n_fatal": len(f),
                          "mean_v0_fatal": float(f.v0.mean()) if len(f) else np.nan,
                          "capability_limited_share": float((f.v0 <= args.primary_delta).mean()) if len(f) else np.nan,
                          "first_error_decisive_share": float((nl.decisive_step == nl.e).mean()) if len(nl) else np.nan,
                          "mean_drop_at_error_fatal": float(f.drop_at_error.mean()) if len(f) else np.nan})
    pd.DataFrame(tier_rows).to_csv(out / "per_tier.csv", index=False)

    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
