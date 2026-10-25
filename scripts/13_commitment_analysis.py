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


def load_curves(paths: list, batches: list | None = None, drop_truncated: int | None = None) -> dict:
    """{trace_id: dict(meta, rows sorted by k)}; appends the observed outcome as k = n.

    Accepts several JSONL files (GPU shards and/or independent sampling batches). Rows are keyed by
    (trace, k, batch): shards of one batch never overlap; rows of different batches for the same
    prefix are concatenated, and every continuation keeps its batch label. A duplicated
    (trace, k, batch) row (a resumed run) keeps the first copy. `batches` restricts to those labels.
    """
    by_key: dict = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # last line cut by an interrupted run
                continue
            if row.get("skipped") or row.get("m", 0) == 0:
                continue  # prefix could not be rolled out; the trace becomes incomplete and is dropped
            batch = row.get("batch_id") or f"seed{row.get('seed', 'na')}"
            if batches is not None and batch not in batches:
                continue
            if drop_truncated:  # sensitivity: a continuation cut at the token limit without an answer is missing, not wrong
                keep = [i for i, (t, a) in enumerate(zip(row["new_tokens"], row["answers"]))
                        if not (t >= drop_truncated and a is None)]
                if len(keep) < 2:
                    continue  # fewer than two usable continuations: the prefix (and its trace) is dropped
                row = dict(row, correct=[row["correct"][i] for i in keep], new_tokens=[row["new_tokens"][i] for i in keep],
                           answers=[row["answers"][i] for i in keep], m=len(keep))
            by_key.setdefault((row["trace_id"], row["k"], batch), row)
    by_trace: dict = defaultdict(lambda: defaultdict(dict))
    for (tid, k, batch), row in by_key.items():
        by_trace[tid][batch][k] = row
    curves = {}
    for tid, per_batch in by_trace.items():
        complete = {b: rows for b, rows in per_batch.items()
                    if sorted(rows) == list(range(next(iter(rows.values()))["n_steps"]))}
        if not complete:  # incomplete trace (interrupted run): skip
            continue
        names = sorted(complete)
        first = complete[names[0]]
        n = first[0]["n_steps"]
        correct, labels, tokens = [], [], []
        for k in range(n):
            c, lab, tk = [], [], []
            for b in names:
                r = complete[b][k]
                c += list(r["correct"])
                lab += [b] * len(r["correct"])
                tk += list(r["new_tokens"])
            correct.append(np.asarray(c, dtype=bool))
            labels.append(np.asarray(lab))
            tokens.append(np.mean(tk))
        curves[tid] = {"meta": first[0], "correct": correct, "batch": labels,
                       "observed": bool(first[0]["final_correct"]),
                       "tokens": np.array(tokens + [0.0]), "backend": first[0].get("backend", "?"),
                       "batches": names}
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
    ap.add_argument("--batches", nargs="*", default=None, help="use only these batch labels (default: all)")
    ap.add_argument("--drop-truncated", type=int, default=None,
                    help="sensitivity: drop continuations with >= this many new tokens and no answer")
    ap.add_argument("--n-splits", type=int, default=50,
                    help="random A/B splits of the m continuations; backtracking numbers average over them")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    curves = load_curves([Path(p) for p in args.rollouts], args.batches, args.drop_truncated)
    backends = {c["backend"] for c in curves.values()}
    if "mock" in backends:
        print("WARNING: rollouts come from the MOCK backend; numbers are not results.")
    rng = np.random.default_rng(args.seed)           # event-study placebo anchors
    split_rng = np.random.default_rng(args.seed + 1)  # A/B splits
    splits: dict = {}
    per_trace = []
    for tid, c in curves.items():
        meta = c["meta"]
        grp = group_of(meta)
        v = v_from(c["correct"], c["observed"])
        n = meta["n_steps"]
        e = meta["first_error_idx"]
        e = None if e is None or (isinstance(e, float) and np.isnan(e)) else int(e)
        drops = v[:-1] - v[1:]  # drops[k-1] = drop caused by step k
        m = len(c["correct"][0])
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
            # Repeated sample splitting: in each split, half A selects the trace (v0_A > delta) and
            # picks the decisive step; half B scores every backtracking policy. Nothing is chosen
            # and scored on the same continuations. Averaging over splits removes split luck.
            sp = {k: np.empty(args.n_splits) for k in ("sel", "restart", "err", "dec", "tok_dec")}
            for s_i in range(args.n_splits):
                if args.drop_truncated:  # unequal sample counts per prefix: split each prefix separately
                    halves = [split_rng.permutation(len(x)) for x in c["correct"]]
                    obs = 1.0 if c["observed"] else 0.0
                    v_a = np.array([x[h[: len(x) // 2]].mean() for x, h in zip(c["correct"], halves)] + [obs])
                    v_b = np.array([x[h[len(x) // 2:]].mean() for x, h in zip(c["correct"], halves)] + [obs])
                else:
                    perm = split_rng.permutation(m)
                    a_idx, b_idx = np.sort(perm[: m // 2]), np.sort(perm[m // 2:])
                    v_a = v_from(c["correct"], c["observed"], a_idx)
                    v_b = v_from(c["correct"], c["observed"], b_idx)
                d_a = cm.decisive_step(v_a) or 1
                sp["sel"][s_i] = v_a[0] > args.primary_delta
                sp["restart"][s_i], sp["err"][s_i], sp["dec"][s_i] = v_b[0], v_b[e - 1], v_b[d_a - 1]
                sp["tok_dec"][s_i] = c["tokens"][d_a - 1]
            splits[tid] = sp
            row.update({"bt_select_rate": sp["sel"].mean(), "bt_restart": sp["restart"].mean(),
                        "bt_before_error": sp["err"].mean(), "bt_before_decisive": sp["dec"].mean(),
                        "bt_best_prefix_oracle": float(np.max(v[:-1])),
                        "tok_restart": c["tokens"][0], "tok_before_error": c["tokens"][e - 1],
                        "tok_before_decisive": sp["tok_dec"].mean()})
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
    # Robustness of the chance rate: (a) a position-matched null that gives each trace the relative
    # error positions of the other traces (keeps any tendency of errors to sit late); (b) dropping
    # traces whose first error is the last step, where v_n is the observed outcome.
    rel, nn, dd = (not_limited.e / not_limited.n).to_numpy(), not_limited.n.to_numpy(), not_limited.decisive_step.to_numpy()
    null = [np.mean(np.clip(np.ceil(np.delete(rel, i) * nn[i]).astype(int), 1, nn[i]) == dd[i])
            for i in range(len(not_limited))] if len(not_limited) > 1 else [float("nan")]
    not_last = not_limited[not_limited.e < not_limited.n]
    share_nl = bootstrap_share((not_last.decisive_step == not_last.e).to_numpy(), seed=args.seed)
    lines.append(f"[fatal, not capability-limited] position-matched null: {np.mean(null):.3f}; "
                 f"first error at the last step: {float((not_limited.e == not_limited.n).mean()):.3f}; "
                 f"decisive share without those (n={len(not_last)}): {share_nl[0]:.3f} [{share_nl[1]:.3f}, {share_nl[2]:.3f}]")
    diff_c3 = (not_limited.decisive_step == not_limited.e).to_numpy().astype(float) - np.asarray(null, dtype=float)
    c3 = bootstrap_share(diff_c3, seed=args.seed) if len(not_limited) > 1 else (float("nan"),) * 3
    c3_boot = np.random.default_rng(args.seed)
    c3_b = np.array([diff_c3[c3_boot.integers(0, len(diff_c3), len(diff_c3))].mean() for _ in range(2000)]) \
        if len(not_limited) > 1 else np.array([np.nan])
    c3_p = float(min(1.0, 2 * min((c3_b <= 0).mean(), (c3_b >= 0).mean())))
    lines.append(f"[fatal, not capability-limited] decisive share minus position-matched null: "
                 f"{c3[0]:+.3f} [{c3[1]:+.3f}, {c3[2]:+.3f}], bootstrap p = {c3_p:.4f}")
    summary["first_error_is_decisive"] = {"share": share, "chance": chance, "n": int(len(not_limited)),
                                          "minus_null": c3, "minus_null_p": c3_p,
                                          "position_matched_null": float(np.mean(null)),
                                          "share_error_at_last_step": float((not_limited.e == not_limited.n).mean()),
                                          "share_excluding_last_step": share_nl, "n_excluding_last_step": int(len(not_last))}

    # 4. Backtracking (repeated sample splitting): where to resample from? In every split a trace
    #    enters the pool only if half A says it is not capability-limited, the decisive step comes
    #    from half A, and all policies are scored on half B. The statistic is the split-averaged
    #    pool mean; its CI resamples traces (each with all its splits) with replacement.
    if splits:
        ids = [t for t in fatal.trace_id if t in splits]
        sel = np.stack([splits[t]["sel"] for t in ids])           # (traces, splits)
        pol = {"bt_restart": np.stack([splits[t]["restart"] for t in ids]),
               "bt_before_error": np.stack([splits[t]["err"] for t in ids]),
               "bt_before_decisive": np.stack([splits[t]["dec"] for t in ids]),
               "bt_best_prefix_oracle": np.repeat(
                   fatal.set_index("trace_id").loc[ids, "bt_best_prefix_oracle"].to_numpy()[:, None],
                   args.n_splits, axis=1),
               "tok_restart": np.repeat(fatal.set_index("trace_id").loc[ids, "tok_restart"].to_numpy()[:, None],
                                        args.n_splits, axis=1),
               "tok_before_error": np.repeat(
                   fatal.set_index("trace_id").loc[ids, "tok_before_error"].to_numpy()[:, None],
                   args.n_splits, axis=1),
               "tok_before_decisive": np.stack([splits[t]["tok_dec"] for t in ids])}
        pol["dec_minus_err"] = pol["bt_before_decisive"] - pol["bt_before_error"]
        pol["restart_minus_err"] = pol["bt_restart"] - pol["bt_before_error"]
        pol["dec_minus_restart"] = pol["bt_before_decisive"] - pol["bt_restart"]

        def split_mean(x, w):
            num, den = (x * w).sum(axis=0), w.sum(axis=0)
            return float(np.mean(num[den > 0] / den[den > 0]))

        boot_rng = np.random.default_rng(args.seed + 2)
        gens = np.array([curves[t]["meta"]["generator"] for t in ids])
        if len(set(gens)) == 1:
            boot_idx = [boot_rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
        else:  # pooled over generators: resample traces within each generator (stratified bootstrap)
            groups = [np.nonzero(gens == g)[0] for g in sorted(set(gens))]
            boot_idx = [np.concatenate([gi[boot_rng.integers(0, len(gi), len(gi))] for gi in groups])
                        for _ in range(2000)]

        extra: dict = {}

        def est(x, name=None):
            point = split_mean(x, sel)
            boots = np.array([split_mean(x[i], sel[i]) for i in boot_idx])
            if name:  # 90% interval (TOST at +-0.05) and two-sided bootstrap p-value against 0
                extra[name] = {"ci90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
                               "p": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}
            return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

        n_pool = float(sel.sum(axis=0).mean())
        bt = {}
        for col in ["bt_restart", "bt_before_error", "bt_before_decisive", "bt_best_prefix_oracle"]:
            bt[col] = est(pol[col])
            lines.append(f"[backtrack, fatal, v0_A > delta, mean pool {n_pool:.1f}, {args.n_splits} splits] {col:<24} "
                         f"success {bt[col][0]:.3f} [{bt[col][1]:.3f}, {bt[col][2]:.3f}]")
        d, r = est(pol["dec_minus_err"], "dec_minus_err"), est(pol["restart_minus_err"], "restart_minus_err")
        dr = est(pol["dec_minus_restart"], "dec_minus_restart")
        lines.append(f"[backtrack] decisive-step minus first-error: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        lines.append(f"[backtrack] restart minus first-error: {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]")
        lines.append(f"[backtrack] decisive-step minus restart: {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}]")
        tok = {col: split_mean(pol[col], sel) for col in ["tok_restart", "tok_before_error", "tok_before_decisive"]}
        for col, val in tok.items():
            lines.append(f"[backtrack] mean continuation tokens {col:<20} {val:.0f}")
        summary["backtrack"] = bt
        summary["backtrack_n"] = n_pool
        summary["backtrack_n_splits"] = args.n_splits
        summary["backtrack_decisive_minus_error"] = d
        summary["backtrack_restart_minus_error"] = r
        summary["backtrack_decisive_minus_restart"] = dr
        summary["backtrack_tokens"] = tok
        summary["backtrack_tests"] = extra
        lo, hi = extra["dec_minus_restart"]["ci90"]
        tost = "equivalent" if (lo > -0.05 and hi < 0.05) else ("different" if (lo > 0.05 or hi < -0.05) else "inconclusive")
        summary["backtrack_tost_0.05"] = tost
        lines.append(f"[backtrack] TOST decisive vs restart, margin 0.05: 90% CI [{lo:+.3f}, {hi:+.3f}] -> {tost}")
        lines.append("[backtrack] bootstrap p: " + ", ".join(f"{k} {v['p']:.4f}" for k, v in extra.items()))
        # Robustness: all fatal traces, no selection at all.
        ones = np.ones_like(sel)
        d_all = (split_mean(pol["dec_minus_err"], ones),
                 *np.quantile([split_mean(pol["dec_minus_err"][i], ones) for i in boot_idx], [0.025, 0.975]))
        d_all = tuple(float(x) for x in d_all)
        lines.append(f"[backtrack, all fatal, n={len(ids)}] decisive-step minus first-error: "
                     f"{d_all[0]:+.3f} [{d_all[1]:+.3f}, {d_all[2]:+.3f}]")
        summary["backtrack_all_fatal_decisive_minus_error"] = d_all

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
