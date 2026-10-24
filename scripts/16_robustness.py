"""Stage 16 (CPU): how precise are the value curves and the decisive step?

For each generator's rollouts:
  1. Decisive-step uncertainty. Every v_k is drawn from its Jeffreys posterior
     Beta(c + 1/2, m - c + 1/2) (v_n is the observed outcome); the decisive step is recomputed on
     each of D draws. Per trace this gives a posterior over which step is decisive, P(d = e), and the
     posterior probability of the modal step.
  2. Reliability of the value curve. Continuations are split into two halves (by sampling batch
     when there are several independent batches, otherwise at random); the correlation of v_k
     between halves over all prefixes, its Spearman-Brown projection to the full m, and the
     agreement of the decisive step between halves against a position-matched null.
  3. Precision as a function of m: the same split-half correlation after subsampling m' per prefix.
  4. Threshold and definition sensitivity: share of first errors that are decisive under three
     definitions (largest drop; point of no return c = e; first drop above the clean noise floor)
     at delta in {1/16, 1/8, 1/4}.

Usage:
  python scripts/16_robustness.py --gen Qwen2.5-1.5B-Instruct --rollouts results/commitment/Qwen2.5-1.5B-Instruct_s*.jsonl \
      --out results/commitment/robustness_Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
s13 = import_module("13_commitment_analysis")


def group_of(meta: dict) -> str:
    return s13.group_of(meta)


def decisive(v: np.ndarray) -> int:
    """1-based step with the largest drop v_{k-1} - v_k (first one on ties)."""
    return int(np.argmax(v[:-1] - v[1:])) + 1


def commit_step(v: np.ndarray, delta: float) -> int:
    """Point of no return: first k with v_j <= delta for all j >= k."""
    k = len(v) - 1
    while k > 0 and v[k - 1] <= delta:
        k -= 1
    return k


def boot_ci(x: np.ndarray, seed: int = 0, n: int = 2000) -> tuple:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="16: measurement robustness")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.125)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    curves = s13.load_curves([Path(p) for p in args.rollouts])
    summary: dict = {"generator": args.gen, "n_traces": len(curves)}

    # ---- 1. posterior over the decisive step -------------------------------------------------
    rows = []
    for tid, c in curves.items():
        meta = c["meta"]
        if group_of(meta) != "fatal":
            continue
        e = int(meta["first_error_idx"])
        cnt = np.array([x.sum() for x in c["correct"]], dtype=float)
        m = np.array([len(x) for x in c["correct"]], dtype=float)
        v = np.append(cnt / m, 1.0 if c["observed"] else 0.0)
        if v[0] <= args.delta:
            continue
        n = len(cnt)
        draws = rng.beta(cnt + 0.5, m - cnt + 0.5, size=(args.draws, n))
        draws = np.hstack([draws, np.full((args.draws, 1), v[-1])])
        d = np.argmax(draws[:, :-1] - draws[:, 1:], axis=1) + 1
        post = np.bincount(d, minlength=n + 1)[1:] / args.draws
        rows.append({"trace_id": tid, "n": n, "e": e, "m": int(m[0]), "d_point": decisive(v),
                     "p_d_eq_e": float(post[e - 1]), "p_modal": float(post.max()), "modal": int(post.argmax()) + 1,
                     "post_entropy_norm": float(-(post[post > 0] * np.log(post[post > 0])).sum() / np.log(max(n, 2)))})
    post = pd.DataFrame(rows)
    post.to_csv(out / "decisive_posterior.csv", index=False)
    if len(post):
        summary["decisive_posterior"] = {
            "n": int(len(post)),
            "posterior_mean_share_d_eq_e": boot_ci(post.p_d_eq_e.to_numpy(), args.seed),
            "point_share_d_eq_e": boot_ci((post.d_point == post.e).to_numpy(), args.seed),
            "share_modal_ge_0.5": float((post.p_modal >= 0.5).mean()),
            "share_modal_ge_0.8": float((post.p_modal >= 0.8).mean()),
            "median_p_modal": float(post.p_modal.median()),
            "mean_entropy_norm": float(post.post_entropy_norm.mean()),
        }

    # ---- 2./3. reliability of v and of the decisive step -------------------------------------
    def halves(c, k, mprime=None):
        x, lab = c["correct"][k], c["batch"][k]
        idx = np.arange(len(x))
        if mprime:
            idx = np.sort(rng.choice(idx, size=min(mprime, len(idx)), replace=False))
        names = sorted(set(lab[idx]))
        if len(names) >= 2 and not mprime:  # independent batches: split by batch
            a = idx[np.isin(lab[idx], names[: len(names) // 2])]
            b = idx[np.isin(lab[idx], names[len(names) // 2:])]
        else:
            perm = rng.permutation(idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        return x[a].mean(), x[b].mean()

    va, vb = [], []
    for c in curves.values():
        for k in range(len(c["correct"])):
            a, b = halves(c, k)
            va.append(a)
            vb.append(b)
    r = float(np.corrcoef(va, vb)[0, 1])
    summary["split_half"] = {"pearson_r": r, "spearman_brown_full_m": float(2 * r / (1 + r)),
                             "n_prefixes": len(va), "mean_abs_diff": float(np.mean(np.abs(np.subtract(va, vb)))),
                             "split_by": "batch" if any(len(c["batches"]) > 1 for c in curves.values()) else "random"}
    m_full = int(np.median([len(c["correct"][0]) for c in curves.values()]))
    prec = []
    for mp in [2, 4, 8, 16, 32, 64]:
        if mp > m_full:
            break
        a_, b_ = [], []
        for c in curves.values():
            for k in range(len(c["correct"])):
                a, b = halves(c, k, mprime=mp)
                a_.append(a)
                b_.append(b)
        rr = float(np.corrcoef(a_, b_)[0, 1])
        prec.append({"m_total": mp, "m_per_half": mp // 2, "split_half_r": rr, "projected_r_at_m": 2 * rr / (1 + rr)})
    pd.DataFrame(prec).to_csv(out / "precision_vs_m.csv", index=False)
    summary["precision_vs_m"] = prec

    # decisive-step agreement between halves (fatal, not capability-limited on the full sample)
    agree, null_agree = [], []
    rel_e = []
    fat = [c for c in curves.values() if group_of(c["meta"]) == "fatal"]
    for c in fat:
        cnt = np.array([x.mean() for x in c["correct"]])
        if cnt[0] <= args.delta:
            continue
        hv = [halves(c, k) for k in range(len(c["correct"]))]
        obs = 1.0 if c["observed"] else 0.0
        da = decisive(np.append([h[0] for h in hv], obs))
        db = decisive(np.append([h[1] for h in hv], obs))
        agree.append(da == db)
        rel_e.append((da / len(hv), db / len(hv), len(hv)))
    if agree:
        rel = np.array([x[0] for x in rel_e])
        for i, (_, rb, n) in enumerate(rel_e):  # null: half A's step replaced by other traces' relative positions
            others = np.delete(rel, i)
            null_agree.append(np.mean(np.clip(np.ceil(others * n), 1, n) == np.clip(np.ceil(rb * n), 1, n)))
        summary["decisive_agreement_between_halves"] = {"n": len(agree), "agreement": boot_ci(np.array(agree), args.seed),
                                                         "position_matched_null": float(np.mean(null_agree))}

    # ---- 4. threshold / definition sensitivity ----------------------------------------------
    clean_drops = [float(np.max(np.append([x.mean() for x in c["correct"]], 1.0)[:-1]
                                - np.append([x.mean() for x in c["correct"]], 1.0)[1:]))
                   for c in curves.values() if group_of(c["meta"]) == "clean"]
    floor95 = float(np.quantile(clean_drops, 0.95)) if clean_drops else float("nan")
    sens = []
    for delta in [0.0625, 0.125, 0.25]:
        argmax_hit, commit_hit, floor_hit = [], [], []
        for c in fat:
            e = int(c["meta"]["first_error_idx"])
            v = np.append([x.mean() for x in c["correct"]], 1.0 if c["observed"] else 0.0)
            if v[0] <= delta:
                continue
            argmax_hit.append(decisive(v) == e)
            commit_hit.append(commit_step(v, delta) == e)
            drops = v[:-1] - v[1:]
            above = np.nonzero(drops > floor95)[0]
            floor_hit.append(bool(len(above)) and int(above[0]) + 1 == e)
        for name, hits in [("largest_drop", argmax_hit), ("point_of_no_return", commit_hit),
                           ("first_drop_above_clean_floor", floor_hit)]:
            mean, lo, hi = boot_ci(np.array(hits, dtype=float), args.seed)
            sens.append({"delta": delta, "definition": name, "n": len(hits), "share_first_error": mean,
                         "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(sens).to_csv(out / "definition_sensitivity.csv", index=False)
    summary["clean_floor_q95"] = floor95
    summary["definition_sensitivity"] = sens

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    dp = summary.get("decisive_posterior", {})
    print(f"{args.gen}: split-half r = {r:.3f} (Spearman-Brown {2 * r / (1 + r):.3f}, by {summary['split_half']['split_by']}); "
          f"posterior share d=e {dp.get('posterior_mean_share_d_eq_e', [np.nan])[0]:.3f}; "
          f"modal >=0.5 in {dp.get('share_modal_ge_0.5', np.nan):.2f}; "
          f"halves agree {summary.get('decisive_agreement_between_halves', {}).get('agreement', [np.nan])[0]:.3f} "
          f"(null {summary.get('decisive_agreement_between_halves', {}).get('position_matched_null', np.nan):.3f})")


if __name__ == "__main__":
    main()
