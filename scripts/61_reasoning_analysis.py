"""Stage 61 (CPU): preregistered analysis of Run D (docs/PREREGISTRATION_RUN_D.md).

D1  Proposition 1 on exactly on-policy traces: for every fixed cut t, mean over failed traces of
    (v_t - v_0) <= 0. A violation is a one-sided Holm-adjusted p < 0.05 for a positive mean.
D2  Matched compute: P(solved) of failed traces within B restart-equivalents of generated tokens,
    restart vs resampling from the cut chosen on half A of the continuations (scored on half B,
    50 splits). "Restart not beaten" if the upper 95% bound of (cut - restart) is below +0.05 at B = 4.
D3  Descriptive: capability-limited share (v_0 <= 1/8) and where the largest drop in v falls.
D4  Sensitivity: D1 and D2 on failures that ended with a wrong answer (truncated traces removed).

    python scripts/61_reasoning_analysis.py --rollouts results/run_d --exclude-first 20 --out results/run_d/analysis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import betaln

DELTA = 0.125
BUDGETS = [1, 2, 4]
N_SPLIT, N_BOOT = 50, 2000


def jeffreys_pass(c: float, n: float, N: float) -> float:
    if N < 1 or n == 0:
        return 0.0
    a, b = c + 0.5, n - c + 0.5
    return float(1 - np.exp(betaln(a, b + N) - betaln(a, b)))


def boot(x: np.ndarray, rng) -> list:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 3 + [0]
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return [float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(x))]


def p_positive(x: np.ndarray, rng) -> float:
    """One-sided bootstrap p-value for H0: mean <= 0 (share of centred resamples >= observed mean)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 1.0
    c = x - x.mean()
    bs = c[rng.integers(0, len(c), (N_BOOT, len(c)))].mean(axis=1)
    return float((np.sum(bs >= x.mean()) + 1) / (N_BOOT + 1))


def holm(ps: list) -> list:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def load(path: Path, exclude_first: int):
    traces, rows = {}, defaultdict(dict)
    for f in sorted(path.glob("traces.*.jsonl")):
        for x in open(f, encoding="utf-8"):
            t = json.loads(x)
            traces[(t["model"], t["problem_id"])] = t
    for f in sorted(path.glob("rollouts.*.jsonl")):
        for x in open(f, encoding="utf-8"):
            r = json.loads(x)
            rows[(r["model"], r["problem_id"])].setdefault(r["cut"], r)
    excluded = set()
    if exclude_first:  # pilot problems (first N of MATH-500 in dataset order) are not analysed
        from datasets import load_dataset
        excluded = {r["unique_id"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(exclude_first))}
    return traces, rows, excluded


def analyse(model: str, traces: dict, rows: dict, keep, rng) -> dict:
    fails = [k for k, t in traces.items() if k[0] == model and not t["correct"] and keep(t) and 0 in rows.get(k, {})]
    cuts = sorted({c for k in fails for c in rows[k] if c > 0})
    v = {}
    for k in fails:
        r = rows[k]
        v[k] = {c: (np.mean(r[c]["correct"]) if r[c]["m"] else 0.0) for c in r}   # ended cut: the failed trace itself
    out = {"n_failed": len(fails), "n_traces": sum(1 for k in traces if k[0] == model),
           "truncated_share": float(np.mean([traces[k]["finish"] == "length" for k in fails])) if fails else float("nan")}
    # D1
    d1, ps = {}, []
    for c in cuts:
        diff = np.array([v[k][c] - v[k][0] for k in fails if c in v[k]])
        d1[c] = boot(diff, rng) + [float(np.mean([rows[k][c].get("ended", False) for k in fails if c in rows[k]]))]
        ps.append(p_positive(diff, rng))
    for c, p in zip(cuts, holm(ps)):
        d1[c].append(p)
    out["D1"] = {"cut_mean_ci_n_endedshare_pholm": d1, "violated": any(d1[c][-1] < 0.05 for c in cuts)}
    # D3
    v0 = np.array([v[k][0] for k in fails])
    out["D3"] = {"capability_limited": boot((v0 <= DELTA).astype(float), rng), "v0_mean": boot(v0, rng)}
    drops = []
    for k in fails:
        if v[k][0] <= DELTA:
            continue
        seq = [0] + [c for c in cuts if c in v[k]]
        vals = [v[k][c] for c in seq] + [0.0]
        drops.append((seq + ["end"])[int(np.argmax(np.array(vals[:-1]) - np.array(vals[1:]))) + 1])
    out["D3"]["largest_drop_after_cut"] = {str(c): drops.count(c) for c in [0] + cuts + ["end"]}
    # D2: matched budget in generated tokens
    gains = {b: [] for b in BUDGETS}
    for k in fails:
        r = rows[k]
        avail = [c for c in r if r[c]["m"] >= 2]
        if 0 not in avail:
            continue
        m = min(r[c]["m"] for c in avail)
        c0 = max(np.mean(r[0]["new_tokens"]), 1.0)
        per = {b: [] for b in BUDGETS}
        for _ in range(N_SPLIT):
            perm = rng.permutation(m)
            A, B = perm[: m // 2], perm[m // 2:]
            def val(c, idx, b):
                ok = np.array(r[c]["correct"])[idx]
                cost = max(np.mean(np.array(r[c]["new_tokens"])[idx]), 1.0)
                return jeffreys_pass(ok.sum(), len(idx), np.floor(b * c0 / cost))
            for b in BUDGETS:
                best = max(avail, key=lambda c: val(c, A, b))          # chosen on half A
                per[b].append(val(best, B, b) - val(0, B, b))         # scored on half B
        for b in BUDGETS:
            gains[b].append(float(np.mean(per[b])))
    out["D2"] = {str(b): boot(np.array(gains[b]), rng) for b in BUDGETS}
    out["D2"]["restart_not_beaten_at_4"] = bool(out["D2"]["4"][2] < 0.05)
    return out


def figure(res: dict, path: Path) -> None:
    """Fig. 29: (a) value of a failed trace's prefix minus a fresh start, by cut; (b) matched-compute gain over restart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    a.axhline(0, color="#52514e", lw=0.9, ls=(0, (3, 3)))
    for i, (model, r) in enumerate(res.items()):
        d1 = r["primary"]["D1"]["cut_mean_ci_n_endedshare_pholm"]
        cuts = [0] + [int(c) for c in d1]
        mu = [0.0] + [d1[c][0] for c in d1]
        lo = [0.0] + [d1[c][1] for c in d1]
        hi = [0.0] + [d1[c][2] for c in d1]
        lab = f"{model.split('/')[-1]} ({r['primary']['n_failed']} failures)"
        a.plot(cuts, mu, "o-", color=cols[i % 2], ms=4, lw=1.6, label=lab)
        a.fill_between(cuts, lo, hi, color=cols[i % 2], alpha=0.15, lw=0)
        bs = [str(x) for x in BUDGETS]
        g = [r["primary"]["D2"][x] for x in bs]
        x = np.arange(len(bs)) + (i - 0.5) * 0.3
        b.bar(x, [v[0] for v in g], 0.28, color=cols[i % 2])
        b.errorbar(x, [v[0] for v in g], yerr=[[v[0] - v[1] for v in g], [v[2] - v[0] for v in g]], fmt="none", ecolor="#0b0b0b", lw=1)
    a.set(xlabel="Cut position t in the failed trace (generated tokens)", ylabel="v(t) − v(0)",
          title="a  Is a failed partial trace worth more than a fresh start?")
    a.text(a.get_xlim()[1], 0.005, "fresh start", ha="right", va="bottom", fontsize=8.5, color="#52514e")
    a.legend(loc="lower left")
    b.axhline(0.05, color="#52514e", lw=0.9, ls=(0, (3, 3)))
    b.axhline(0, color="#52514e", lw=0.6)
    b.set_xticks(np.arange(len(BUDGETS)), [f"{x}×" for x in BUDGETS])
    b.set(xlabel="Compute budget (restart-equivalents)", ylabel="Best cut − restart, P(solved)",
          title="b  Best resampling point vs restart")
    b.text(-0.45, 0.052, "+5-point bar", ha="left", va="bottom", fontsize=8.5, color="#52514e")
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="results/run_d")
    ap.add_argument("--exclude-first", type=int, default=20)
    ap.add_argument("--out", default="results/run_d/analysis")
    ap.add_argument("--fig", default="results/figures/fig29_reasoning")
    args = ap.parse_args()
    rng = np.random.default_rng(61)
    traces, rows, excluded = load(Path(args.rollouts), args.exclude_first)
    traces = {k: t for k, t in traces.items() if t["problem_id"] not in excluded}
    res = {}
    for model in sorted({k[0] for k in traces}):
        acc = np.mean([t["correct"] for k, t in traces.items() if k[0] == model])
        res[model] = {"accuracy": float(acc),
                      "primary": analyse(model, traces, rows, lambda t: True, rng),
                      "D4_wrong_answer_only": analyse(model, traces, rows, lambda t: t["finish"] != "length", rng)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    figure(res, Path(args.fig))
    for model, r in res.items():
        p = r["primary"]
        print(f"\n== {model}: accuracy {r['accuracy']:.3f}, failed {p['n_failed']} (truncated {p['truncated_share']:.2f})")
        for c, row in p["D1"]["cut_mean_ci_n_endedshare_pholm"].items():
            print(f"  D1 t={c}: v_t - v_0 = {row[0]:+.3f} [{row[1]:+.3f}, {row[2]:+.3f}] n={row[3]} ended={row[4]:.2f} p_holm={row[5]:.3f}")
        print(f"  D1 violated: {p['D1']['violated']}")
        print("  D2: " + "; ".join(f"B={b}: {p['D2'][str(b)][0]:+.3f} [{p['D2'][str(b)][1]:+.3f}, {p['D2'][str(b)][2]:+.3f}]" for b in BUDGETS)
              + f"; restart not beaten at 4: {p['D2']['restart_not_beaten_at_4']}")
        print(f"  D3: capability-limited {p['D3']['capability_limited'][0]:.2f}, mean v0 {p['D3']['v0_mean'][0]:.3f}, "
              f"largest drop after cut: {p['D3']['largest_drop_after_cut']}")


if __name__ == "__main__":
    main()
