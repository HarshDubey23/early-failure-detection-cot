"""Grade any step localiser on ChokeBench v1 from the released JSONL alone (CPU, seconds).

A localiser names one 1-based step per trace. For each open failed trace (v_0 > 1/8, not
capability-limited) we report
  acc_human     named step == human first error (the ProcessBench target)
  acc_decisive  posterior P(named step is the choke point), Jeffreys posterior on every v_k
  gain          v_{j-1} - v_0: one attempt resampled before the named step, minus one restart

Simplified protocol: no sample-split weighting, so numbers differ slightly from the paper's
(scripts/49_chokebench.py, which needs the full rollouts). Usage:
  python scripts/chokebench_score.py [path/to/chokebench_v1.jsonl] [submission.json]
submission.json maps trace_id|generator -> step; without it the built-in baselines are scored.
"""
from __future__ import annotations

import json
import sys

import numpy as np

DELTA, N_POST = 0.125, 2000


def load(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    keep = []
    for r in rows:
        v = np.array(r["successes"]) / np.array(r["continuations"])
        if not r["capability_limited"] and v[0] > DELTA:
            keep.append(r)
    return keep


def posterior_choke(r: dict, rng: np.random.Generator) -> np.ndarray:
    """P(d = j), j = 1..n: the step where v drops most; v_n = 0 because the trace failed."""
    s, c = np.array(r["successes"], float), np.array(r["continuations"], float)
    draws = rng.beta(s + 0.5, c - s + 0.5, size=(N_POST, len(s)))
    draws = np.hstack([draws, np.zeros((N_POST, 1))])
    return np.bincount(np.argmax(draws[:, :-1] - draws[:, 1:], axis=1), minlength=len(s)) / N_POST


BASELINES = {
    "human_first_error": lambda r: r["human_first_error"],
    "prm_first_below_0.5": lambda r: next((i + 1 for i, p in enumerate(r["prm"]) if p < 0.5), None),
    "prm_largest_drop": lambda r: int(np.argmax(np.r_[1.0, r["prm"][:-1]] - np.array(r["prm"]))) + 1,
    "prm_min": lambda r: int(np.argmin(r["prm"])) + 1,
    "first_step (= restart)": lambda r: 1,
}


def score(rows: list[dict], name_step) -> dict:
    rng = np.random.default_rng(0)
    h, d, g = [], [], []
    for r in rows:
        j = name_step(r)
        if j is None or not 1 <= j <= r["n_steps"]:  # names no step: a miss that falls back to restart
            h.append(False), d.append(0.0), g.append(0.0)
            continue
        v = np.array(r["successes"]) / np.array(r["continuations"])
        h.append(j == r["human_first_error"])
        d.append(posterior_choke(r, rng)[j - 1])
        g.append(v[j - 1] - v[0])
    return {"n": len(h), "acc_human": float(np.mean(h)), "acc_decisive": float(np.mean(d)),
            "gain_vs_restart": float(np.mean(g))}


if __name__ == "__main__":
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "results/chokebench/chokebench_v1.jsonl")
    if len(sys.argv) > 2:
        sub = json.load(open(sys.argv[2], encoding="utf-8"))
        print(json.dumps(score(rows, lambda r: sub.get(f"{r['trace_id']}|{r['generator']}")), indent=2))
    else:
        assert score(rows, BASELINES["first_step (= restart)"])["gain_vs_restart"] == 0.0
        print(f"{'localiser':24s} {'n':>4s} {'human':>6s} {'choke':>6s} {'gain':>7s}")
        for k, f in BASELINES.items():
            s = score(rows, f)
            print(f"{k:24s} {s['n']:4d} {s['acc_human']:6.3f} {s['acc_decisive']:6.3f} {s['gain_vs_restart']:+7.3f}")
