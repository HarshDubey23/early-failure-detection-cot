"""Stage 20 (CPU): answer curves a_k from the forced-answer probes (stage 15).

a_k = P(the model's immediate answer after k steps is correct). Reported per generator:
  - mean a_k aligned at the annotated first error (r = k - e), by outcome group;
  - for recovered traces, a_e (right after the error step) and the classification
      answer already correct at the error   a_e >= 0.5
      repair                                a_e < 0.5 (the written trace still ends correct)
  - for fatal traces, how often the model's answer was already wrong before the error
    (a_{e-1} < 0.5) versus becoming wrong at the error;
  - agreement between answer commitment and the value curve, where rollouts exist.

  python scripts/20_answer_curves.py --probes results/run_c/probe.*.jsonl --out results/run_c/answer_curves
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def boot(x, seed=0, n=2000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="20: answer curves")
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(dict)
    meta = {}
    for p in args.probes:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by[r["trace_id"]][r["k"]] = np.mean(r["correct"]) if r["m"] else np.nan
            meta[r["trace_id"]] = r
    ev, rec_rows, fat_rows = [], [], []
    for tid, ks in by.items():
        r = meta[tid]
        n, e = r["n_steps"], r["first_error_idx"]
        if sorted(ks) != list(range(n + 1)):
            continue
        a = np.array([ks[k] for k in range(n + 1)])
        grp = "clean" if e is None else ("recovered" if r["final_correct"] else "fatal")
        anchor = e if e is not None else max(1, n // 2)
        for rr in range(-3, 4):
            if 0 <= anchor + rr <= n:
                ev.append({"generator": r["generator"], "group": grp, "r": rr, "a": a[anchor + rr]})
        if grp == "recovered":
            rec_rows.append({"generator": r["generator"], "trace_id": tid, "a_before": a[e - 1], "a_after": a[e],
                             "class": "answer_already_correct_at_error" if a[e] >= 0.5 else "repair"})
        if grp == "fatal":
            fat_rows.append({"generator": r["generator"], "trace_id": tid, "a0": a[0], "a_before": a[e - 1], "a_after": a[e],
                             "class": ("never_right" if a[0] < 0.5 and a[e - 1] < 0.5 else
                                       "wrong_before_error" if a[e - 1] < 0.5 else
                                       "turns_wrong_at_error" if a[e] < 0.5 else "turns_wrong_later")})
    ev, rec, fat = pd.DataFrame(ev), pd.DataFrame(rec_rows), pd.DataFrame(fat_rows)
    summary: dict = {}
    if len(ev):
        tab = ev.groupby(["generator", "group", "r"]).a.mean().reset_index()
        tab.to_csv(out / "answer_event_study.csv", index=False)
    for gen in sorted(set(ev.generator)) if len(ev) else []:
        s = {}
        rg = rec[rec.generator == gen] if len(rec) else rec
        if len(rg):
            s["recovered_n"] = int(len(rg))
            s["recovered_answer_already_correct_at_error"] = boot((rg["class"] == "answer_already_correct_at_error").to_numpy())
            s["recovered_mean_a_before_after"] = [float(rg.a_before.mean()), float(rg.a_after.mean())]
        fg = fat[fat.generator == gen] if len(fat) else fat
        if len(fg):
            s["fatal_n"] = int(len(fg))
            for c in ["never_right", "wrong_before_error", "turns_wrong_at_error", "turns_wrong_later"]:
                s[f"fatal_{c}"] = boot((fg["class"] == c).to_numpy())
        summary[gen] = s
    rec.to_csv(out / "recovered_classes.csv", index=False)
    fat.to_csv(out / "fatal_classes.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
