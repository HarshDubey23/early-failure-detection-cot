"""Stage 19 (CPU): does our Qwen2.5-Math-PRM-7B implementation reproduce the published ProcessBench F1?

ProcessBench metric (Zheng et al., 2025): a prediction is the first step whose score is below the
threshold (or "no error"). Accuracy is computed separately on erroneous and on error-free
samples; F1 is their harmonic mean, per subset. The published value for Qwen2.5-Math-PRM-7B
(Zhang et al., 2025) is the reference.

  python scripts/19_prm_check.py --prm results/run_c/signals_prm.jsonl --out results/run_c/prm_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="19: PRM implementation check")
    ap.add_argument("--prm", nargs="+", required=True)
    ap.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = {}
    for line in open(args.traces, encoding="utf-8"):
        r = json.loads(line)
        meta[r["trace_id"]] = r
    res: dict = {}
    rows = []
    for p in args.prm:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("skipped") or not r.get("ok", True) or r["trace_id"] not in meta:
                continue
            t = meta[r["trace_id"]]
            below = np.nonzero(np.asarray(r["prm"]) < args.threshold)[0]
            pred = int(below[0]) + 1 if len(below) else None
            rows.append((t["subset"], t["generator"], t["first_error_idx"], pred))
    for subset in sorted({x[0] for x in rows}) + ["all"]:
        sub = [x for x in rows if subset == "all" or x[0] == subset]
        err = [x for x in sub if x[2] is not None]
        ok = [x for x in sub if x[2] is None]
        acc_err = float(np.mean([x[2] == x[3] for x in err])) if err else float("nan")
        acc_ok = float(np.mean([x[3] is None for x in ok])) if ok else float("nan")
        f1 = 2 * acc_err * acc_ok / (acc_err + acc_ok) if (acc_err + acc_ok) > 0 else float("nan")
        res[subset] = {"n": len(sub), "acc_erroneous": acc_err, "acc_correct": acc_ok, "f1": f1}
    subsets = [k for k in res if k != "all"]
    res["mean_f1_over_subsets"] = float(np.mean([res[k]["f1"] for k in subsets])) if subsets else float("nan")
    res["published_reference"] = "Qwen2.5-Math-PRM-7B ProcessBench average F1 73.5 (Zhang et al., 2025)"
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
