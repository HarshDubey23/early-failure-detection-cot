"""Demo (CPU, seconds): from raw on-policy rollouts to one trace's value curve and choke point.

    python scripts/demo_value_curve.py                       # a real Qwen2.5-1.5B trace from Run A
    python scripts/demo_value_curve.py --file results/commitment/Qwen2.5-Math-7B-Instruct_s0.jsonl --trace <trace_id>

Each row of a rollout file is one prefix k of one trace: m continuations sampled from the trace's own
model, and which of them reached the gold answer. v_k = n_correct / m; the trace failed, so v_n = 0.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="results/commitment/Qwen2.5-1.5B-Instruct_s0.jsonl")
ap.add_argument("--trace", default="processbench:gsm8k:gsm8k-156")
args = ap.parse_args()

rows = sorted((r for r in map(json.loads, open(args.file, encoding="utf-8")) if r["trace_id"] == args.trace), key=lambda r: r["k"])
if not rows:
    raise SystemExit(f"trace {args.trace} not in {args.file}")
v = [r["n_correct"] / r["m"] for r in rows] + [0.0]
print(f"{args.trace}  model={rows[0]['generator']}  m={rows[0]['m']} continuations per prefix, temperature={rows[0]['temperature']}")
print(f"human-labelled first error: step {rows[0]['first_error_idx']}")
for k, x in enumerate(v):
    drop = f"   drop at step {k}: {v[k - 1] - x:+.3f}" if k else "   (restart point)"
    print(f"  v_{k} = {x:.3f}  {'#' * round(20 * x):<20}{drop}")
print(f"choke point (largest drop): step {max(range(1, len(v)), key=lambda j: v[j - 1] - v[j])}")
