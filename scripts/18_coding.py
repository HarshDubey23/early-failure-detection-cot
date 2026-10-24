"""Stage 18: human coding of recovered traces (first error annotated, final answer correct).

  python scripts/18_coding.py make  --n 80 --overlap 40      # blinded sheets for two coders
  python scripts/18_coding.py agree --a results/coding/coder_A.csv --b results/coding/coder_B.csv

`make` draws a seeded sample stratified by difficulty tier (all GSM8K items are kept, since there
are only seven) and writes, for every item, the question and the steps with the annotated first
error marked. Coders see nothing else: no generator name, no counterfactual result. Coder A codes
all items, coder B codes the first `--overlap` items (shuffled again so the order differs).

`agree` computes per-field agreement and Cohen's kappa on the overlap, and the category
distribution of coder A with bootstrap intervals. The codebook is docs/CODING_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["label_ok", "category", "confidence", "note"]
CATEGORIES = ["explicit_repair", "silent_repair", "error_not_load_bearing", "answer_not_determined_by_reasoning",
              "label_questionable", "other"]


def make(args) -> None:
    traces = [json.loads(line) for line in open(args.traces, encoding="utf-8") if line.strip()]
    rec = [t for t in traces if t["first_error_idx"] is not None and t["final_correct"]]
    rng = random.Random(args.seed)
    by_tier: dict = {}
    for t in rec:
        by_tier.setdefault(t["subset"], []).append(t)
    for v in by_tier.values():
        rng.shuffle(v)
    sample = list(by_tier.get("gsm8k", []))
    rest = args.n - len(sample)
    other = {k: v for k, v in by_tier.items() if k != "gsm8k"}
    other_total = sum(len(v) for v in other.values())
    for k, v in sorted(other.items()):
        sample += v[: round(rest * len(v) / other_total)]
    sample = sample[: args.n]
    rng.shuffle(sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = pd.DataFrame([{"item": f"R{i + 1:03d}", "trace_id": t["trace_id"], "tier": t["subset"],
                         "generator": t["generator"], "e": t["first_error_idx"], "n": len(t["steps"])}
                        for i, t in enumerate(sample)])
    key.to_csv(out / "sample_key.csv", index=False)  # not shown to coders
    blank = pd.DataFrame({"item": key.item, **{f: "" for f in FIELDS}})
    blank.to_csv(out / "coder_A_blank.csv", index=False)
    b_items = list(key.item[: args.overlap])
    rng.shuffle(b_items)
    pd.DataFrame({"item": b_items, **{f: "" for f in FIELDS}}).to_csv(out / "coder_B_blank.csv", index=False)
    parts = ["<!doctype html><meta charset='utf-8'><title>Recovered-trace coding</title>",
             "<style>body{font:15px/1.5 system-ui;max-width:860px;margin:24px auto;padding:0 16px}"
             ".s{margin:6px 0;padding:6px 10px;border-left:3px solid #ccc}.err{border-color:#d33;background:#fff3f3}"
             "h2{margin-top:40px;border-top:1px solid #ddd;padding-top:16px}</style>",
             "<h1>Recovered-trace coding sheet</h1><p>Read docs/CODING_PROTOCOL.md first. For each item, fill the "
             "row with the same item id in your CSV. The step marked <b>ANNOTATED FIRST ERROR</b> is the expert label; "
             "the final answer of every trace is correct.</p>"]
    for item, t in zip(key.item, sample):
        e = t["first_error_idx"]
        parts.append(f"<h2>{item} <small>({t['subset']}, {len(t['steps'])} steps)</small></h2>")
        parts.append(f"<p><b>Question.</b> {html.escape(t['question'])}</p>")
        for j, s in enumerate(t["steps"], start=1):
            tag = " err" if j == e else ""
            label = "ANNOTATED FIRST ERROR - " if j == e else ""
            parts.append(f"<div class='s{tag}'><b>Step {j}. {label}</b>{html.escape(s)}</div>")
    (out / "coding_items.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {len(key)} items ({key.tier.value_counts().to_dict()}), overlap {args.overlap} -> {out}")


def kappa(a: list, b: list) -> float:
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return float("nan")
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def agree(args) -> None:
    a = pd.read_csv(args.a).set_index("item")
    b = pd.read_csv(args.b).set_index("item")
    common = [i for i in b.index if i in a.index and str(a.loc[i, "category"]).strip() and str(b.loc[i, "category"]).strip()]
    res = {"n_overlap": len(common)}
    for f in ["label_ok", "category"]:
        xa = [str(a.loc[i, f]).strip().lower() for i in common]
        xb = [str(b.loc[i, f]).strip().lower() for i in common]
        res[f] = {"percent_agreement": float(np.mean([x == y for x, y in zip(xa, xb)])) if common else float("nan"),
                  "cohen_kappa": kappa(xa, xb)}
    cats = a.category.dropna().astype(str).str.strip().str.lower()
    cats = cats[cats != ""]
    rng = np.random.default_rng(0)
    dist = {}
    for c in CATEGORIES:
        x = (cats == c).to_numpy(dtype=float)
        boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)] if len(x) else [np.nan]
        dist[c] = [float(x.mean()) if len(x) else float("nan"), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    res["coder_A_distribution"] = dist
    res["coder_A_n"] = int(len(cats))
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="18: human coding of recovered traces")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    m.add_argument("--n", type=int, default=80)
    m.add_argument("--overlap", type=int, default=40)
    m.add_argument("--seed", type=int, default=2026)
    m.add_argument("--out", default="results/coding")
    g = sub.add_parser("agree")
    g.add_argument("--a", required=True)
    g.add_argument("--b", required=True)
    g.add_argument("--out", default="results/coding/agreement.json")
    args = ap.parse_args()
    (make if args.cmd == "make" else agree)(args)


if __name__ == "__main__":
    main()
