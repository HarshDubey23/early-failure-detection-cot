"""Stage 15 (GPU): what answer has the model already settled on after k steps?

For every prefix s_1..s_k of a trace, the trace's own generator is forced to answer at once
("Final answer: $\\boxed{") and m short answers are sampled. This gives the *answer curve*
a_k = P(forced answer is correct | question, first k steps), the counterpart of the value curve
v_k = P(the continued solution is correct). Comparing the two separates:

  - repair: the forced answer is wrong right after the annotated error and the written
    reasoning later fixes it (a_e low, final answer correct);
  - non-determination: the model's answer is already correct before or at the error and the
    error does not move it (a_{e-1} and a_e high), i.e. the written step does not determine
    the answer.

Output: JSONL, one row per (trace, k), in the same format family as stage 12.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis.commitment import answers_match, last_boxed  # noqa: E402

s12 = import_module("12_counterfactual_rollouts")
FORCE = "Final answer: $\\boxed{"


def main() -> None:
    ap = argparse.ArgumentParser(description="15: forced-answer probes (GPU)")
    ap.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--groups", nargs="*", default=["fatal", "recovered", "clean"])
    ap.add_argument("--max-clean", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N traces only (smoke test)")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=5005)
    ap.add_argument("--select-seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    traces = [json.loads(line) for line in open(args.traces, encoding="utf-8") if line.strip()]
    gold = s12.gold_from_sources(log)
    for key, value in s12.gold_from_correct_traces(traces).items():
        gold.setdefault(key, value)

    def group_of(t):
        if t["first_error_idx"] is None:
            return "clean"
        return "recovered" if t["final_correct"] else "fatal"

    rng = random.Random(args.select_seed)  # same selection as stage 12 with --select-seed
    selected = []
    for grp in args.groups:
        pool = [t for t in traces if t["generator"] == args.generator and group_of(t) == grp]
        rng.shuffle(pool)
        if grp == "clean" and args.max_clean:
            pool = pool[: args.max_clean]
        selected += [t for t in pool if s12.norm_problem(t["question"]) in gold]
    if args.num_shards > 1:
        selected = [t for t in selected
                    if int(hashlib.md5(t["trace_id"].encode()).hexdigest(), 16) % args.num_shards == args.shard]
    if args.limit:
        selected = selected[: args.limit]
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["trace_id"], r["k"]))
            except json.JSONDecodeError:
                continue
    jobs = [(t, k) for t in selected for k in range(len(t["steps"]) + 1) if (t["trace_id"], k) not in done]
    log(f"{args.generator}: {len(selected)} traces, {len(jobs)} prefixes to probe, m={args.m}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = args.model or s12.default_model(args.generator)
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=args.tp, dtype=args.dtype, seed=args.seed,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, disable_custom_all_reduce=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(jobs), 256):
            batch = jobs[start: start + 256]
            inputs, params, keep = [], [], []
            for i, (t, k) in enumerate(batch):
                text = s12.chat_prompt(tok, t["question"], t["steps"][:k]) + FORCE
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) + args.max_tokens > args.max_model_len:
                    continue
                seed = int(hashlib.sha256(f"{t['trace_id']}|{k}|{args.seed}".encode()).hexdigest()[:8], 16)
                inputs.append({"prompt_token_ids": ids})
                params.append(SamplingParams(n=args.m, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                                             max_tokens=args.max_tokens, seed=seed))
                keep.append(i)
            outs = llm.generate(inputs, params, use_tqdm=False) if inputs else []
            for i, o in zip(keep, outs):
                t, k = batch[i]
                g, _ = gold[s12.norm_problem(t["question"])]
                own = last_boxed(t["steps"][-1]) or last_boxed(" ".join(t["steps"]))
                answers = [last_boxed("\\boxed{" + c.text) for c in o.outputs]
                fh.write(json.dumps({
                    "trace_id": t["trace_id"], "generator": t["generator"], "subset": t["subset"], "k": k,
                    "n_steps": len(t["steps"]), "first_error_idx": t["first_error_idx"],
                    "final_correct": t["final_correct"], "gold": g, "own_answer": own, "answers": answers,
                    "correct": [answers_match(a, g) for a in answers],
                    "match_own": [bool(own) and answers_match(a, own) for a in answers],
                    "model": model, "seed": args.seed, "m": len(answers)}, ensure_ascii=False) + "\n")
            fh.flush()
            log(f"  {min(start + 256, len(jobs))}/{len(jobs)} prefixes, {time.time() - t0:.0f}s")
    log("done")
    os._exit(0)  # vLLM worker processes can keep the interpreter alive after the run


if __name__ == "__main__":
    main()
