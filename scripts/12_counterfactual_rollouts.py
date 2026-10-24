"""Stage 12 (GPU): counterfactual rollouts from every prefix of ProcessBench traces.

For each selected trace of generator G, and each prefix k = 0..n-1, sample m
continuations FROM G ITSELF (on-policy) and record whether each reaches the
gold answer. Stage 13 turns these counts into value curves v_k and compares
the counterfactual decision point with the human-annotated first error e.

Why ProcessBench and the trace's own generator: ProcessBench supplies expert
first-error labels, and several of its generators are open (Qwen2.5-1.5B/7B-
Instruct, Qwen2.5-Math-7B/72B-Instruct), so continuations can come from the
same model that wrote the trace. ProcessBench reformatted solutions into
paragraphs, so prefixes are near- (not exactly) on-policy; report this.

Runs on a single 16 GB GPU for the 1.5B pilot and on 2x T4 (tensor parallel)
or one 24-48 GB GPU for the 7B models. `--backend mock` exercises the whole
pipeline on CPU with a fake generator (used by the tests; produces NO results).

Example (Kaggle, 2x T4):
  pip install vllm math-verify datasets
  python scripts/12_counterfactual_rollouts.py --generator Qwen2.5-1.5B-Instruct \
      --model Qwen/Qwen2.5-1.5B-Instruct --m 16 --out results/commitment/pilot_qwen25_1p5b.jsonl
  python scripts/12_counterfactual_rollouts.py --generator Qwen2.5-Math-7B-Instruct \
      --model Qwen/Qwen2.5-Math-7B-Instruct --tp 2 --dtype half --m 16 \
      --out results/commitment/qwen25_math_7b.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.commitment import answers_match, last_boxed  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def norm_problem(text: str) -> str:
    """Key used to match a ProcessBench problem to its source dataset entry."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# ---------------------------------------------------------------------------
# Gold answers (ProcessBench ships none)
# ---------------------------------------------------------------------------


def gold_from_sources(log) -> dict:
    """{normalised problem: (gold answer, source)} from the public source datasets.

    Every source is optional; failures are logged and coverage is reported.
    """
    gold: dict = {}

    def add(problem, answer, source):
        if problem and answer is not None and str(answer).strip():
            gold.setdefault(norm_problem(problem), (str(answer).strip(), source))

    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets not installed: no source gold answers")
        return gold
    try:
        for row in load_dataset("openai/gsm8k", "main", split="test"):
            add(row["question"], row["answer"].split("####")[-1].strip().replace(",", ""), "gsm8k")
    except Exception as exc:  # noqa: BLE001
        log(f"gsm8k gold unavailable: {exc}")
    for repo in ("HuggingFaceH4/MATH-500",):
        try:
            for row in load_dataset(repo, split="test"):
                add(row["problem"], row["answer"], repo)
        except Exception as exc:  # noqa: BLE001
            log(f"{repo} gold unavailable: {exc}")
    try:
        subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                    "number_theory", "prealgebra", "precalculus"]
        for subject in subjects:
            for row in load_dataset("EleutherAI/hendrycks_math", subject, split="test"):
                add(row["problem"], last_boxed(row["solution"]), "hendrycks_math")
    except Exception as exc:  # noqa: BLE001
        log(f"hendrycks_math gold unavailable: {exc}")
    try:
        for row in load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train"):
            answer = row.get("final_answer")
            if isinstance(answer, list):
                answer = answer[0] if len(answer) == 1 else None  # multi-answer items skipped
            add(row["question"], answer, "olympiadbench")
    except Exception as exc:  # noqa: BLE001
        log(f"OlympiadBench gold unavailable: {exc}")
    try:
        for row in load_dataset("KbsdJames/Omni-MATH", split="test"):
            add(row["problem"], row["answer"], "omnimath")
    except Exception as exc:  # noqa: BLE001
        log(f"Omni-MATH gold unavailable: {exc}")
    return gold


def gold_from_correct_traces(traces: list) -> dict:
    """Fallback: the boxed answer of a ProcessBench trace marked correct."""
    gold = {}
    for t in traces:
        if t["final_correct"]:
            ans = last_boxed(t["steps"][-1]) or last_boxed(" ".join(t["steps"]))
            if ans:
                gold.setdefault(norm_problem(t["question"]), (ans, "processbench-correct-trace"))
    return gold


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class VLLMBackend:
    def __init__(self, model: str, tp: int, dtype: str, max_model_len: int, seed: int, gpu_mem: float):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, tensor_parallel_size=tp, dtype=dtype, seed=seed,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_mem)

    def prompt(self, question: str, prefix_steps: list) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        head = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return head + ("\n\n".join(prefix_steps) + "\n\n" if prefix_steps else "")

    def generate(self, prompts: list, m: int, temperature: float, top_p: float, max_tokens: int, seed: int):
        from vllm import SamplingParams

        params = SamplingParams(n=m, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]


class MockBackend:
    """CPU stand-in that fabricates continuations: for TESTING THE PIPELINE ONLY.

    The fake success probability falls after the annotated error, so stage 13
    has something to find. Rows it writes are tagged backend=mock.
    """

    def __init__(self, gold: dict, traces_by_prompt: dict):
        self.gold, self.meta = gold, traces_by_prompt

    def prompt(self, question: str, prefix_steps: list) -> str:
        key = f"{question}\x00{len(prefix_steps)}"
        return key

    def generate(self, prompts, m, temperature, top_p, max_tokens, seed):
        out = []
        for p in prompts:
            question, k = p.split("\x00")
            k = int(k)
            gold, e = self.meta[p]
            rng = random.Random(int(hashlib.md5(p.encode()).hexdigest()[:8], 16) + seed)
            p_ok = 0.7 if (e is None or k < e) else 0.02
            out.append([(f"... \\boxed{{{gold if rng.random() < p_ok else 'wrong'}}}", rng.randint(50, 400))
                        for _ in range(m)])
        return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="12: counterfactual rollouts (GPU)")
    ap.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    ap.add_argument("--generator", required=True, help="ProcessBench generator name, e.g. Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--model", default=None, help="HF model id (default Qwen/<generator>)")
    ap.add_argument("--tiers", nargs="*", default=["gsm8k", "math", "olympiadbench", "omnimath"])
    ap.add_argument("--groups", nargs="*", default=["fatal", "recovered", "clean"])
    ap.add_argument("--max-per-group", type=int, default=0, help="0 = all")
    ap.add_argument("--m", type=int, default=16, help="continuations per prefix")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="auto", help="'half' on T4 (no bf16)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=128, help="prompts per generate() call")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--no-source-gold", action="store_true")
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
    gold = {} if args.no_source_gold else gold_from_sources(log)
    fallback = gold_from_correct_traces(traces)
    for key, value in fallback.items():
        gold.setdefault(key, value)
    if args.backend == "mock":  # the fake generator needs a target for every trace
        for t in traces:
            gold.setdefault(norm_problem(t["question"]), ("42", "mock"))

    def group_of(t):
        if t["first_error_idx"] is None:
            return "clean"
        return "recovered" if t["final_correct"] else "fatal"

    rng = random.Random(args.seed)
    selected, missing_gold = [], 0
    for grp in args.groups:
        pool = [t for t in traces if t["generator"] == args.generator and t["subset"] in args.tiers
                and group_of(t) == grp]
        rng.shuffle(pool)
        if args.max_per_group:
            pool = pool[: args.max_per_group]
        for t in pool:
            if norm_problem(t["question"]) in gold:
                selected.append(t)
            else:
                missing_gold += 1
    log(f"generator={args.generator}: selected {len(selected)} traces with gold "
        f"({missing_gold} skipped for missing gold); gold sources: "
        f"{sorted({v[1] for v in gold.values()})}")
    if not selected:
        raise SystemExit("no traces selected")

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            row = json.loads(line)
            done.add((row["trace_id"], row["k"]))
    jobs = []  # (trace, k)
    for t in selected:
        for k in range(len(t["steps"])):  # k = n is the observed trace itself
            if (t["trace_id"], k) not in done:
                jobs.append((t, k))
    log(f"{len(jobs)} prefixes to roll out ({len(done)} already done), m={args.m}")

    if args.backend == "vllm":
        model = args.model or f"Qwen/{args.generator}"
        backend = VLLMBackend(model, args.tp, args.dtype, args.max_model_len, args.seed, args.gpu_mem)
    else:
        meta = {}
        backend = MockBackend(gold, meta)
        for t, k in jobs:
            meta[backend.prompt(t["question"], t["steps"][:k])] = (gold[norm_problem(t["question"])][0],
                                                                  t["first_error_idx"])
        model = "mock"

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(jobs), args.batch):
            batch = jobs[start: start + args.batch]
            prompts = [backend.prompt(t["question"], t["steps"][:k]) for t, k in batch]
            results = backend.generate(prompts, args.m, args.temperature, args.top_p, args.max_tokens,
                                       args.seed + start)
            for (t, k), samples in zip(batch, results):
                g, src = gold[norm_problem(t["question"])]
                answers = [last_boxed(text) for text, _ in samples]
                correct = [answers_match(a, g) for a in answers]
                fh.write(json.dumps({
                    "trace_id": t["trace_id"], "generator": t["generator"], "subset": t["subset"],
                    "k": k, "n_steps": len(t["steps"]), "first_error_idx": t["first_error_idx"],
                    "final_correct": t["final_correct"], "gold": g, "gold_source": src,
                    "m": len(samples), "n_correct": int(sum(correct)), "correct": correct,
                    "n_boxed": int(sum(a is not None for a in answers)),
                    "new_tokens": [n for _, n in samples], "answers": answers,
                    "model": model, "backend": args.backend, "temperature": args.temperature,
                    "top_p": args.top_p, "seed": args.seed,
                }, ensure_ascii=False) + "\n")
            fh.flush()
            log(f"  {min(start + args.batch, len(jobs))}/{len(jobs)} prefixes, {time.time() - t0:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
