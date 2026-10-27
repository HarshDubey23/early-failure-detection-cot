"""Stage 60 (GPU): Run D, counterfactual value curves for long-reasoning models (docs/PREREGISTRATION_RUN_D.md).

Unlike Runs A-C, the traces are generated here by the same model, with the same sampling settings
and the same total token budget as their continuations, so every prefix is exactly on-policy.

  stage 1  one trace per MATH-500 problem; graded with math-verify against the gold answer
  stage 2  for each failed trace, cut its reasoning at fixed token positions t (snapped back to the
           last paragraph break before t, so the cut uses only the past: a stopping time), and
           sample m continuations from each cut under the same total budget

    python scripts/60_reasoning_rollouts.py --model Qwen/Qwen3-1.7B --shard 0/2 --out results/run_d
    python scripts/60_reasoning_rollouts.py --model mock --n-problems 6 --out /tmp/run_d_mock   # CPU pipeline test

Resumable: problems and (trace, cut) pairs already written are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.commitment import answers_match, last_boxed  # noqa: E402

CUTS = [0, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120]
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {  # each model's recommended thinking-mode settings, used for traces and continuations alike
    "Qwen/Qwen3-1.7B": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"temperature": 0.6, "top_p": 0.95, "top_k": -1},
}


def short(model: str) -> str:
    return model.split("/")[-1]


def answer_of(text: str) -> str | None:
    """Final answer: last \\boxed{} after the reasoning block (None if the reasoning never closed)."""
    if "</think>" in text:
        return last_boxed(text.split("</think>")[-1])
    return None


def snap_cut(text: str, keep_from: int = 0) -> int:
    """Character position of the last paragraph break in text (uses only text before the cut)."""
    for sep in ("\n\n", "\n"):
        i = text.rfind(sep, keep_from)
        if i > 0:
            return i + len(sep)
    return len(text)


class MockLLM:
    """CPU stand-in with vLLM's interface, for testing the pipeline without a GPU."""

    def __init__(self, *_, **__):
        self.rng = random.Random(0)

    def generate(self, prompts, params):
        out = []
        for _, p in zip(prompts, params):
            comps = []
            for _ in range(p.n):
                ok = self.rng.random() < 0.5
                body = "\n\n".join(f"step {i}: thinking" for i in range(self.rng.randint(3, 40)))
                text = body + "\n</think>\n" + ("\\boxed{1}" if ok else "\\boxed{2}")
                comps.append(type("C", (), {"text": text, "token_ids": list(range(len(text) // 4)),
                                            "finish_reason": "stop"}))
            out.append(type("O", (), {"outputs": comps}))
        return out


class MockTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        return "<user>" + messages[0]["content"] + "<assistant><think>\n"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))

    def decode(self, ids, skip_special_tokens=False):
        return "x" * (4 * len(ids))


def main() -> None:
    ap = argparse.ArgumentParser(description="60: Run D rollouts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--skip-problems", type=int, default=0, help="pilot problems excluded from the confirmatory run")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-gen", type=int, default=6144, help="total generated-token budget per answer")
    ap.add_argument("--chunk", type=int, default=8, help="failed traces per generate call")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    si, sn = map(int, args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = short(args.model)
    f_traces = out / f"traces.{name}.s{si}.jsonl"
    f_roll = out / f"rollouts.{name}.s{si}.jsonl"

    if args.model == "mock":
        problems = [{"unique_id": f"mock/{i}", "problem": f"What is {i}?", "answer": "1", "level": 1, "subject": "mock"}
                    for i in range(args.n_problems)]
        llm, tok = MockLLM(), MockTok()

        class SP:
            def __init__(self, n=1, **kw):
                self.n, self.kw = n, kw
        SamplingParams = SP
        samp = {"temperature": 0.6, "top_p": 0.95, "top_k": -1}
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [dict(r) for r in ds]
        tok = AutoTokenizer.from_pretrained(args.model)
        # T4 (SM75): float16, no custom all-reduce. CUDA graphs are ~2x faster for a small model; eager is the fallback.
        kw = dict(model=args.model, dtype="half", max_model_len=args.max_gen + 1024, gpu_memory_utilization=0.92,
                  disable_custom_all_reduce=True, seed=0)
        try:
            llm = LLM(**kw)
        except Exception as e:  # noqa: BLE001
            print(f"CUDA-graph start failed ({e!r}); retrying in eager mode", flush=True)
            llm = LLM(**kw, enforce_eager=True)
        samp = dict(SAMPLING[args.model], skip_special_tokens=False)  # keep </think> in the text
    problems = problems[args.skip_problems:args.skip_problems + args.n_problems]
    problems = [p for i, p in enumerate(problems) if i % sn == si]

    def prompt_of(p: dict) -> str:
        msgs = [{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}]
        kw = {"enable_thinking": True} if "Qwen3" in args.model else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def seed_of(*parts) -> int:
        return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

    # ---- stage 1: one trace per problem -------------------------------------------------
    done = {json.loads(x)["problem_id"] for x in open(f_traces, encoding="utf-8")} if f_traces.exists() else set()
    todo = [p for p in problems if p["unique_id"] not in done]
    if todo:
        prompts = [prompt_of(p) for p in todo]
        params = [SamplingParams(n=1, max_tokens=args.max_gen, seed=seed_of("trace", p["unique_id"]), **samp) for p in todo]
        with open(f_traces, "a", encoding="utf-8") as fh:
            for p, pr, o in zip(todo, prompts, llm.generate(prompts, params)):
                c = o.outputs[0]
                ans = answer_of(c.text)
                fh.write(json.dumps({"problem_id": p["unique_id"], "level": p.get("level"), "subject": p.get("subject"),
                                     "gold": p["answer"], "model": args.model, "prompt": pr, "text": c.text,
                                     "gen_tokens": len(c.token_ids), "finish": c.finish_reason, "answer": ans,
                                     "correct": answers_match(ans, p["answer"]), "sampling": samp,
                                     "max_gen": args.max_gen}) + "\n")
    traces = [json.loads(x) for x in open(f_traces, encoding="utf-8")]
    failed = [t for t in traces if not t["correct"]]
    print(f"[{name} s{si}] traces {len(traces)}, failed {len(failed)}", flush=True)

    # ---- stage 2: value curves at fixed token cuts for failed traces ----------------------
    done = set()
    if f_roll.exists():
        for x in open(f_roll, encoding="utf-8"):
            r = json.loads(x)
            done.add((r["problem_id"], r["cut"]))
    for start in range(0, len(failed), args.chunk):
        reqs = []
        for t in failed[start:start + args.chunk]:
            body = t["text"].split("</think>")[0]
            ids = tok.encode(body, add_special_tokens=False)
            for cut in CUTS:
                if (t["problem_id"], cut) in done:
                    continue
                if cut == 0:
                    prefix = ""
                elif cut >= len(ids):          # reasoning already ended before t: the trace is its own continuation
                    reqs.append((t, cut, None, len(ids)))
                    continue
                else:
                    head = tok.decode(ids[:cut], skip_special_tokens=False)
                    prefix = head[:snap_cut(head)]
                n_pre = len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
                reqs.append((t, cut, prefix, n_pre))
        run = [r for r in reqs if r[2] is not None and args.max_gen - r[3] > 64]
        outs = llm.generate([r[0]["prompt"] + r[2] for r in run],
                            [SamplingParams(n=args.m, max_tokens=args.max_gen - r[3], seed=seed_of("cont", r[0]["problem_id"], r[1]),
                                            **samp) for r in run]) if run else []
        by_req = {id(r): o for r, o in zip(run, outs)}
        with open(f_roll, "a", encoding="utf-8") as fh:
            for r in reqs:
                t, cut, prefix, n_pre = r
                row = {"problem_id": t["problem_id"], "model": args.model, "cut": cut, "prefix_tokens": n_pre,
                       "trace_tokens": t["gen_tokens"], "gold": t["gold"], "level": t.get("level")}
                if prefix is None:
                    row.update({"ended": True, "m": 0, "correct": [], "new_tokens": [], "answers": [], "finish": []})
                elif id(r) not in by_req:
                    row.update({"ended": False, "no_room": True, "m": 0, "correct": [], "new_tokens": [], "answers": [], "finish": []})
                else:
                    comps = by_req[id(r)].outputs
                    answers = [answer_of(prefix + c.text) for c in comps]
                    row.update({"ended": False, "m": len(comps), "prefix_chars": len(prefix),
                                "correct": [answers_match(a, t["gold"]) for a in answers],
                                "new_tokens": [len(c.token_ids) for c in comps], "answers": answers,
                                "finish": [c.finish_reason for c in comps]})
                fh.write(json.dumps(row) + "\n")
        print(f"[{name} s{si}] cut rows written for traces {start}..{start + args.chunk}", flush=True)


if __name__ == "__main__":
    main()
