"""Stage 14 (GPU): cheap per-step signals for every ProcessBench trace of some generators.

Two modes, both a single forward pass per trace (no sampling):

--mode generator   the trace under its OWN generator: per step, token count, mean / max negative
                   log-likelihood of the written tokens and mean / max entropy of the next-token
                   distribution; plus the token count of the chat-formatted question. These are the
                   signals a deployed system has for free, and the token counts give exact prefill
                   costs for the matched-compute comparison (stage 16).
--mode prm         Qwen2.5-Math-PRM-7B step scores: the probability that each step is correct, the
                   signal a verifier-guided search would use. The reward head is loaded from the
                   checkpoint directly, so no remote code is needed.

Output: JSONL, one row per trace. Example (Kaggle, 2x T4):
  python scripts/14_step_signals.py --mode generator --generators Qwen2.5-1.5B-Instruct --out results/run_c/signals_gen_qwen25_1p5b.jsonl
  python scripts/14_step_signals.py --mode prm --generators all --out results/run_c/signals_prm.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
PRM_ID = "Qwen/Qwen2.5-Math-PRM-7B"
FOCUS = ["Qwen2.5-1.5B-Instruct", "Qwen2-1.5B-Instruct", "Qwen2.5-Math-7B-Instruct", "Llama-3.1-8B-Instruct"]


def load_traces(path: str, generators: list) -> list:
    traces = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if generators != ["all"]:
        traces = [t for t in traces if t["generator"] in generators]
    order = {g: i for i, g in enumerate(FOCUS)}  # the four rolled-out generators first
    return sorted(traces, key=lambda t: (order.get(t["generator"], 99), t["trace_id"]))


def step_spans(head: str, steps: list) -> tuple:
    """Full text and the [start, end) character span of each step (a step owns its trailing separator)."""
    spans, pos, parts = [], len(head), []
    for j, s in enumerate(steps):
        sep = "\n\n" if j < len(steps) - 1 else ""
        spans.append((pos, pos + len(s) + len(sep)))
        parts.append(s + sep)
        pos += len(s) + len(sep)
    return head + "".join(parts), spans


def run_generator(args, traces, log) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    default_model = import_module("12_counterfactual_rollouts").default_model

    by_gen: dict = {}
    for t in traces:
        by_gen.setdefault(t["generator"], []).append(t)
    out = open(args.out, "a", encoding="utf-8")
    done = {json.loads(line)["trace_id"] for line in open(args.out, encoding="utf-8")} if Path(args.out).stat().st_size else set()
    for gen, ts in by_gen.items():
        model_id = default_model(gen)
        log(f"{gen}: {len(ts)} traces, model {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto").eval()
        t0 = time.time()
        for i, t in enumerate(ts):
            if t["trace_id"] in done:
                continue
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["question"]}]
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text, spans = step_spans(head, t["steps"])
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) > args.max_len:
                out.write(json.dumps({"trace_id": t["trace_id"], "generator": gen, "skipped": "length"}) + "\n")
                continue
            with torch.no_grad():
                logits = model(torch.tensor([ids], device=model.device)).logits[0]
            nll, ent = [0.0] * len(ids), [0.0] * len(ids)
            for a in range(0, len(ids) - 1, 512):  # chunks keep the float32 softmax small
                lg = logits[a: a + 512].float()
                logp = torch.log_softmax(lg, dim=-1)
                tgt = torch.tensor(ids[a + 1: a + 1 + lg.shape[0]], device=lg.device)
                k = tgt.shape[0]
                nll[a + 1: a + 1 + k] = (-logp[:k].gather(1, tgt[:, None])[:, 0]).tolist()
                ent[a + 1: a + 1 + k] = (-(logp[:k].exp() * logp[:k]).sum(-1)).tolist()
            n = len(t["steps"])
            per = [[] for _ in range(n)]
            prompt_tokens = 0
            for i_tok, (s, _) in enumerate(offs):
                if s < len(head):
                    prompt_tokens += 1
                    continue
                j = next((j for j, (a0, b0) in enumerate(spans) if a0 <= s < b0), n - 1)
                per[j].append(i_tok)
            row = {"trace_id": t["trace_id"], "generator": gen, "model": model_id, "n_steps": n,
                   "prompt_tokens": prompt_tokens, "step_tokens": [len(p) for p in per],
                   "nll_mean": [float(sum(nll[x] for x in p) / max(len(p), 1)) for p in per],
                   "nll_max": [float(max((nll[x] for x in p), default=0.0)) for p in per],
                   "ent_mean": [float(sum(ent[x] for x in p) / max(len(p), 1)) for p in per],
                   "ent_max": [float(max((ent[x] for x in p), default=0.0)) for p in per]}
            out.write(json.dumps(row) + "\n")
            if i % 50 == 0:
                out.flush()
                log(f"  {gen} {i + 1}/{len(ts)} ({time.time() - t0:.0f}s)")
        out.flush()
        del model
        torch.cuda.empty_cache()
    out.close()


def load_prm(log):
    """Qwen2.5-Math-PRM-7B = Qwen2 backbone + 2-layer head on the hidden state of each <extra_0>."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PRM_ID)
    base = AutoModel.from_pretrained(PRM_ID, dtype=torch.float16, device_map="auto").eval()
    index = json.load(open(hf_hub_download(PRM_ID, "model.safetensors.index.json")))["weight_map"]
    keys = sorted(k for k in index if k.startswith("score."))
    log(f"PRM head keys: {keys}")
    tensors = {}
    for f in sorted({index[k] for k in keys}):
        with safe_open(hf_hub_download(PRM_ID, f), framework="pt") as fh:
            for k in keys:
                if index[k] == f:
                    tensors[k] = fh.get_tensor(k)
    layers = sorted({int(k.split(".")[1]) for k in keys})
    mods = []
    for li in range(max(layers) + 1):
        if li in layers:
            w = tensors[f"score.{li}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data = w.float()
            lin.bias.data = tensors[f"score.{li}.bias"].float()
            mods.append(lin)
        else:
            mods.append(torch.nn.ReLU())
    head = torch.nn.Sequential(*mods).float()
    return tok, base, head


def run_prm(args, traces, log) -> None:
    import torch

    tok, base, head = load_prm(log)
    dev = next(base.parameters()).device
    head = head.to(dev)
    sep = tok.encode("<extra_0>", add_special_tokens=False)
    assert len(sep) == 1, sep
    out = open(args.out, "a", encoding="utf-8")
    done = {json.loads(line)["trace_id"] for line in open(args.out, encoding="utf-8")} if Path(args.out).stat().st_size else set()
    t0 = time.time()
    for i, t in enumerate(traces):
        if t["trace_id"] in done:
            continue
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["question"]},
                {"role": "assistant", "content": "<extra_0>".join(t["steps"]) + "<extra_0>"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > args.max_len:
            out.write(json.dumps({"trace_id": t["trace_id"], "generator": t["generator"], "skipped": "length"}) + "\n")
            continue
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            hid = base(input_ids=x).last_hidden_state[0]
            pos = (x[0] == sep[0]).nonzero()[:, 0].to(hid.device)
            probs = torch.softmax(head.to(hid.device)(hid[pos].float()), dim=-1)[:, 1].tolist()
        row = {"trace_id": t["trace_id"], "generator": t["generator"], "n_steps": len(t["steps"]),
               "prm": probs, "prm_model": PRM_ID, "ok": len(probs) == len(t["steps"])}
        out.write(json.dumps(row) + "\n")
        if i % 100 == 0:
            out.flush()
            log(f"  prm {i + 1}/{len(traces)} ({time.time() - t0:.0f}s)")
    out.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="14: per-step signals (GPU)")
    ap.add_argument("--traces", default="data/processed/processed_traces.jsonl")
    ap.add_argument("--mode", choices=["generator", "prm"], required=True)
    ap.add_argument("--generators", nargs="+", required=True, help="generator names, or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="first N traces only (smoke test)")
    ap.add_argument("--max-len", type=int, default=6000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).touch()
    log_path = Path(args.out).with_suffix(".log")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    traces = load_traces(args.traces, args.generators)
    if args.limit:
        traces = traces[: args.limit]
    log(f"mode={args.mode}: {len(traces)} traces")
    (run_generator if args.mode == "generator" else run_prm)(args, traces, log)
    log("done")


if __name__ == "__main__":
    main()
