"""Stage 32 (GPU): every generator call the recovery experiment needs, for one generator.

Substages (run in this order; each appends to its own JSONL and resumes after an interruption):

  drafts  one response per item that has no released draft (temperature 0.7, top-p 0.95)
  bank    the counterfactual rollout bank: m continuations from sentence boundaries of each
          draft. System points (every draft the detector D flags): k = 0, k = j*-1 (just before
          D's first flagged sentence) and the midpoint between them. Mechanism study (RAGTruth
          drafts with human labels): every boundary k = 0..n-1 of every human-hallucinated draft
          and of up to --max-clean-curves human-clean drafts.
  repair  one greedy rewrite per D-flagged sentence of each flagged draft ("claim repair")
  cove    factored Chain-of-Verification on every draft (plan, answer each question from the
          source without seeing the draft, revise): the verification-and-revision baseline

Continuations are sampled from the generator that wrote the draft, at the draft's own sampling
temperature when it is known (RAGTruth records it), so the bank estimates the on-policy
probability that a response finished from boundary k is faithful.

Usage (Kaggle, 2x T4):
  python scripts/32_generate.py --items data/recovery/items.ragtruth.llama-2-7b-chat.jsonl \
      --generator llama-2-7b-chat --stage bank --flags results/recovery/run/scores_D.jsonl --tp 2
  python scripts/32_generate.py ... --backend mock      # CPU pipeline test, NOT results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recovery import engine as E  # noqa: E402
from src.recovery import prompts as P  # noqa: E402
from src.recovery.ragtruth import mechanism_points, sentence_labels, sentence_spans  # noqa: E402
from src.recovery.replay import first_flag, flagged_sentences, system_points  # noqa: E402

GEN_IDS = {"llama-2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
           "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
           "mistral-7B-instruct": "mistralai/Mistral-7B-Instruct-v0.1"}


def read_jsonl(path) -> list:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def draft_text(item: dict, drafts: dict) -> str | None:
    return item["draft"] if item.get("draft") is not None else drafts.get(item["item_id"], {}).get("draft")


def main() -> None:
    ap = argparse.ArgumentParser(description="32: generator calls (GPU)")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--stage", choices=["drafts", "bank", "repair", "cove"], nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/recovery/run")
    ap.add_argument("--flags", default=None, help="detector D scores of the drafts (stage 33)")
    ap.add_argument("--tau", type=float, default=0.5, help="sentence flagged when D score >= tau")
    ap.add_argument("--m", type=int, default=8, help="continuations per system point")
    ap.add_argument("--m-curve", type=int, default=8, help="continuations per boundary, mechanism study")
    ap.add_argument("--max-clean-curves", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--batch", type=int, default=48, help="prompts per generate() call")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds; stop starting new batches after this")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log_path = out / f"gen.{args.generator}.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def out_of_time():
        return args.time_limit and time.time() - t_start > args.time_limit

    items = [it for path in args.items for it in read_jsonl(path) if it["generator"] == args.generator]
    log(f"{len(items)} items for {args.generator}; stages {args.stage}; backend {args.backend}")
    if args.backend == "vllm":
        gen = E.VLLMGenerator(args.model or GEN_IDS[args.generator], args.generator, tp=args.tp, dtype=args.dtype,
                              max_model_len=args.max_model_len, gpu_mem=args.gpu_mem, seed=args.seed)
    else:
        gen = E.MockGenerator(args.generator)
    drafts_path = out / f"drafts.{args.generator}.jsonl"
    drafts = {r["item_id"]: r for r in read_jsonl(drafts_path)}

    def run_batches(jobs, write, n, temperature, top_p, max_tokens, label):
        """jobs: list of (key, prompt_string, meta). Generates in batches and calls write()."""
        done = 0
        t0 = time.time()
        for start in range(0, len(jobs), args.batch):
            if out_of_time():
                log(f"{label}: time limit reached after {done}/{len(jobs)} jobs")
                return False
            chunk = jobs[start: start + args.batch]
            seeds = [E.seed_for(args.seed, label, key) for key, _, _ in chunk]
            res, secs = gen.generate([p for _, p, _ in chunk], n, temperature, top_p, max_tokens, seeds)
            write(chunk, res, secs)
            done += len(chunk)
            log(f"{label}: {done}/{len(jobs)} jobs, {time.time() - t0:.0f}s")
        return True

    # ---------------------------------------------------------------- drafts
    if "drafts" in args.stage:
        todo = [it for it in items if it.get("draft") is None and it["item_id"] not in drafts]
        jobs = [(it["item_id"], gen.fmt(it["prompt"]), it) for it in todo]
        fh = open(drafts_path, "a", encoding="utf-8")

        def write(chunk, res, secs):
            n_tok = sum(r["samples"][0][1] for r in res if r)
            for (key, _, it), r in zip(chunk, res):
                row = {"item_id": key, "generator": args.generator, "backend": gen.backend,
                       "temperature": args.temperature, "top_p": args.top_p, "batch_seconds": secs,
                       "batch_gen_tokens": n_tok}
                if r is None:
                    row.update({"draft": None, "skipped": "context"})
                else:
                    row.update({"draft": r["samples"][0][0].strip(), "gen_tokens": r["samples"][0][1],
                                "prompt_tokens": r["prompt_tokens"]})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                drafts[key] = row
            fh.flush()
        run_batches(jobs, write, 1, args.temperature, args.top_p, args.max_tokens, "drafts")
        fh.close()

    # flags from detector D on the drafts
    flags = {}
    if args.flags:
        for r in read_jsonl(args.flags):
            if r.get("kind") == "draft" and r.get("sent_scores") is not None:
                flags[r["item_id"]] = r["sent_scores"]

    # ---------------------------------------------------------------- bank
    if "bank" in args.stage:
        bank_path = out / f"bank.{args.generator}.jsonl"
        have = {(r["item_id"], r["k"]) for r in read_jsonl(bank_path)}
        jobs = []
        # mechanism study: position-matched placebo anchors for human-clean responses
        rels = []
        for it in items:
            if it.get("human_labels") and it.get("draft_quality") == "good":
                lab = sentence_labels(it["draft"], it["human_labels"])
                if any(lab):
                    rels.append((lab.index(1) + 1) / len(lab))
        n_clean = {}
        for it in items:
            d = draft_text(it, drafts)
            if not d:
                continue
            spans = sentence_spans(d)
            n = len(spans)
            pts = {}
            sc = flags.get(it["item_id"])
            if sc is not None and first_flag(sc, args.tau) is not None:
                for k in system_points(sc, args.tau):
                    pts[k] = args.m
            if it.get("human_labels") is not None and it.get("draft_quality") == "good" and n >= 2:
                is_h = len(it["human_labels"]) > 0
                if is_h or n_clean.get(it["task"], 0) < args.max_clean_curves // 3:
                    n_clean[it["task"]] = n_clean.get(it["task"], 0) + (0 if is_h else 1)
                    for k in mechanism_points(d, it["human_labels"], it["item_id"], rels)["points"]:
                        pts[k] = max(pts.get(k, 0), args.m_curve)
            temp = it.get("draft_temperature") or args.temperature
            top_p = 1.0 if it.get("draft_temperature") else args.top_p
            for k, mk in sorted(pts.items()):
                if (it["item_id"], k) in have:
                    continue
                prefix = d[: spans[k][0]] if k > 0 else ""
                jobs.append(((it["item_id"], k), gen.fmt(it["prompt"], prefix), (it, k, mk, prefix, temp, top_p, n)))
        log(f"bank: {len(jobs)} (item, boundary) jobs; {sum(j[2][2] for j in jobs)} continuations")
        fh = open(bank_path, "a", encoding="utf-8")
        # group by (m, temperature, top_p) so each generate() call has one sampling setting
        groups = {}
        for j in jobs:
            groups.setdefault((j[2][2], j[2][4], j[2][5]), []).append(j)
        for (mk, temp, top_p), gjobs in sorted(groups.items()):
            def write(chunk, res, secs, mk=mk, temp=temp, top_p=top_p):
                n_tok = sum(s[1] for r in res if r for s in r["samples"])
                for (key, _, (it, k, _, prefix, _, _, n)), r in zip(chunk, res):
                    row = {"item_id": it["item_id"], "k": k, "n_sent": n, "generator": args.generator,
                           "backend": gen.backend, "temperature": temp, "top_p": top_p, "m": mk,
                           "prefix": prefix, "prefix_tokens": gen.n_tokens(prefix) if prefix else 0,
                           "batch_seconds": secs, "batch_gen_tokens": n_tok}
                    if r is None:
                        row.update({"skipped": "context", "conts": [], "gen_tokens": []})
                    else:
                        row.update({"prompt_tokens": r["prompt_tokens"], "conts": [s[0] for s in r["samples"]],
                                    "gen_tokens": [s[1] for s in r["samples"]]})
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if not run_batches(gjobs, write, mk, temp, top_p, args.max_tokens, f"bank m={mk} T={temp}"):
                break
        fh.close()

    # ---------------------------------------------------------------- repair
    if "repair" in args.stage:
        rep_path = out / f"repair.{args.generator}.jsonl"
        have = {(r["item_id"], r["j"]) for r in read_jsonl(rep_path)}
        jobs = []
        for it in items:
            d = draft_text(it, drafts)
            sc = flags.get(it["item_id"])
            if not d or sc is None:
                continue
            spans = sentence_spans(d)
            for j in flagged_sentences(sc, args.tau):  # 1-based
                if (it["item_id"], j) in have:
                    continue
                sent = d[spans[j - 1][0]: spans[j - 1][1]]
                jobs.append(((it["item_id"], j), gen.fmt(P.repair_prompt(it["prompt"], d, j, sent)), (it, j, sent)))
        shared = {}  # item -> tokens of the prompt prefix all its repair calls share (prefix cache)
        by_item = {}
        for key, prompt, _ in jobs:
            by_item.setdefault(key[0], []).append(prompt)
        for iid, prompts in by_item.items():
            shared[iid] = gen.n_tokens(os.path.commonprefix(prompts)) if len(prompts) > 1 else None
        log(f"repair: {len(jobs)} flagged sentences")
        fh = open(rep_path, "a", encoding="utf-8")

        def write(chunk, res, secs):
            for (key, _, (it, j, sent)), r in zip(chunk, res):
                row = {"item_id": it["item_id"], "j": j, "original": sent, "backend": gen.backend,
                       "batch_seconds": secs, "shared_tokens": shared.get(it["item_id"])}
                if r is None:
                    row.update({"raw": None, "rewrite": None, "skipped": "context"})
                else:
                    raw = r["samples"][0][0]
                    row.update({"raw": raw, "rewrite": P.parse_repair(raw), "gen_tokens": r["samples"][0][1],
                                "prompt_tokens": r["prompt_tokens"]})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        run_batches(jobs, write, 1, 0.0, 1.0, 160, "repair")
        fh.close()

    # ---------------------------------------------------------------- cove
    if "cove" in args.stage:
        cove_path = out / f"cove.{args.generator}.jsonl"
        have = {r["item_id"] for r in read_jsonl(cove_path)}
        todo = [it for it in items if it["item_id"] not in have and draft_text(it, drafts)]
        log(f"cove: {len(todo)} drafts")
        for start in range(0, len(todo), args.batch):
            if out_of_time():
                log("cove: time limit reached")
                break
            chunk = todo[start: start + args.batch]
            recs = {it["item_id"]: {"item_id": it["item_id"], "backend": gen.backend, "calls": []} for it in chunk}
            # 1. plan
            res, secs = gen.generate([gen.fmt(P.cove_plan_prompt(it["prompt"], draft_text(it, drafts))) for it in chunk],
                                     1, 0.0, 1.0, 200, [E.seed_for(args.seed, "plan", it["item_id"]) for it in chunk])
            qs = {}
            for it, r in zip(chunk, res):
                rec = recs[it["item_id"]]
                if r is None:
                    rec["skipped"] = "context"
                    continue
                rec["calls"].append({"step": "plan", "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["samples"][0][1]})
                qs[it["item_id"]] = P.parse_questions(r["samples"][0][0])
                rec["questions"] = qs[it["item_id"]]
            # 2. answer every question from the source alone (factored)
            ajobs = [(it, q) for it in chunk for q in qs.get(it["item_id"], [])]
            answers = {}
            if ajobs:
                res, _ = gen.generate([gen.fmt(P.cove_answer_prompt(it["prompt"], q)) for it, q in ajobs], 1, 0.0, 1.0,
                                      80, [E.seed_for(args.seed, "ans", it["item_id"], q) for it, q in ajobs])
                for (it, q), r in zip(ajobs, res):
                    a = r["samples"][0][0].strip() if r else "Not stated in the source."
                    answers.setdefault(it["item_id"], []).append((q, a))
                    if r:
                        recs[it["item_id"]]["calls"].append({"step": "answer", "prompt_tokens": r["prompt_tokens"],
                                                             "gen_tokens": r["samples"][0][1]})
            for it in chunk:  # prompt prefix shared by all CoVe calls of one item (prefix cache)
                fam = [gen.fmt(P.cove_answer_prompt(it["prompt"], q)) for q in qs.get(it["item_id"], [])[:2]]
                fam.append(gen.fmt(P.cove_plan_prompt(it["prompt"], draft_text(it, drafts))))
                recs[it["item_id"]]["shared_tokens"] = gen.n_tokens(os.path.commonprefix(fam))
            # 3. revise
            rjobs = [it for it in chunk if "skipped" not in recs[it["item_id"]]]
            if rjobs:
                res, _ = gen.generate([gen.fmt(P.cove_revise_prompt(it["prompt"], draft_text(it, drafts),
                                                                    answers.get(it["item_id"], []))) for it in rjobs],
                                      1, 0.0, 1.0, args.max_tokens, [E.seed_for(args.seed, "rev", it["item_id"]) for it in rjobs])
                for it, r in zip(rjobs, res):
                    rec = recs[it["item_id"]]
                    rec["qa"] = answers.get(it["item_id"], [])
                    if r is None:
                        rec["skipped"] = "context"
                        continue
                    rec["final"] = r["samples"][0][0].strip()
                    rec["calls"].append({"step": "revise", "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["samples"][0][1]})
            with open(cove_path, "a", encoding="utf-8") as fh:
                for rec in recs.values():
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"cove: {min(start + args.batch, len(todo))}/{len(todo)}")

    log("done")
    if args.backend == "vllm":
        E_hard_exit()


def E_hard_exit():
    import os

    try:
        import psutil

        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
