"""Stage 33 (GPU): score candidate outputs with the system detector D and the evaluators E2, E1.

  D   KRLabsOrg/lettucedect-base-modernbert-en-v1   the detector the system USES (flags, early stop)
  E2  KRLabsOrg/lettucedect-large-modernbert-en-v1  secondary evaluator (same family as D: optimistic
                                                    for D-selected outputs; reported as such)
  E1  LLM judge, Qwen2.5-7B-Instruct by default      primary evaluator: a different model family from
                                                    the generators and from D, validated against the
                                                    human RAGTruth labels on the released drafts

Which candidates are scored (see src/recovery/candidates.py for the ids):
  --kinds draft            drafts only (needed before stage 32 can build the bank)
  --stage 1                drafts, bank, delete, repair, CoVe
  --stage 2                repair-then-delete (needs D scores of the repair candidates)
E1 scores every non-bank candidate, every bank sample at a system point, and a deterministic 15%
of the other bank samples (mechanism-study boundaries) to calibrate E2 against E1 there.

Usage:
  python scripts/33_score.py --scorer D --items data/recovery/items.*.jsonl --kinds draft
  python scripts/33_score.py --scorer E1 --items ... --stage 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import candidates as C  # noqa: E402
from src.recovery import engine as E  # noqa: E402
from src.recovery.ragtruth import sentence_labels  # noqa: E402
from src.recovery.replay import system_points  # noqa: E402

DETECTORS = {"D": "KRLabsOrg/lettucedect-base-modernbert-en-v1", "E2": "KRLabsOrg/lettucedect-large-modernbert-en-v1"}


def load_scores(path) -> dict:
    return {r["cand_id"]: r for r in C.read_jsonl(path) if r.get("sent_scores") is not None or "unsupported" in r}


def main() -> None:
    ap = argparse.ArgumentParser(description="33: score candidates")
    ap.add_argument("--scorer", choices=["D", "E2", "E1"], required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--run-dir", default="results/recovery/run")
    ap.add_argument("--kinds", nargs="*", default=None, help="restrict to these candidate kinds")
    ap.add_argument("--stage", type=int, nargs="*", default=[1])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-quant", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--backend", choices=["real", "mock"], default="real")
    ap.add_argument("--calib-share", type=float, default=0.15)
    ap.add_argument("--time-limit", type=float, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    t_start = time.time()
    out_path = run_dir / (f"scores_{args.scorer}.jsonl" if args.scorer != "E1" else "judge_E1.jsonl")
    log_path = run_dir / f"score_{args.scorer}.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    run = C.load_run(run_dir, args.items)
    d_scores = {k: v["sent_scores"] for k, v in load_scores(run_dir / "scores_D.jsonl").items()}
    cands = []
    for st in args.stage:
        cands += C.build_candidates(run, d_scores, args.tau, st)
    if args.kinds:
        cands = [c for c in cands if c["kind"] in args.kinds]
    if args.scorer == "E1":
        human_e = {}  # first human-labelled hallucinated sentence (1-based) of each released draft
        for iid, it in run["items"].items():
            if it.get("human_labels") and iid in run["drafts"]:
                lab = sentence_labels(run["drafts"][iid]["draft"], it["human_labels"])
                if any(lab):
                    human_e[iid] = lab.index(1) + 1
        keep = []
        for c in cands:
            if c["kind"] != "bank":
                keep.append(c)
                continue
            ds = d_scores.get(f"{c['item_id']}|draft")
            e = human_e.get(c["item_id"])
            if ds is not None and c["k"] in system_points(ds, args.tau):
                keep.append(c)
            elif e is not None and c["k"] in (0, e - 1):  # mechanism study: restart vs before the human error
                keep.append(c)
            elif int(hashlib.md5(c["cand_id"].encode()).hexdigest()[:6], 16) / 0xFFFFFF < args.calib_share:
                keep.append(c)
        cands = keep
    done = set(load_scores(out_path))
    todo = [c for c in cands if c["cand_id"] not in done]
    log(f"{args.scorer}: {len(cands)} candidates, {len(todo)} to score")
    if not todo:
        return

    if args.backend == "mock":
        scorer = E.MockJudge() if args.scorer == "E1" else E.MockDetector(noise=0.15 if args.scorer == "D" else 0.08,
                                                                             seed=1 if args.scorer == "D" else 2)
    elif args.scorer == "E1":
        scorer = E.VLLMJudge(args.judge_model, tp=args.tp, quantization=args.judge_quant)
    else:
        scorer = E.LettuceDetector(DETECTORS[args.scorer], device=args.device)
    model_name = getattr(scorer, "model_path", getattr(scorer, "model_id", "?"))

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.batch):
            if args.time_limit and time.time() - t_start > args.time_limit:
                log(f"time limit reached after {start}/{len(todo)}")
                break
            chunk = todo[start: start + args.batch]
            res = scorer.judge(chunk) if args.scorer == "E1" else scorer.score_batch(chunk)
            for c, r in zip(chunk, res):
                row = {"cand_id": c["cand_id"], "item_id": c["item_id"], "kind": c["kind"], "scorer": args.scorer,
                       "model": model_name, "backend": scorer.backend,
                       "det_tokens": C.approx_tokens(c["prompt"]) + C.approx_tokens(c["text"])}
                row.update(r)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            log(f"{args.scorer}: {start + len(chunk)}/{len(todo)} scored, {time.time() - t0:.0f}s")
    log("done")
    if args.scorer == "E1" and args.backend == "real":
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
