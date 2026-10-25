"""Stage 31 (CPU, needs network for HotpotQA): benchmark items for the recovery experiment.

Writes data/recovery/items.<benchmark>.<generator>.jsonl, one row per input:

  ragtruth  RAGTruth test sources (450: 150 QA, 150 Summary, 150 Data2txt). For a RAGTruth
            generator with open weights (llama-2-7b-chat) the draft IS the released response,
            with its human hallucination spans, so the base model's hallucination rate is
            measured by people, not by a judge. For any other generator the draft is generated
            in stage 32.
  hotpot    Controlled evidence sufficiency on HotpotQA (distractor dev set). The same
            question appears twice: once with both gold paragraphs among 6 passages
            (answerable) and once with 6 distractor passages only (insufficient evidence; the
            correct output is the refusal). Yes/no questions are excluded; answerable items need
            the gold answer string in their passages and unanswerable items must not contain it.

Usage:
  python scripts/31_build_items.py --benchmark ragtruth --generator llama-2-7b-chat
  python scripts/31_build_items.py --benchmark hotpot --generator llama-3.1-8b-instruct --n-questions 150
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import prompts as P  # noqa: E402
from src.recovery import ragtruth as rt  # noqa: E402

GENERATORS = {  # name -> open weights (ungated copies of the original releases)
    "llama-2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
    "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral-7B-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}
ON_POLICY_RAGTRUTH = {"llama-2-7b-chat", "llama-2-13b-chat", "mistral-7B-instruct"}


def norm(s: str) -> str:
    """SQuAD-style answer normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def ragtruth_items(root: str, generator: str) -> list:
    resp, src = rt.load(root)
    by_src = {}
    for r in resp:
        if r["split"] == "test" and r["model"] == generator:
            by_src[r["source_id"]] = r
    items = []
    for sid in sorted({r["source_id"] for r in resp if r["split"] == "test"}, key=int):
        s = src[sid]
        draft = by_src.get(sid) if generator in ON_POLICY_RAGTRUTH else None
        items.append({
            "item_id": f"ragtruth:{sid}:{generator}", "benchmark": "ragtruth", "source_id": sid,
            "task": s["task_type"], "generator": generator, "prompt": s["prompt"],
            "question": rt.question_text(s), "context": rt.context_text(s),
            "draft": draft["response"] if draft else None,
            "draft_temperature": draft["temperature"] if draft else None,
            "draft_quality": draft["quality"] if draft else None,
            "human_labels": draft["labels"] if draft else None,
            "gold": None, "answerable": None, "is_qa": s["task_type"] == "QA"})
    return items


def load_hotpot(log) -> list:
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        rows = []
        for r in ds:
            ctx = list(zip(r["context"]["title"], r["context"]["sentences"]))
            rows.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                         "level": r["level"], "context": ctx, "sup_titles": sorted(set(r["supporting_facts"]["title"]))})
        log(f"hotpot: {len(rows)} rows from the Hugging Face hub")
        return rows
    except Exception as exc:  # noqa: BLE001
        log(f"hotpot via datasets failed ({exc!r}); trying the original release")
    import urllib.request

    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    data = json.loads(urllib.request.urlopen(url, timeout=120).read())
    rows = [{"id": r["_id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
             "level": r["level"], "context": [(t, s) for t, s in r["context"]],
             "sup_titles": sorted({t for t, _ in r["supporting_facts"]})} for r in data]
    log(f"hotpot: {len(rows)} rows from {url}")
    return rows


def synthetic_hotpot(n: int = 30) -> list:
    """Fake HotpotQA-shaped rows for the CPU pipeline test (MOCK=1). Never a result."""
    rows = []
    for i in range(n):
        titles = [f"T{i}_{j}" for j in range(10)]
        ctx = [(t, [f"{t} is about something else. "]) for t in titles]
        ctx[0] = (titles[0], [f"{titles[0]} mentions Answer{i} clearly. "])
        rows.append({"id": f"synthetic{i}", "question": f"What does T{i}_0 mention?", "answer": f"Answer{i}",
                     "type": "bridge", "level": "easy", "context": ctx, "sup_titles": [titles[0], titles[1]]})
    return rows


def hotpot_items(rows: list, generator: str, n_questions: int, seed: int, n_passages: int = 6) -> list:
    rng = random.Random(seed)
    rows = [r for r in rows if norm(r["answer"]) not in {"yes", "no"} and len(r["sup_titles"]) == 2]
    rng.shuffle(rows)
    items = []
    for r in rows:
        if len(items) >= 2 * n_questions:
            break
        para = {t: (t + ": " + "".join(s).strip()) for t, s in r["context"]}
        gold_p = [para[t] for t in r["sup_titles"] if t in para]
        distract = [p for t, p in para.items() if t not in r["sup_titles"]]
        if len(gold_p) != 2 or len(distract) < n_passages:
            continue
        rng.shuffle(distract)
        ans_pass = gold_p + distract[: n_passages - 2]
        una_pass = distract[: n_passages]
        rng.shuffle(ans_pass)
        rng.shuffle(una_pass)
        a = norm(r["answer"])
        if a not in norm(" ".join(ans_pass)) or a in norm(" ".join(una_pass)):
            continue
        for answerable, passages in [(True, ans_pass), (False, una_pass)]:
            items.append({
                "item_id": f"hotpot:{r['id']}:{'ans' if answerable else 'unans'}:{generator}",
                "benchmark": "hotpot", "source_id": r["id"], "task": "QA", "generator": generator,
                "prompt": P.qa_prompt(r["question"], passages), "question": r["question"],
                "context": "\n\n".join(passages), "draft": None, "draft_temperature": None, "draft_quality": None,
                "human_labels": None, "gold": [r["answer"]], "answerable": answerable, "is_qa": True,
                "hotpot_type": r["type"], "hotpot_level": r["level"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="31: build benchmark items")
    ap.add_argument("--benchmark", choices=["ragtruth", "hotpot"], required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--ragtruth", default="data/external/RAGTruth")
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="keep the first N items (smoke test); 0 = all")
    ap.add_argument("--synthetic", action="store_true", help="hotpot: fake rows, for the CPU pipeline test only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="data/recovery")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    if args.benchmark == "ragtruth":
        items = ragtruth_items(args.ragtruth, args.generator)
    else:
        rows = synthetic_hotpot() if args.synthetic else load_hotpot(log)
        items = hotpot_items(rows, args.generator, args.n_questions, args.seed)
    if args.limit:  # keep tasks / conditions balanced in a smoke test
        by = {}
        for it in items:
            by.setdefault((it["task"], it["answerable"]), []).append(it)
        per = max(1, args.limit // len(by))
        items = [it for group in by.values() for it in group[:per]]
    path = out / f"items.{args.benchmark}.{args.generator}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_draft = sum(it["draft"] is not None for it in items)
    log(f"wrote {len(items)} items to {path} ({n_draft} with released drafts)")


if __name__ == "__main__":
    main()
