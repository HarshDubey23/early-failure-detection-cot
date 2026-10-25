"""
WHAT: The candidate outputs of every policy, with stable ids, built from the generation files.

WHY:  Scoring (detector D, evaluators E1/E2) and the offline replay of policies must refer to
      exactly the same texts. Every text any policy could output is a "candidate" with an id:

        <item_id>|draft            the first response
        <item_id>|k<k>|s<i>        bank: first k draft sentences + i-th sampled continuation
        <item_id>|delete           draft with the sentences D flags removed
        <item_id>|repair           draft with each D-flagged sentence rewritten (or removed)
        <item_id>|repairdel        repair, then the sentences D still flags removed
        <item_id>|cove             Chain-of-Verification final response
      The refusal ("Unable to answer based on given passages.") needs no scoring: it makes no
      claim, so it is never a hallucination and never covers the question.

HOW:  Pure functions over the JSONL files written by stages 31-33.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.recovery.prompts import REFUSAL
from src.recovery.ragtruth import sentence_spans


def read_jsonl(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def approx_tokens(text: str) -> int:
    """Used only where an exact count is unavailable (released RAGTruth drafts)."""
    return max(1, round(len(text) / 3.8))


def flagged(scores: list | None, tau: float) -> list:
    """1-based indices of sentences with detector score >= tau."""
    return [j + 1 for j, s in enumerate(scores or []) if s >= tau]


def drop_sentences(text: str, drop: list) -> str:
    """Remove the 1-based sentences in `drop`; whitespace is tidied."""
    spans = sentence_spans(text)
    keep = [text[a:b].strip() for j, (a, b) in enumerate(spans) if (j + 1) not in set(drop)]
    return " ".join(s for s in keep if s).strip()


def replace_sentences(text: str, rewrites: dict) -> str:
    """rewrites: {1-based j: new sentence or None (delete)}."""
    spans = sentence_spans(text)
    parts = []
    for j, (a, b) in enumerate(spans):
        if (j + 1) in rewrites:
            if rewrites[j + 1]:
                parts.append(rewrites[j + 1].strip())
        else:
            parts.append(text[a:b].strip())
    return " ".join(p for p in parts if p).strip()


def load_run(run_dir: str | Path, items_paths: list) -> dict:
    """Everything stage 32 wrote, keyed by item, for the items in `items_paths`."""
    run = Path(run_dir)
    items = {}
    for p in items_paths:
        for it in read_jsonl(p):
            items[it["item_id"]] = it
    gens = {it["generator"] for it in items.values()}
    drafts, bank, repair, cove = {}, {}, {}, {}
    for g in gens:
        for r in read_jsonl(run / f"drafts.{g}.jsonl"):
            if r.get("draft") is not None:
                drafts[r["item_id"]] = r
        for r in read_jsonl(run / f"bank.{g}.jsonl"):
            if not r.get("skipped"):
                bank.setdefault(r["item_id"], {})[r["k"]] = r
        for r in read_jsonl(run / f"repair.{g}.jsonl"):
            repair.setdefault(r["item_id"], {})[r["j"]] = r
        for r in read_jsonl(run / f"cove.{g}.jsonl"):
            cove[r["item_id"]] = r
    for iid, it in items.items():
        if it.get("draft") is not None:
            drafts[iid] = {"item_id": iid, "draft": it["draft"], "gen_tokens": approx_tokens(it["draft"]),
                           "prompt_tokens": approx_tokens(it["prompt"]) + 8, "released": True}
    return {"items": items, "drafts": drafts, "bank": bank, "repair": repair, "cove": cove}


def build_candidates(run: dict, d_scores: dict, tau: float, stage: int) -> list:
    """Candidate rows {cand_id, item_id, kind, text, prompt, ...}.

    stage 1: draft, bank, delete, repair, cove (needs D on drafts for delete/repair);
    stage 2: repairdel (needs D on the repair candidate as well).
    """
    rows = []
    for iid, it in run["items"].items():
        d = run["drafts"].get(iid)
        if d is None:
            continue
        draft = d["draft"]
        base = {"item_id": iid, "prompt": it["prompt"], "context": it.get("context"), "question": it.get("question")}
        if stage == 1:
            rows.append({**base, "cand_id": f"{iid}|draft", "kind": "draft", "text": draft})
            for k, b in sorted(run["bank"].get(iid, {}).items()):
                for i, cont in enumerate(b["conts"]):
                    rows.append({**base, "cand_id": f"{iid}|k{k}|s{i}", "kind": "bank", "k": k, "s": i,
                                 "text": (b["prefix"] + cont).strip()})
            if iid in run["cove"] and run["cove"][iid].get("final"):
                rows.append({**base, "cand_id": f"{iid}|cove", "kind": "cove", "text": run["cove"][iid]["final"]})
        ds = d_scores.get(f"{iid}|draft")
        fl = flagged(ds, tau) if ds is not None else []
        if not fl:
            continue
        if stage == 1:
            rows.append({**base, "cand_id": f"{iid}|delete", "kind": "delete", "text": drop_sentences(draft, fl)})
            reps = run["repair"].get(iid, {})
            if all(j in reps for j in fl):
                rw = {j: reps[j].get("rewrite") for j in fl}
                rows.append({**base, "cand_id": f"{iid}|repair", "kind": "repair", "text": replace_sentences(draft, rw)})
        else:
            rs = d_scores.get(f"{iid}|repair")
            reps = run["repair"].get(iid, {})
            if rs is None or not all(j in reps for j in fl):
                continue
            rep_text = replace_sentences(draft, {j: reps[j].get("rewrite") for j in fl})
            rows.append({**base, "cand_id": f"{iid}|repairdel", "kind": "repairdel",
                         "text": drop_sentences(rep_text, flagged(rs, tau))})
    return rows


def is_refusal_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return (not t) or (REFUSAL.lower().rstrip(".") in t and len(t.split()) <= 20)
