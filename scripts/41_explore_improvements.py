"""Stage 41 (CPU, EXPLORATORY): which changes to the recovery system look promising?

Uses only Run R outputs that already exist (no model is run), to choose what to preregister for
the next GPU run. Items are split 50/50 by source id: candidates are compared on the DEV half, and
the TEST half is reported once, as a check that the ranking holds. Nothing here is a confirmatory
result: the same items were used to design the analysis.

Candidates (all at budget b = 2 where a budget applies):
  restart_fb          the best fixed strategy from Run R (reference)
  care                the Run R controller
  task_aware          QA: abstain as soon as the detector flags the draft; other tasks: restart_fb
  cove_flag_clean     flagged drafts -> CoVe, scored on the revised answer before the echoed
                      verification block; unflagged drafts kept (192 RAGTruth items only)
  restart_fb_b4       restart_fb with twice the budget
Metrics (judge E1): hallucinated, coverage, and faithful_answered = answered and not hallucinated.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import stats as S  # noqa: E402
from src.recovery.ragtruth import sentence_spans  # noqa: E402

ECHO = re.compile(r"^(verification\b|q\s*:|a\s*:)", re.I)


def cove_clean() -> dict:
    judge = {}
    for line in open("results/recovery/run/judge_E1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("unsupported") is not None:
            judge[r["cand_id"]] = r["unsupported"]
    out = {}
    for line in open("results/recovery/run/cove.llama-2-7b-chat.jsonl", encoding="utf-8"):
        c = json.loads(line)
        if c.get("final") and c["item_id"] + "|cove" in judge:
            sents = [c["final"][a:b].strip() for a, b in sentence_spans(c["final"])]
            cut = next((i for i, s in enumerate(sents) if ECHO.match(s)), len(sents))
            out[c["item_id"]] = float(any(k <= cut for k in judge[c["item_id"] + "|cove"])) if cut else 0.0
    return out


def main() -> None:
    P = pd.read_csv("results/recovery/final/per_item.csv")
    P = P[P.benchmark == "ragtruth"]
    pick = lambda pol, b: P[(P.policy == pol) & (P.budget == b)].set_index("item_id")  # noqa: E731
    base, rf, rf4, care, abst = pick("base", 0), pick("restart_fb", 2), pick("restart_fb", 4), pick("care", 2), pick("abstain_flag", 0)
    cove = cove_clean()
    rows = []
    for iid in base.index:
        b = base.loc[iid]
        split = "dev" if int(hashlib.md5(str(b.source_id).encode()).hexdigest(), 16) % 2 == 0 else "test"
        cand = {"base": base.loc[iid], "restart_fb": rf.loc[iid], "care": care.loc[iid], "restart_fb_b4": rf4.loc[iid],
                "task_aware": abst.loc[iid] if b.task == "QA" else rf.loc[iid]}
        for name, r in cand.items():
            rows.append({"item_id": iid, "split": split, "task": b.task, "policy": name, "hall": r.hall_E1,
                         "covered": r.covered, "cost": r.cost_total})
        if iid in cove:
            fl = bool(b.flagged)
            rows.append({"item_id": iid, "split": split, "task": b.task, "policy": "cove_flag_clean",
                         "hall": cove[iid] if fl else b.hall_E1, "covered": 1.0, "cost": np.nan})
    df = pd.DataFrame(rows)
    df["faithful_answered"] = df.covered * (1 - df.hall)
    report = {}
    for split, sub in df.groupby("split"):
        ref = sub[sub.policy == "restart_fb"].set_index("item_id")
        for pol, g in sub.groupby("policy"):
            g = g.set_index("item_id")
            common = g.index.intersection(ref.index)
            d = (g.loc[common, "hall"] - ref.loc[common, "hall"]).to_numpy()
            report[f"{split}/{pol}"] = {"n": int(len(g)), "hall": round(float(g.hall.mean()), 3),
                                        "covered": round(float(g.covered.mean()), 3),
                                        "faithful_answered": round(float(g.faithful_answered.mean()), 3),
                                        "cost": round(float(g.cost.mean()), 0) if g.cost.notna().any() else None,
                                        "hall_minus_restart_fb": [round(x, 3) for x in S.boot_ci(d, np.arange(len(d)))]}
    json.dump(report, open("results/recovery/final/explore_improvements.json", "w"), indent=2)
    for k, v in report.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
