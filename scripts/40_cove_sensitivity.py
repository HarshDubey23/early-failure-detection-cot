"""Stage 40 (CPU): sensitivity of the CoVe baseline to a formatting failure (Run R, exploratory).

Llama-2-7B-chat often copies the "VERIFICATION: Q: ... A: ..." block of the CoVe prompt into its
final response, and the judge marks those question/answer lines as unsupported. This stage re-reads
the judge's existing per-sentence verdicts and counts only the sentences before the echoed block
(the revised response proper). No model is re-run, so the judge still saw the full text.

  python scripts/40_cove_sensitivity.py
"""
from __future__ import annotations

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


def main() -> None:
    run = Path("results/recovery/run")
    judge = {}
    for line in open(run / "judge_E1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("unsupported") is not None:
            judge[r["cand_id"]] = r
    per = pd.read_csv("results/recovery/final/per_item.csv")
    rows = []
    for line in open(run / "cove.llama-2-7b-chat.jsonl", encoding="utf-8"):
        c = json.loads(line)
        j = judge.get(c["item_id"] + "|cove")
        if not c.get("final") or j is None:
            continue
        sents = [c["final"][a:b].strip() for a, b in sentence_spans(c["final"])]
        cut = next((i for i, s in enumerate(sents) if ECHO.match(s)), len(sents))
        bad = [k for k in j["unsupported"] if k <= cut]
        rows.append({"item_id": c["item_id"], "echoed": cut < len(sents), "hall_full": float(bool(j["unsupported"])),
                     "hall_before_echo": float(bool(bad)) if cut > 0 else 0.0})
    df = pd.DataFrame(rows)
    care = per[(per.policy == "care") & (per.budget == 2)].set_index("item_id").hall_E1
    base = per[per.policy == "base"].set_index("item_id").hall_E1
    df["care"], df["base"] = df.item_id.map(care), df.item_id.map(base)
    df = df.dropna()
    cl = np.arange(len(df))
    out = {"n": len(df), "share_echoing_verification_block": float(df.echoed.mean()),
           "base": S.boot_ci(df.base, cl), "care_b2": S.boot_ci(df.care, cl),
           "cove_full_output": S.boot_ci(df.hall_full, cl), "cove_before_echo": S.boot_ci(df.hall_before_echo, cl),
           "care_minus_cove_before_echo": S.boot_ci((df.care - df.hall_before_echo).to_numpy(), cl),
           "p_signflip": S.signflip_p((df.care - df.hall_before_echo).to_numpy(), cl)}
    Path("results/recovery/final").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/recovery/final/cove_sensitivity.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
