"""Stage 42 (CPU, EXPLORATORY): how far do the detector and the judge agree with people?

Run R's primary evaluator (the LLM judge E1) passed its preregistered validity bar only narrowly
(kappa 0.22). This stage asks what that means for the size of the measured reduction:
  - on the 450 released drafts (human labels), how often is a draft hallucinated by human judgement
    when the detector D passes it, and when the judge E1 passes it?
  - on the 7k judged resampled continuations (no human labels), how often does E1 flag a
    continuation that D passes?
If people find far fewer hallucinations in D-passed text than E1 does, E1 understates the
reduction that a D-gated system achieves. Only the drafts have human labels, so this is an
inference, not a measurement of the system's outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import stats as S  # noqa: E402


def rd(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main() -> None:
    run = Path("results/recovery/run")
    D = {r["cand_id"]: max(r["sent_scores"] or [0]) for r in rd(run / "scores_D.jsonl")}
    J = {r["cand_id"]: len(r["unsupported"]) > 0 for r in rd(run / "judge_E1.jsonl") if r.get("unsupported") is not None}
    items = {r["item_id"]: r for r in rd("data/recovery/items.ragtruth.llama-2-7b-chat.jsonl")}
    drafts = [(i, len(it["human_labels"]) > 0) for i, it in items.items() if it["draft_quality"] == "good"]

    def share(xs):
        xs = np.asarray(xs, dtype=float)
        return S.boot_ci(xs, np.arange(len(xs)), n_boot=2000) + (int(len(xs)),)

    out = {
        "human_hallucinated_given_D_passes_draft": share([h for i, h in drafts if D[i + "|draft"] < 0.5]),
        "human_hallucinated_given_D_flags_draft": share([h for i, h in drafts if D[i + "|draft"] >= 0.5]),
        "human_hallucinated_given_E1_passes_draft": share([h for i, h in drafts if i + "|draft" in J and not J[i + "|draft"]]),
        "human_hallucinated_given_E1_flags_draft": share([h for i, h in drafts if i + "|draft" in J and J[i + "|draft"]]),
        "E1_flags_given_D_passes_continuation": share([J[c] for c in J if "|k" in c and c in D and D[c] < 0.5]),
        "E1_flags_given_D_flags_continuation": share([J[c] for c in J if "|k" in c and c in D and D[c] >= 0.5]),
    }
    Path("results/recovery/final").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/recovery/final/evaluator_ceiling.json", "w"), indent=2)
    for k, v in out.items():
        print(f"{k:45s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] n={v[3]}")


if __name__ == "__main__":
    main()
