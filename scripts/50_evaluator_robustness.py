"""Stage 50 (CPU): which Part 3 conclusions survive a change of evaluator? (exploratory)

The recovery system gates on detector D (LettuceDetect-base). Evaluator E2 (LettuceDetect-large) is
trained on the same RAGTruth data, so it tends to pass whatever D passes. Any evaluator that needs
E2 to fire, including the "dual consensus" rule E1 AND E2>=0.3 of stage 43, therefore favours the
policies that resample until D is satisfied (restart_fb, backtrack_fb, CARE) over policies that do
not consult D (Chain-of-Verification). This stage:

1. Builds an evaluator that never reads a LettuceDetect score: the LLM judge E1 (Qwen2.5-7B) with a
   threshold on the share of sentences it calls unsupported, chosen on the DEV half of the 449
   human-labelled RAGTruth drafts and tested on the TEST half (same md5 split as stage 43).
   The threshold grid is fixed below, before looking at any policy output.
2. Scores every RAGTruth policy at budget 2 on the TEST half under five evaluators
   (E1 as preregistered, E1-calibrated, E2, the dual rule, and D itself), with paired source-level
   bootstrap differences against restart_fb.
3. Reports which conclusions keep their sign under every evaluator.

Note on stage 43: tau = 0.3 in the dual rule was chosen on all 449 drafts (commit 0ba841e) before
the DEV/TEST split was introduced, so its TEST kappa is not an out-of-sample estimate.

Output: results/recovery/final/evaluator_robustness.{json,md}
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery import stats as S  # noqa: E402
from src.recovery.ragtruth import sentence_spans  # noqa: E402

RUN = ROOT / "results" / "recovery" / "run"
FIN = ROOT / "results" / "recovery" / "final"
ECHO = re.compile(r"^(verification\b|q\s*:|a\s*:)", re.I)
FRACTION_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]   # E1 flags if share of unsupported sentences > f
POLICIES = [("base", 0.0), ("delete", 0.0), ("repairdel", 0.0), ("abstain_flag", 0.0),
            ("restart_fb", 2.0), ("backtrack_fb", 2.0), ("care", 2.0), ("care@restart", 2.0), ("cove_flag_clean", 0.0)]
REF = ("restart_fb", 2.0)


def split_of(source_id) -> str:
    return "dev" if int(hashlib.md5(str(source_id).encode()).hexdigest(), 16) % 2 == 0 else "test"


def kappa(p: np.ndarray, t: np.ndarray) -> float:
    p, t = np.asarray(p, int), np.asarray(t, int)
    po = np.mean(p == t)
    pe = p.mean() * t.mean() + (1 - p.mean()) * (1 - t.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def main() -> None:
    e1 = {}
    for line in open(RUN / "judge_E1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("unsupported") is not None:
            e1[r["cand_id"]] = (list(r["unsupported"]), int(r.get("n_sent") or 0))
    e2 = {json.loads(x)["cand_id"]: json.loads(x).get("sent_scores") or [0.0] for x in open(RUN / "scores_E2.jsonl", encoding="utf-8")}
    dd = {json.loads(x)["cand_id"]: json.loads(x).get("sent_scores") or [0.0] for x in open(RUN / "scores_D.jsonl", encoding="utf-8")}

    # CoVe: keep only the revised answer, before any echoed verification block (as in stage 40)
    cove_cut = {}
    for line in open(RUN / "cove.llama-2-7b-chat.jsonl", encoding="utf-8"):
        c = json.loads(line)
        if c.get("final"):
            sents = [c["final"][a:b].strip() for a, b in sentence_spans(c["final"])]
            cove_cut[c["item_id"] + "|cove"] = next((i for i, s in enumerate(sents) if ECHO.match(s)), len(sents))

    def e1_share(cid: str):
        if cid not in e1:
            return None
        bad, n = e1[cid]
        cut = cove_cut.get(cid)
        if cut is not None:
            bad, n = [k for k in bad if k <= cut], cut
        return (len(bad) / n) if n else 0.0

    def e2max(cid: str, src=e2):
        if cid not in src:
            return None
        sc = src[cid]
        cut = cove_cut.get(cid)
        if cut is not None:
            sc = sc[:cut]
        return max(sc or [0.0])

    def verdict(cid: str, ev: str, f_star: float):
        if cid in ("refuse", "") or cid.startswith("<REFUS") or "REFUSE" in cid:
            return 0.0
        s1, s2, sd = e1_share(cid), e2max(cid), e2max(cid, dd)
        if ev == "D":
            return None if sd is None else float(sd >= 0.5)
        if ev == "E2":
            return None if s2 is None else float(s2 >= 0.5)
        if s1 is None:                                   # preregistered fallback: E2 decides
            return None if s2 is None else float(s2 >= 0.5)
        if ev == "E1":
            return float(s1 > 0)
        if ev == "E1cal":
            return float(s1 > f_star)
        if ev == "dual":
            return float(s1 > 0 and s2 is not None and s2 >= 0.3)
        raise ValueError(ev)

    # ---- 1. calibrate E1 on DEV drafts, test on TEST drafts --------------------------------
    lab = pd.read_csv(ROOT / "results/recovery/ragtruth_labels/responses.csv")
    lab = lab[(lab.model == "llama-2-7b-chat") & (lab.split == "test") & (lab.quality == "good")].copy()
    lab["cid"] = [f"ragtruth:{s}:llama-2-7b-chat|draft" for s in lab.source_id]
    lab = lab[lab.cid.isin(e1.keys())].copy()
    lab["split"] = lab.source_id.map(split_of)
    lab["share"] = lab.cid.map(e1_share)
    lab["e2"] = lab.cid.map(e2max)
    lab["d"] = lab.cid.map(lambda c: e2max(c, dd))
    dev, test = lab[lab.split == "dev"], lab[lab.split == "test"]
    grid = {f: kappa(dev.share > f, dev.halluc) for f in FRACTION_GRID}
    f_star = max(grid, key=grid.get)
    cal = {
        "fraction_grid_dev_kappa": grid, "f_star": f_star,
        "n_dev": len(dev), "n_test": len(test),
        "test": {
            "E1 (any sentence)": kappa(test.share > 0, test.halluc),
            f"E1-calibrated (share > {f_star})": kappa(test.share > f_star, test.halluc),
            "E2 (LettuceDetect-large >= 0.5)": kappa(test.e2 >= 0.5, test.halluc),
            "dual (E1 and E2 >= 0.3; tau chosen on all drafts)": kappa((test.share > 0) & (test.e2 >= 0.3), test.halluc),
            "D (system detector >= 0.5)": kappa(test.d >= 0.5, test.halluc),
        },
        "test_predicted_rate": {
            "E1-calibrated": float((test.share > f_star).mean()), "human": float(test.halluc.mean()),
        },
        "agreement_with_D_on_all_drafts": {
            "E2": float(np.mean((lab.e2 >= 0.5) == (lab.d >= 0.5))),
            "E1-calibrated": float(np.mean((lab.share > f_star) == (lab.d >= 0.5))),
        },
    }

    # ---- 2. score policies on TEST items under every evaluator -----------------------------
    per = pd.read_csv(FIN / "per_item.csv")
    per = per[per.benchmark == "ragtruth"]
    outs = pd.read_csv(FIN / "outputs_round0.csv")
    per = per.merge(outs[["item_id", "policy", "budget", "cid"]], on=["item_id", "policy", "budget"], how="left")
    per["cid"] = per.cid.fillna("")
    per["split"] = per.source_id.map(split_of)
    base = per[(per.policy == "base")].set_index("item_id")
    flagged = base.flagged.astype(bool)

    rows = []
    for _, r in per[per.split == "test"].iterrows():
        rows.append({"item_id": r.item_id, "policy": r.policy, "budget": r.budget, "cid": r.cid,
                     "covered": r.covered, "cost": r.cost_total})
    # CoVe applied to flagged drafts only (the deployed configuration); unflagged drafts pass unchanged
    cove_ids = {k.split("|")[0] for k in cove_cut}
    for iid in base[(base.split == "test")].index:
        if iid in cove_ids:
            rows.append({"item_id": iid, "policy": "cove_flag_clean", "budget": 0.0,
                         "cid": iid + "|cove" if flagged[iid] else base.loc[iid, "cid"], "covered": 1.0, "cost": np.nan})
    P = pd.DataFrame(rows)
    EVS = ["E1", "E1cal", "E2", "dual", "D"]
    for ev in EVS:
        P[ev] = [verdict(c, ev, f_star) for c in P.cid]

    table, diffs = {}, {}
    for pol, b in POLICIES:
        sub = P[(P.policy == pol) & (P.budget == b)].set_index("item_id")
        ref = P[(P.policy == REF[0]) & (P.budget == REF[1])].set_index("item_id")
        common = sub.index.intersection(ref.index)
        rec = {"n": len(sub), "coverage": float(sub.covered.mean()),
               "cost": None if sub.cost.isna().all() else float(sub.cost.mean())}
        for ev in EVS:
            rec[ev] = float(np.nanmean(sub[ev]))
            rec[ev + "_faithful_delivered"] = float(np.nanmean(sub.covered * (1 - sub[ev])))
            dv = (sub.loc[common, ev] - ref.loc[common, ev]).to_numpy(dtype=float)
            diffs[(pol, ev)] = S.boot_ci(dv, np.arange(len(dv)))
            rec[ev + "_minus_restart"] = diffs[(pol, ev)]
        table[f"{pol}@{b:g}"] = rec

    # ---- 3. which conclusions are evaluator-invariant? -----------------------------------
    def sign(ci):
        m, lo, hi = ci
        return "+" if lo > 0 else "-" if hi < 0 else "0"
    invariance = {}
    for pol, b in POLICIES:
        if (pol, b) == REF:
            continue
        signs = {ev: sign(diffs[(pol, ev)]) for ev in EVS if not np.isnan(diffs[(pol, ev)][0])}
        lofi = {ev: s for ev, s in signs.items() if ev in ("E1", "E1cal")}
        hifi = {ev: s for ev, s in signs.items() if ev in ("E2", "dual", "D")}
        invariance[f"{pol}@{b:g}"] = {"signs": signs,
                                      "invariant": len(set(signs.values())) == 1,
                                      "D_independent_evaluators": lofi, "D_correlated_evaluators": hifi}
    out = {"calibration": cal, "policies_test_split": table, "invariance_vs_restart_fb": invariance}
    (FIN / "evaluator_robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    names = {"E1": "E1 judge", "E1cal": "E1 calibrated", "E2": "E2", "dual": "dual rule", "D": "D (system's own detector)"}
    L = ["# Part 3 under five evaluators (RAGTruth, TEST half of the sources)", "",
         f"E1-calibrated flags an output when the judge calls more than {f_star:.0%} of its sentences unsupported; "
         f"the threshold was chosen on the {len(dev)} DEV drafts (grid fixed in advance) and never reads a "
         "LettuceDetect score. Human-agreement kappa on the "
         f"{len(test)} TEST drafts: " + "; ".join(f"{k}: {v:.3f}" for k, v in cal["test"].items()) + ".", "",
         "E2, the dual rule and D share LettuceDetect's training data with the detector the system resamples "
         "against, so they are expected to favour detector-gated policies.", "",
         "Hallucination rate (lower is better), and difference from restart_fb at budget 2 with 95% bootstrap CI:", "",
         "| Policy | coverage | tokens | " + " | ".join(names[e] for e in EVS) + " |",
         "|---|---|---|" + "---|" * len(EVS)]
    for key, rec in table.items():
        cells = []
        for ev in EVS:
            m, lo, hi = rec[ev + "_minus_restart"]
            d = "" if key == f"{REF[0]}@{REF[1]:g}" else f" ({m:+.3f} [{lo:+.3f}, {hi:+.3f}])"
            cells.append(f"{rec[ev]:.3f}{d}")
        cost = "–" if rec["cost"] is None else f"{rec['cost']:,.0f}"
        L.append(f"| {key} (n={rec['n']}) | {rec['coverage']:.3f} | {cost} | " + " | ".join(cells) + " |")
    L += ["", "Sign of the difference from restart_fb under each evaluator (+ worse than restart, − better, 0 no "
          "difference detected):", "", "| Policy | " + " | ".join(names[e] for e in EVS) + " | same under all? |",
          "|---|" + "---|" * (len(EVS) + 1)]
    for key, inv in invariance.items():
        L.append(f"| {key} | " + " | ".join(inv["signs"].get(e, "–") for e in EVS) + f" | {'yes' if inv['invariant'] else 'no'} |")
    (FIN / "evaluator_robustness.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
