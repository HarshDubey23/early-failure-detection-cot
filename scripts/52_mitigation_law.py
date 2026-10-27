"""Stage 52 (CPU): the detect-restart-abstain law, checked against the data (exploratory).

Proposition 3 of the paper (Appendix): if a system draws independent answers and returns the first
one its detector passes, the share of returned answers that are hallucinated is

    pi = h * beta / (h * beta + (1 - h) * (1 - alpha))      (independent of the retry budget)

with h the generator's hallucination rate, beta the detector's miss rate and alpha its false-alarm
rate. pi is simply P(hallucinated | detector passes), measurable on human-labelled drafts. This stage

1. estimates pi for the system's detector D on the 449 human-labelled RAGTruth drafts;
2. checks it a second, independent way: the LLM judge E1 never reads D, so inverting E1's own error
   rates (measured on the same drafts) turns E1's flag rate on D-passed continuations into a
   human-scale hallucination rate; the same inversion on D-flagged continuations is the control;
3. collects the cost of every RAG mitigation policy relative to the base model, and how many failed
   maths solutions plain restart recovers at each budget.

Output: results/recovery/final/mitigation_law.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "results" / "recovery" / "run"
FIN = ROOT / "results" / "recovery" / "final"
N_BOOT = 2000


def main() -> None:
    rng = np.random.default_rng(52)
    e1 = {}
    for line in open(RUN / "judge_E1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("unsupported") is not None:
            e1[r["cand_id"]] = (r["kind"], len(r["unsupported"]) > 0)
    dmax = {json.loads(x)["cand_id"]: max(json.loads(x).get("sent_scores") or [0.0])
            for x in open(RUN / "scores_D.jsonl", encoding="utf-8")}

    lab = pd.read_csv(ROOT / "results/recovery/ragtruth_labels/responses.csv")
    lab = lab[(lab.model == "llama-2-7b-chat") & (lab.split == "test") & (lab.quality == "good")].copy()
    lab["cid"] = [f"ragtruth:{s}:llama-2-7b-chat|draft" for s in lab.source_id]
    lab = lab[lab.cid.isin(e1.keys()) & lab.cid.isin(dmax.keys())]
    H = lab.halluc.to_numpy(int)
    Dflag = np.array([dmax[c] >= 0.5 for c in lab.cid], int)
    E1flag = np.array([e1[c][1] for c in lab.cid], int)

    cont = [(dmax[c] >= 0.5, f) for c, (kind, f) in e1.items() if kind == "bank" and c in dmax]
    cD = np.array([d for d, _ in cont], int)
    cE = np.array([f for _, f in cont], int)

    def stats(idx_d: np.ndarray, idx_c: np.ndarray) -> dict:
        h, df, ef = H[idx_d], Dflag[idx_d], E1flag[idx_d]
        alpha = df[h == 0].mean()                   # D false alarm
        beta = 1 - df[h == 1].mean()                # D miss
        pi = h[df == 0].mean()                      # P(halluc | D passes) = Proposition 3
        tpr, fpr = ef[h == 1].mean(), ef[h == 0].mean()
        cd, ce = cD[idx_c], cE[idx_c]
        f_pass, f_flag = ce[cd == 0].mean(), ce[cd == 1].mean()
        inv = lambda f: (f - fpr) / (tpr - fpr)     # noqa: E731  human-scale rate implied by E1's flag rate
        return {"h": h.mean(), "alpha": alpha, "beta": beta, "q": df.mean(), "pi": pi,
                "pi_formula": h.mean() * beta / (h.mean() * beta + (1 - h.mean()) * (1 - alpha)),
                "E1_tpr": tpr, "E1_fpr": fpr, "E1_flag_D_passed_cont": f_pass, "E1_flag_D_flagged_cont": f_flag,
                "implied_halluc_D_passed_cont": inv(f_pass), "implied_halluc_D_flagged_cont": inv(f_flag),
                "reduction_vs_base": h.mean() - pi}

    point = stats(np.arange(len(H)), np.arange(len(cD)))
    boots = [stats(rng.integers(0, len(H), len(H)), rng.integers(0, len(cD), len(cD))) for _ in range(N_BOOT)]
    law = {k: [float(v), float(np.quantile([b[k] for b in boots], 0.025)), float(np.quantile([b[k] for b in boots], 0.975))]
           for k, v in point.items()}
    law["n_drafts"], law["n_continuations"] = int(len(H)), int(len(cD))
    assert abs(point["pi"] - point["pi_formula"]) < 1e-9   # Bayes identity: the law and the direct estimate agree

    s = pd.read_csv(FIN / "summary.csv")
    s = s[(s.benchmark == "ragtruth") & (s.scope == "all")]
    base_cost = float(s[s.policy == "base"].cost_total.iloc[0])
    pols = [("base", 0, "Base model (no mitigation)"), ("delete", 0, "Detect, delete flagged sentences"),
            ("repairdel", 0, "Detect, repair claims"), ("restart_fb", 2, "Detect, restart, abstain (DRA)"),
            ("backtrack_fb", 2, "Detect, backtrack, abstain"), ("care", 2, "CARE (choke-point controller)"),
            ("abstain_flag", 0, "Detect, abstain on QA"), ("cove_all", 0, "Chain-of-Verification (n=192)")]
    cost = []
    for pol, b, name in pols:
        r = s[(s.policy == pol) & (s.budget == b)].iloc[0]
        cost.append({"policy": pol, "name": name, "tokens": float(r.cost_total), "cost_vs_base": float(r.cost_total) / base_cost,
                     "hall_E1": float(r.hall_E1), "hall_E2": float(r.hall_E2), "coverage": float(r.covered)})

    p = pd.read_csv(ROOT / "results/commitment/policies/policy_summary.csv")
    p = p[(p.policy == "P0_restart") & (p.accounting == "flops") & (p.population == "all_fatal")]
    maths = {g: {str(int(b)): [float(r.success), float(r.success_lo), float(r.success_hi)]
                 for b, r in sub.set_index("budget").iterrows()} for g, sub in p.groupby("generator")}

    out = {"law": law, "cost": cost, "maths_restart_recovery": maths}
    (FIN / "mitigation_law.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    f = lambda k, pct=True: (f"{100 * law[k][0]:.1f}% [{100 * law[k][1]:.1f}, {100 * law[k][2]:.1f}]" if pct  # noqa: E731
                             else f"{law[k][0]:.3f} [{law[k][1]:.3f}, {law[k][2]:.3f}]")
    L = ["# Detect-restart-abstain: the law and the data (exploratory)", "",
         f"Human-labelled RAGTruth drafts: n = {law['n_drafts']}; resampled continuations judged by E1: n = {law['n_continuations']}.", "",
         "| Quantity | Estimate [95% CI] |", "|---|---|",
         f"| base hallucination rate h (people) | {f('h')} |",
         f"| detector D: false-alarm rate alpha | {f('alpha')} |",
         f"| detector D: miss rate beta | {f('beta')} |",
         f"| **predicted hallucination among delivered answers, pi** | **{f('pi')}** |",
         f"| predicted reduction, h - pi | {f('reduction_vs_base')} |",
         f"| check: E1-implied human-scale rate, D-passed continuations | {f('implied_halluc_D_passed_cont')} |",
         f"| control: E1-implied rate, D-flagged continuations | {f('implied_halluc_D_flagged_cont')} |", "",
         "Cost of each RAG mitigation (RAGTruth, 450 sources; tokens per item, all calls counted):", "",
         "| Policy | tokens | cost vs base | E1 | E2 | coverage |", "|---|---|---|---|---|---|"]
    for c in cost:
        L.append(f"| {c['name']} | {c['tokens']:,.0f} | {c['cost_vs_base']:.2f}x | {c['hall_E1']:.3f} | {c['hall_E2']:.3f} | {c['coverage']:.3f} |")
    L += ["", "Failed maths solutions solved by plain restart (P(solved), FLOP-matched budget in restart attempts):", "",
          "| Generator | 1 | 2 | 4 | 8 |", "|---|---|---|---|---|"]
    for g, d in maths.items():
        L.append(f"| {g} | " + " | ".join(f"{d[k][0]:.3f}" for k in ("1", "2", "4", "8")) + " |")
    (FIN / "mitigation_law.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
