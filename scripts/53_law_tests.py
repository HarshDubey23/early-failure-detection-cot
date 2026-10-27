"""Stage 53 (CPU): two checks of Proposition 3 (the restart-loop law) on committed data (exploratory).

1. Budget invariance. The law predicts that a larger retry budget raises coverage but leaves the
   hallucination rate of delivered answers unchanged. The replay ran restart-until-D-passes at
   budgets 1, 2 and 4 on the same 450 RAGTruth sources, so we compare delivered-answer hallucination
   and coverage between budgets 1 and 4, paired by source (any grader is a valid yardstick here,
   because the law holds for any fixed definition of "hallucinated").
2. Operating curve. Sweeping the detector threshold tau trades hallucination against cost. From the
   449 human-labelled drafts the law gives, for each tau and each gate (D, E2, or both), the predicted
   hallucination of delivered answers pi(tau), the flag rate q(tau), the expected number of draws per
   answered input 1/(1-q) and the coverage with two draws 1-q^2.

Output: results/recovery/final/law_tests.{json,md}
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
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def boot_mean_diff(a: np.ndarray, b: np.ndarray, rng) -> list:
    """Mean of a - b (paired by row, NaN rows dropped) and its 95% bootstrap CI."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(d))]


def boot_ratio_diff(h4, c4, h1, c1, rng) -> list:
    """Delivered-answer hallucination at budget 4 minus budget 1: sum(h*c)/sum(c), items resampled jointly."""
    n = len(h1)
    stat = lambda i: (h4[i] * c4[i]).sum() / c4[i].sum() - (h1[i] * c1[i]).sum() / c1[i].sum()  # noqa: E731
    bs = [stat(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(stat(np.arange(n))), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), n]


def main() -> None:
    rng = np.random.default_rng(53)
    P = pd.read_csv(FIN / "per_item.csv")
    P = P[P.benchmark == "ragtruth"]
    inv = {}
    for pol in ("restart_fb", "backtrack_fb"):
        b1 = P[(P.policy == pol) & (P.budget == 1)].set_index("item_id").sort_index()
        b4 = P[(P.policy == pol) & (P.budget == 4)].set_index("item_id").loc[b1.index]
        rec = {"coverage_b1": float(b1.covered.mean()), "coverage_b4": float(b4.covered.mean()),
               "coverage_diff": boot_mean_diff(b4.covered.to_numpy(float), b1.covered.to_numpy(float), rng),
               "tokens_b1": float(b1.cost_total.mean()), "tokens_b4": float(b4.cost_total.mean())}
        for g in ("hall_E1", "hall_E2"):
            h1, h4 = b1[g].fillna(0).to_numpy(float), b4[g].fillna(0).to_numpy(float)
            c1, c4 = b1.covered.to_numpy(float), b4.covered.to_numpy(float)
            rec[f"{g}_delivered_b1"] = float((h1 * c1).sum() / c1.sum())
            rec[f"{g}_delivered_b4"] = float((h4 * c4).sum() / c4.sum())
            rec[f"{g}_delivered_diff"] = boot_ratio_diff(h4, c4, h1, c1, rng)
        inv[pol] = rec

    # operating curve from human-labelled drafts
    dmax = {json.loads(x)["cand_id"]: max(json.loads(x).get("sent_scores") or [0.0]) for x in open(RUN / "scores_D.jsonl", encoding="utf-8")}
    e2max = {json.loads(x)["cand_id"]: max(json.loads(x).get("sent_scores") or [0.0]) for x in open(RUN / "scores_E2.jsonl", encoding="utf-8")}
    lab = pd.read_csv(ROOT / "results/recovery/ragtruth_labels/responses.csv")
    lab = lab[(lab.model == "llama-2-7b-chat") & (lab.split == "test") & (lab.quality == "good")].copy()
    lab["cid"] = [f"ragtruth:{s}:llama-2-7b-chat|draft" for s in lab.source_id]
    lab = lab[lab.cid.isin(dmax.keys()) & lab.cid.isin(e2max.keys())]
    H = lab.halluc.to_numpy(int)
    sD = lab.cid.map(dmax).to_numpy(float)
    sE = lab.cid.map(e2max).to_numpy(float)
    gates = {"D (LettuceDetect-base)": lambda t: sD >= t, "E2 (LettuceDetect-large)": lambda t: sE >= t,
             "D or E2 flags": lambda t: (sD >= t) | (sE >= t)}
    curve = []
    for gname, flag in gates.items():
        for t in TAUS:
            def one(idx):
                f = flag(t)[idx]
                h = H[idx]
                q = f.mean()
                return h[~f].mean() if (~f).any() else np.nan, q
            pi, q = one(np.arange(len(H)))
            bs = np.array([one(rng.integers(0, len(H), len(H)))[0] for _ in range(500)])
            curve.append({"gate": gname, "tau": t, "pi": float(pi), "pi_lo": float(np.nanquantile(bs, 0.025)),
                          "pi_hi": float(np.nanquantile(bs, 0.975)), "q": float(q),
                          "draws_per_answer": float(1 / (1 - q)) if q < 1 else float("inf"),
                          "coverage_two_draws": float(1 - q ** 2)})
    # The preregistered H0 tested CARE against the base model; DRA against the base model is tested here
    # directly (exploratory), on the same items, so the two are not conflated.
    base = P[(P.policy == "base") & (P.budget == 0)].set_index("item_id").sort_index()
    vs_base = {}
    for pol in ("restart_fb", "care"):
        x = P[(P.policy == pol) & (P.budget == 2)].set_index("item_id").loc[base.index]
        vs_base[pol] = {g: boot_mean_diff(x[g].to_numpy(float), base[g].to_numpy(float), rng) for g in ("hall_E1", "hall_E2")}
    out = {"budget_invariance": inv, "operating_curve": curve, "vs_base_b2": vs_base,
           "human_base_rate": float(H.mean()), "n_drafts": int(len(H))}
    (FIN / "law_tests.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fmt = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"  # noqa: E731
    L = ["# Checks of the restart-loop law (Proposition 3), exploratory", "",
         "## 1. More retries buy coverage, not faithfulness", "",
         "RAGTruth, 450 sources, same items at budget 1 and budget 4 (paired). Delivered = answered outputs.", "",
         "| Loop | coverage b=1 -> b=4 | change | delivered hallucination (E1) b=1 -> b=4 | change | (E2) b=1 -> b=4 | change |",
         "|---|---|---|---|---|---|---|"]
    for pol, r in inv.items():
        L.append(f"| {pol} | {r['coverage_b1']:.3f} -> {r['coverage_b4']:.3f} | {fmt(r['coverage_diff'])} | "
                 f"{r['hall_E1_delivered_b1']:.3f} -> {r['hall_E1_delivered_b4']:.3f} | {fmt(r['hall_E1_delivered_diff'])} | "
                 f"{r['hall_E2_delivered_b1']:.3f} -> {r['hall_E2_delivered_b4']:.3f} | {fmt(r['hall_E2_delivered_diff'])} |")
    L += ["", f"## 2. Operating curve predicted from {len(H)} human-labelled drafts (base rate {H.mean():.3f})", "",
          "| Gate | tau | predicted hallucination of delivered answers | flag rate q | draws per answered input | coverage with 2 draws |",
          "|---|---|---|---|---|---|"]
    for c in curve:
        L.append(f"| {c['gate']} | {c['tau']:.1f} | {c['pi']:.3f} [{c['pi_lo']:.3f}, {c['pi_hi']:.3f}] | {c['q']:.3f} | "
                 f"{c['draws_per_answer']:.2f} | {c['coverage_two_draws']:.3f} |")
    L += ["", "## 3. Each loop against the base model, same items (budget 2; exploratory except CARE under E1, which is H0)", "",
          "| Loop | change in hallucination, E1 judge | change, E2 |", "|---|---|---|"]
    for pol, r in vs_base.items():
        L.append(f"| {pol} | {fmt(r['hall_E1'])} | {fmt(r['hall_E2'])} |")
    (FIN / "law_tests.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
