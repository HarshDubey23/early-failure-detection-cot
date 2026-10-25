"""Stage 35 (CPU): counterfactual commitment for hallucinations, against human span labels.

The math study asked whether the annotated first error is where a solution becomes doomed. Here
the same question is asked of RAGTruth: the generator that wrote each released response
(Llama-2-7B-chat) is resampled from sentence boundaries of its own human-labelled responses.

Boundaries (src/recovery/ragtruth.mechanism_points): for a hallucinated response with first
hallucinated sentence e, k in {0, e-2, e-1, e, e+1} (k = sentences kept); for a human-clean
response (placebo), k = 0 and {a-1, a} at anchors a drawn from the hallucinated responses'
relative positions e/n.

For a boundary k and a continuation c sampled from it:
  continuation-clean   no sentence of c is unsupported (evaluator E2 at every boundary; E1 at
                       k = 0 and k = e-1 of every hallucinated response)
  full-clean           prefix human-clean (k < e) and c continuation-clean
  hazard               share of c's sentences that are unsupported (E2)

Questions (paired within a response; intervals from a response bootstrap):
  R1 recovery  u_{e-1} - u_0: is resampling from just before the first human-labelled
               hallucinated sentence better than restarting? (math analogue: restart won for
               three Qwen models, not for Llama-3.1-8B)
  R2 snowball  J = h_e - h_{e-1}: does keeping the hallucinated sentence raise the hazard of the
               rest, beyond the same jump at placebo anchors of clean responses?
  R3 doomed    share of hallucinated responses whose restarts are rarely clean (u_0 <= 1/8)

Usage:
  python scripts/35_mechanism.py --items data/recovery/items.ragtruth.llama-2-7b-chat.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import candidates as C  # noqa: E402
from src.recovery import stats as S  # noqa: E402
from src.recovery.ragtruth import mechanism_points, sentence_labels, sentence_spans  # noqa: E402


def cont_flags(prefix: str, full: str, flags_full: list | None) -> list | None:
    """Flags of the continuation's sentences (sentences that start after the kept prefix)."""
    if flags_full is None:
        return None
    spans = sentence_spans(full)
    if len(spans) != len(flags_full):
        return None
    cut = len(prefix.strip())
    return [f for (a, _), f in zip(spans, flags_full) if cut == 0 or a >= cut - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="35: commitment of hallucinations")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--run-dir", default="results/recovery/run")
    ap.add_argument("--out", default="results/recovery/final/mechanism")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-mock", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    run = C.load_run(run_dir, args.items)
    E2 = {r["cand_id"]: r for r in C.read_jsonl(run_dir / "scores_E2.jsonl") if r.get("sent_scores") is not None}
    E1 = {r["cand_id"]: r for r in C.read_jsonl(run_dir / "judge_E1.jsonl") if r.get("unsupported") is not None}
    if any(r.get("backend") == "mock" for r in list(E2.values())[:20]) and not args.allow_mock:
        raise SystemExit("mock scores: pass --allow-mock (pipeline test only)")

    labelled = {iid: it for iid, it in run["items"].items()
                if it.get("human_labels") is not None and it.get("draft_quality") == "good" and iid in run["drafts"]}
    rels = []
    for iid, it in labelled.items():
        lab = sentence_labels(it["draft"], it["human_labels"])
        if any(lab):
            rels.append((lab.index(1) + 1) / len(lab))

    rows, design = [], {}
    for iid, it in labelled.items():
        draft = it["draft"]
        mp = mechanism_points(draft, it["human_labels"], iid, rels)
        design[iid] = {**mp, "n": len(sentence_spans(draft)), "task": it["task"]}
        bank = run["bank"].get(iid, {})
        for k in mp["points"]:
            if k not in bank:
                continue
            b = bank[k]
            for i, (cont, tok) in enumerate(zip(b["conts"], b["gen_tokens"])):
                cid = f"{iid}|k{k}|s{i}"
                full = (b["prefix"] + cont).strip()
                f2 = cont_flags(b["prefix"], full, [int(s >= args.tau) for s in E2[cid]["sent_scores"]]
                                if cid in E2 else None)
                f1 = None
                if cid in E1:
                    uns = set(E1[cid]["unsupported"])
                    f1 = cont_flags(b["prefix"], full, [int(j + 1 in uns) for j in range(E1[cid]["n_sent"])])
                rows.append({"item_id": iid, "task": it["task"], "e": mp["e"], "k": k, "s": i, "tokens": tok,
                             "clean_E2": (int(sum(f2) == 0) if f2 is not None else np.nan),
                             "hazard_E2": (float(np.mean(f2)) if f2 else (0.0 if f2 == [] else np.nan)),
                             "clean_E1": (int(sum(f1) == 0) if f1 is not None else np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no mechanism rollouts")
        return
    df.to_csv(out / "continuations.csv", index=False)
    pk = df.groupby(["item_id", "k"]).agg(u_E2=("clean_E2", "mean"), u_E1=("clean_E1", "mean"),
                                          h_E2=("hazard_E2", "mean"), tokens=("tokens", "mean"),
                                          m_E1=("clean_E1", "count")).reset_index()
    pk.to_csv(out / "points.csv", index=False)
    V = {(r.item_id, int(r.k)): r for r in pk.itertuples()}
    rep = {"n_hallucinated_responses": int(sum(d["e"] is not None for d in design.values())),
           "n_with_rollouts": int(df.item_id.nunique()), "n_continuations": int(len(df))}

    # R1 recovery: before the first human hallucination vs restart (prefix before e is human-clean)
    r1 = []
    for iid, d in design.items():
        e = d["e"]
        if e is None or (iid, 0) not in V or (iid, e - 1) not in V:
            continue
        r1.append({"item_id": iid, "task": d["task"], "e": e, "n": d["n"],
                   "u0_E2": V[(iid, 0)].u_E2, "ub_E2": V[(iid, e - 1)].u_E2,
                   "u0_E1": V[(iid, 0)].u_E1, "ub_E1": V[(iid, e - 1)].u_E1,
                   "tok0": V[(iid, 0)].tokens, "tokb": V[(iid, e - 1)].tokens})
    R1 = pd.DataFrame(r1)
    R1.to_csv(out / "r1_recovery.csv", index=False)
    res = {}
    for ev in ["E1", "E2"]:
        sub = R1.dropna(subset=[f"u0_{ev}", f"ub_{ev}"]) if not R1.empty else R1
        if sub.empty:
            continue
        d = (sub[f"ub_{ev}"] - sub[f"u0_{ev}"]).to_numpy()
        cl = sub.item_id.to_numpy()
        late = sub.e.to_numpy() > 1  # e = 1: the two targets coincide
        res[ev] = {"n": int(len(sub)), "restart": S.boot_ci(sub[f"u0_{ev}"].to_numpy(), cl, args.n_boot),
                   "before_first_hallucination": S.boot_ci(sub[f"ub_{ev}"].to_numpy(), cl, args.n_boot),
                   "difference": S.boot_ci(d, cl, args.n_boot), "p_signflip": S.signflip_p(d, cl),
                   "difference_e_gt_1": S.boot_ci(d[late], cl[late], args.n_boot),
                   "by_task": {t: S.boot_ci((g[f"ub_{ev}"] - g[f"u0_{ev}"]).to_numpy(), g.item_id.to_numpy(),
                                            args.n_boot) for t, g in sub.groupby("task")}}
    if not R1.empty:
        res["continuation_tokens_restart_vs_before"] = [float(R1.tok0.mean()), float(R1.tokb.mean())]
    rep["R1_recovery"] = res

    # R2 snowball: hazard jump when the hallucinated sentence enters the prefix, vs placebo anchors
    J, P = [], []
    for iid, d in design.items():
        if d["e"] is not None:
            e = d["e"]
            if (iid, e - 1) in V and (iid, e) in V:
                J.append({"item_id": iid, "J": V[(iid, e)].h_E2 - V[(iid, e - 1)].h_E2,
                          "before": V[(iid, e - 1)].h_E2, "after": V[(iid, e)].h_E2})
        else:
            for a in d["anchors"]:
                if (iid, a - 1) in V and (iid, a) in V:
                    P.append({"item_id": iid, "J": V[(iid, a)].h_E2 - V[(iid, a - 1)].h_E2})
    J, P = pd.DataFrame(J).dropna(), pd.DataFrame(P).dropna()
    if not J.empty:
        r2 = {"n_hallucinated": int(len(J)), "hazard_before": float(J.before.mean()), "hazard_after": float(J.after.mean()),
              "jump_hallucinated": S.boot_ci(J.J.to_numpy(), J.item_id.to_numpy(), args.n_boot)}
        if not P.empty:
            rng = np.random.default_rng(args.seed)
            jh, jc = J.groupby("item_id").J.mean().to_numpy(), P.groupby("item_id").J.mean().to_numpy()
            bo = [rng.choice(jh, len(jh)).mean() - rng.choice(jc, len(jc)).mean() for _ in range(args.n_boot)]
            r2.update({"n_placebo_responses": int(len(jc)),
                       "jump_placebo_clean": S.boot_ci(P.J.to_numpy(), P.item_id.to_numpy(), args.n_boot),
                       "excess_jump": [float(jh.mean() - jc.mean()), float(np.quantile(bo, 0.025)),
                                       float(np.quantile(bo, 0.975))]})
        rep["R2_snowball"] = r2
    ev = [{"r": k - d["e"], "h": V[(iid, k)].h_E2, "u": V[(iid, k)].u_E2}
          for iid, d in design.items() if d["e"] is not None for k in d["points"] if (iid, k) in V]
    if ev:
        EV = pd.DataFrame(ev)
        rep["event_study"] = {int(r): {"hazard": float(g.h.mean()), "clean": float(g.u.mean()), "n": int(len(g))}
                              for r, g in EV.groupby("r")}
    if not R1.empty:
        rep["R3_restart_rarely_clean_share_E2"] = S.boot_ci((R1.u0_E2 <= 0.125).astype(float).to_numpy(),
                                                           R1.item_id.to_numpy(), args.n_boot)
    json.dump(rep, open(out / "summary.json", "w"), indent=2, default=float)
    print(json.dumps(rep, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
