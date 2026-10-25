"""Stage 34 (CPU): replay every policy on the scored bank, compute metrics and the planned tests.

Reads results/recovery/run/{drafts,bank,repair,cove}.*.jsonl and the scores of stage 33, and
writes results/recovery/<tag>/:

  evaluator_validation.json   D, E2 and E1 against the human RAGTruth labels of the released drafts
  per_item.csv                every (item, policy, budget): outcomes averaged over R sample orders
  summary.csv                 means with cluster-bootstrap 95% intervals, per benchmark/generator
  tests.json                  the preregistered comparisons (docs/PREREGISTRATION_RECOVERY.md)

Metrics (per output; abstention = the refusal string, which makes no claim):
  hallucinated   any sentence unsupported (E1 judge, primary; E2 detector; D; human for drafts)
  covered        not an abstention and not empty
  correct        HotpotQA answerable: covered and the gold answer appears in the output
  abstain_ok     HotpotQA unanswerable: abstained
  cost           generator-equivalent tokens (replay.py), latency from measured token rates

Usage:
  python scripts/34_replay.py --items data/recovery/items.*.jsonl --out results/recovery/final
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recovery import candidates as C  # noqa: E402
from src.recovery import replay as R  # noqa: E402
from src.recovery import stats as S  # noqa: E402
from src.recovery.ragtruth import sentence_labels, sentence_spans  # noqa: E402

def CONFIRM_REFS(bp: float) -> list:  # noqa: N802 - the preregistered comparison family
    """docs/PREREGISTRATION_RECOVERY.md, section 4: the controller against each reference."""
    return [("restart_fb", bp), ("backtrack_fb", bp), ("repairdel", 0), ("cove_all", 0), ("care@restart", bp)]


PARAMS_B = {"llama-2-7b-chat": 6.74, "llama-3.1-8b-instruct": 8.03, "mistral-7B-instruct": 7.24, "mock": 7.0}
D_PARAMS_B = 0.149  # ModernBERT-base (LettuceDetect base)
CB = " "


def norm(s: str) -> str:
    import re
    import string

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser(description="34: replay policies")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--run-dir", default="results/recovery/run")
    ap.add_argument("--out", default="results/recovery/final")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--primary-budget", type=float, default=2)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--allow-mock", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)

    run = C.load_run(run_dir, args.items)
    D = {r["cand_id"]: r for r in C.read_jsonl(run_dir / "scores_D.jsonl") if r.get("sent_scores") is not None}
    E2 = {r["cand_id"]: r for r in C.read_jsonl(run_dir / "scores_E2.jsonl") if r.get("sent_scores") is not None}
    E1 = {r["cand_id"]: r for r in C.read_jsonl(run_dir / "judge_E1.jsonl") if r.get("unsupported") is not None}
    backends = {r.get("backend") for r in list(D.values())[:50] + list(E1.values())[:50]}
    mock = "mock" in backends or any(r.get("backend") == "mock" for r in run["drafts"].values())
    if mock and not args.allow_mock:
        raise SystemExit("mock outputs found: pass --allow-mock (pipeline test only; numbers are not results)")
    rates = {"prefill": 3000.0, "decode": 22.0, "source": "default (no live measurement)"}
    if (run_dir / "live_rates.json").exists():
        rates = json.load(open(run_dir / "live_rates.json"))
    texts = {c["cand_id"]: c["text"] for st in (1, 2) for c in
             C.build_candidates(run, {k: v["sent_scores"] for k, v in D.items()}, args.tau, st)}

    # ---------------------------------------------------------------- per-candidate outcomes
    missing = Counter()

    def evaluate(cid: str, it: dict) -> dict:
        if cid == R.REFUSE:
            text = ""
        else:
            text = texts.get(cid, "")
        refused = cid == R.REFUSE or C.is_refusal_text(text)
        res = {"covered": float(not refused), "words": float(len(text.split())) if not refused else 0.0}
        if refused:
            res.update({"hall_E1": 0.0, "hall_E2": 0.0, "hall_D": 0.0, "supported_sent": 0.0})
        else:
            e2 = E2.get(cid)
            res["hall_E2"] = float(max(e2["sent_scores"] or [0]) >= args.tau) if e2 else np.nan
            d = D.get(cid)
            res["hall_D"] = float(max(d["sent_scores"] or [0]) >= args.tau) if d else np.nan
            e1 = E1.get(cid)
            if e1 is not None:
                res["hall_E1"] = float(len(e1["unsupported"]) > 0)
                res["supported_sent"] = float(e1["n_sent"] - len(e1["unsupported"]))
            else:
                missing["E1"] += 1
                res["hall_E1"] = res["hall_E2"]  # fallback, counted in missing
                res["supported_sent"] = np.nan
            if e2 is None:
                missing["E2"] += 1
        if it["benchmark"] == "hotpot":
            ok = (not refused) and any(norm(g) in norm(text) for g in it["gold"])
            res["correct"] = float(it["answerable"] and ok)
            res["abstain_ok"] = float((not it["answerable"]) and refused)
            res["behaviour_ok"] = res["correct"] if it["answerable"] else res["abstain_ok"]
        return res

    # ---------------------------------------------------------------- evaluator validation
    val_rows = []
    for iid, it in run["items"].items():
        if it.get("human_labels") is None or iid not in run["drafts"]:
            continue
        cid = f"{iid}|draft"
        draft = run["drafts"][iid]["draft"]
        hum = sentence_labels(draft, it["human_labels"])
        row = {"item_id": iid, "task": it["task"], "human": float(any(hum))}
        for name, store in [("D", D), ("E2", E2)]:
            if cid in store:
                sc = store[cid]["sent_scores"]
                row[name] = float(max(sc or [0]) >= args.tau)
                if len(sc) == len(hum):
                    row[f"{name}_sent"] = list(zip(hum, [int(s >= args.tau) for s in sc]))
        if cid in E1:
            row["E1"] = float(len(E1[cid]["unsupported"]) > 0)
            uns = set(E1[cid]["unsupported"])
            if E1[cid]["n_sent"] == len(hum):
                row["E1_sent"] = list(zip(hum, [int(j + 1 in uns) for j in range(len(hum))]))
        val_rows.append(row)
    validation = {}
    if val_rows:
        V = pd.DataFrame(val_rows)
        for ev in ["D", "E2", "E1"]:
            if ev not in V:
                continue
            sub = V.dropna(subset=[ev])
            validation[ev] = {"response_level": S.binary_agreement(sub[ev], sub.human),
                              "by_task": {t: S.binary_agreement(g[ev], g.human) for t, g in sub.groupby("task")}}
            pairs = [p for lst in V.get(f"{ev}_sent", pd.Series(dtype=object)).dropna() for p in lst]
            if pairs:
                h, p = zip(*pairs)
                validation[ev]["sentence_level"] = S.binary_agreement(p, h)
    json.dump(validation, open(out / "evaluator_validation.json", "w"), indent=2)

    # ---------------------------------------------------------------- item contexts
    def family_cost(calls: list, shared) -> tuple:
        """(cached cost, no-cache cost) of calls sharing a prompt prefix of `shared` tokens."""
        pre = [float(c.get("prompt_tokens") or 0) for c in calls]
        gen = [float(c.get("gen_tokens") or 0) for c in calls]
        if not calls:
            return 0.0, 0.0
        sh = float(shared) if shared else 0.0
        cached = (sh + sum(max(x - sh, 0.0) for x in pre)) if sh else sum(pre)
        return cached + sum(gen), sum(pre) + sum(gen)

    ctxs = {}
    for iid, it in run["items"].items():
        if iid not in run["drafts"] or f"{iid}|draft" not in D:
            continue
        dr = run["drafts"][iid]
        draft = dr["draft"]
        ds = D[f"{iid}|draft"]["sent_scores"] or []
        gen = it["generator"]
        w_d = D_PARAMS_B / PARAMS_B.get(gen, 7.0)
        spans = sentence_spans(draft)
        bank = run["bank"].get(iid, {})
        p0 = float(bank[0]["prompt_tokens"] if 0 in bank and bank[0].get("prompt_tokens")
                   else dr.get("prompt_tokens") or C.approx_tokens(it["prompt"]))
        draft_tok = float(dr.get("gen_tokens") or C.approx_tokens(draft))
        det_prompt = float(C.approx_tokens(it["prompt"]))
        det_draft = float(D[f"{iid}|draft"].get("det_tokens", det_prompt + draft_tok))
        pts, true_v = {}, {}
        for k in R.system_points(ds, args.tau):
            if k not in bank:
                continue
            b = bank[k]
            rest = len(draft) - (spans[k][0] if k < len(spans) else len(draft))
            est = max(8.0, draft_tok * rest / max(len(draft), 1))
            pre_k = float(b.get("prefix_tokens", 0) if k > 0 else 0)
            samples = []
            for i, g in enumerate(b["gen_tokens"]):
                cid = f"{iid}|k{k}|s{i}"
                if cid not in D:
                    continue
                samples.append(R.Sample(cid=cid, gen_tokens=float(g), d_max=float(max(D[cid]["sent_scores"] or [0])),
                                        det_tokens=float(D[cid].get("det_tokens", 0))))
            if samples:
                pts[k] = R.Point(k=k, prefill=pre_k, prefill_nocache=p0 + pre_k, est_gen=est,
                                 est_det=det_prompt + pre_k + est, samples=samples)
                true_v[k] = 1.0 - float(np.nanmean([evaluate(s.cid, it)["hall_E1"] for s in samples]))
        fl = R.flagged_sentences(ds, args.tau)
        rep = [r for j, r in run["repair"].get(iid, {}).items() if j in fl and not r.get("skipped")]
        rc, rcn = family_cost(rep, rep[0].get("shared_tokens") if rep else None)
        det_rep = w_d * float(D[f"{iid}|repair"].get("det_tokens", 0)) if f"{iid}|repair" in D else 0.0
        rep_steps = [(rc - sum(float(r.get("gen_tokens") or 0) for r in rep),
                      float(max([float(r.get("gen_tokens") or 0) for r in rep] or [0])))] if rep else []
        cv = run["cove"].get(iid, {})
        calls = cv.get("calls", [])
        cc, ccn = family_cost(calls, cv.get("shared_tokens"))
        sh = float(cv.get("shared_tokens") or 0)
        plan = [c for c in calls if c["step"] == "plan"]
        ans = [c for c in calls if c["step"] == "answer"]
        rev = [c for c in calls if c["step"] == "revise"]
        cove_steps = [(float(c["prompt_tokens"]), float(c["gen_tokens"])) for c in plan]
        if ans:
            cove_steps.append((float(sum(max(c["prompt_tokens"] - sh, 0) for c in ans)),
                               float(max(c["gen_tokens"] for c in ans))))
        cove_steps += [(float(max(c["prompt_tokens"] - sh, 0)), float(c["gen_tokens"])) for c in rev]
        rd = D.get(f"{iid}|repair")
        ctxs[iid] = R.ItemCtx(
            iid=iid, source_id=str(it["source_id"]), is_qa=bool(it["is_qa"]), flagged=bool(fl), draft_cid=f"{iid}|draft",
            draft_dmax=float(max(ds or [0])), budget_unit=p0 + draft_tok + w_d * det_draft, points=pts, w_d=w_d,
            delete_cid=f"{iid}|delete" if f"{iid}|delete" in texts else None,
            repair_cid=f"{iid}|repair" if (f"{iid}|repair" in texts and rd is not None) else None,
            repair_dmax=float(max(rd["sent_scores"] or [0])) if rd else 1.0,
            repairdel_cid=f"{iid}|repairdel" if f"{iid}|repairdel" in texts else None,
            repair_cost=rc + det_rep, repair_cost_nocache=rcn + det_rep, repair_steps=rep_steps,
            cove_cid=f"{iid}|cove" if f"{iid}|cove" in texts else None, cove_cost=cc, cove_cost_nocache=ccn,
            cove_steps=cove_steps, true_v=true_v, common=p0 + draft_tok, det_draft=w_d * det_draft,
            draft_steps=[(p0, draft_tok)])

    # ---------------------------------------------------------------- replay
    rows, round0 = [], []
    policies = [(p, 0.0) for p in R.POLICIES_FIXED] + [(p, b) for p in R.POLICIES_BUDGETED for b in args.budgets]
    for iid, ctx in ctxs.items():
        it = run["items"][iid]
        uses_d = lambda p: p not in ("base", "cove_all")  # noqa: E731
        for pol, b in policies:
            n_rounds = args.rounds if pol in R.POLICIES_BUDGETED else 1
            acc, actions = [], Counter()
            for rnd in range(n_rounds):
                o = R.run_policy(pol, ctx, rnd, b, args.tau)
                if rnd == 0:
                    round0.append({"item_id": iid, "policy": pol, "budget": b, "cid": o.cid, "action": o.action})
                ev = evaluate(o.cid, it)
                steps = ctx.draft_steps + ([(0.0, 0.0)] if uses_d(pol) else []) + o.steps
                ev.update({"cost_recovery": o.cost, "cost_total": ctx.common + (ctx.det_draft if uses_d(pol) else 0) + o.cost,
                           "cost_total_nocache": ctx.common + (ctx.det_draft if uses_d(pol) else 0) + o.cost_nocache,
                           "gen_tokens_recovery": o.gen_tokens, "calls": o.calls,
                           "latency_s": R.latency(steps, rates["prefill"], rates["decode"]),
                           "budget_used": o.cost / (b * ctx.budget_unit) if b else np.nan})
                acc.append(ev)
                actions[o.action.split("@")[0]] += 1
            mean = {k: float(np.nanmean([a.get(k, np.nan) for a in acc])) for k in acc[0]}
            rows.append({"item_id": iid, "source_id": ctx.source_id, "benchmark": it["benchmark"], "task": it["task"],
                         "generator": it["generator"], "answerable": it.get("answerable"), "flagged": ctx.flagged,
                         "policy": pol, "budget": b, **mean,
                         "actions": json.dumps({k: v / n_rounds for k, v in actions.items()}),
                         "hall_human": (float(len(it["human_labels"]) > 0) if pol == "base" and it.get("human_labels") is not None
                                        else np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_item.csv", index=False)
    pd.DataFrame(round0).to_csv(out / "outputs_round0.csv", index=False)  # concrete outputs, for human review

    # ---------------------------------------------------------------- summaries
    metrics = ["hall_E1", "hall_E2", "hall_D", "hall_human", "covered", "words", "supported_sent", "correct",
               "abstain_ok", "behaviour_ok", "cost_total", "cost_total_nocache", "cost_recovery", "gen_tokens_recovery", "latency_s"]
    summ = []
    for (bench, gen, pol, b), g in df.groupby(["benchmark", "generator", "policy", "budget"]):
        for scope, sub in [("all", g)] + [(f"task={t}", s) for t, s in g.groupby("task") if g.task.nunique() > 1] + \
                          [(f"answerable={a}", s) for a, s in g.groupby("answerable") if bench == "hotpot"]:
            row = {"benchmark": bench, "generator": gen, "policy": pol, "budget": b, "scope": scope, "n": len(sub)}
            for mtr in metrics:
                if mtr in sub and sub[mtr].notna().any():
                    m, lo, hi = S.boot_ci(sub[mtr].to_numpy(), sub.source_id.to_numpy(), n_boot=args.n_boot // 4)
                    row.update({mtr: m, f"{mtr}_lo": lo, f"{mtr}_hi": hi})
            cov = sub.covered.sum()
            row["hall_E1_among_covered"] = float(sub.hall_E1.sum() / cov) if cov else np.nan
            summ.append(row)
    SUM = pd.DataFrame(summ)
    SUM.to_csv(out / "summary.csv", index=False)

    # ---------------------------------------------------------------- planned tests
    tests = {"missing_evaluations": dict(missing), "latency_rates": rates, "n_items": int(df.item_id.nunique())}
    # preregistered evaluator rule (section 7): E1 is primary unless it disagrees with the human labels
    e1v = validation.get("E1", {}).get("response_level", {})
    e1_ok = bool(e1v) and (e1v.get("f1", 0) or 0) >= 0.50 and (e1v.get("kappa", 0) or 0) >= 0.20
    primary = "hall_E1" if e1_ok or not validation else "hall_E2"
    tests["primary_metric"] = {"metric": primary, "E1_response_level": e1v,
                               "rule": "E1 primary unless F1 < 0.50 or kappa < 0.20 against human labels"}
    bp = args.primary_budget

    def compare(sub, pol_a, b_a, pol_b, b_b, metric):
        A = sub[(sub.policy == pol_a) & (sub.budget == b_a)].set_index("item_id")
        B = sub[(sub.policy == pol_b) & (sub.budget == b_b)].set_index("item_id")
        common = A.index.intersection(B.index)
        d = (A.loc[common, metric] - B.loc[common, metric]).to_numpy()
        cl = A.loc[common, "source_id"].to_numpy()
        m, lo, hi = S.boot_ci(d, cl, n_boot=args.n_boot)
        return {"a": f"{pol_a}@{b_a:g}", "b": f"{pol_b}@{b_b:g}", "metric": metric, "n": int(len(common)),
                "diff": m, "lo": lo, "hi": hi, "p": S.signflip_p(d, cl, n_perm=args.n_perm)}

    families = {}
    for (bench, gen), sub in df.groupby(["benchmark", "generator"]):
        key = f"{bench}/{gen}"
        fam = {"sanity_care_vs_base": compare(sub, "care", bp, "base", 0, primary)}
        confirm = {}
        for ref, rb in CONFIRM_REFS(bp):
            c = compare(sub, "care", bp, ref, rb, primary)
            c["coverage"] = compare(sub, "care", bp, ref, rb, "covered")
            c["cost_total"] = compare(sub, "care", bp, ref, rb, "cost_total")
            confirm[f"care_vs_{ref}"] = c
        adj = S.holm({k: v["p"] for k, v in confirm.items()})
        for k, v in confirm.items():
            v["p_holm"] = adj[k]
            v["hall_lower"] = v["hi"] < 0
            v["coverage_noninferior"] = v["coverage"]["lo"] > -0.02
            v["verdict"] = ("care better" if v["hall_lower"] and v["coverage_noninferior"] and v["p_holm"] < 0.05
                            else "care worse" if v["lo"] > 0 and v["p_holm"] < 0.05 else "not shown")
        fam["confirmatory"] = confirm
        fam["E2_robustness"] = {f"care_vs_{ref}": compare(sub, "care", bp, ref, rb, "hall_E2")
                                for ref, rb in CONFIRM_REFS(bp)}
        fam["oracle_headroom"] = {"oracle_vs_restart_allocation": compare(sub, "care@oracle", bp, "care@restart", bp, "hall_E1"),
                                  "care_vs_restart_allocation": compare(sub, "care", bp, "care@restart", bp, "hall_E1")}
        fam["cost_view_nocache"] = {f"care_vs_{ref}": compare(sub, "care", bp, ref, rb, "cost_total_nocache")
                                    for ref, rb in CONFIRM_REFS(bp)}
        ca = sub[(sub.policy == "care") & (sub.budget == bp)].set_index("item_id")
        rf = sub[(sub.policy == "restart_fb") & (sub.budget == bp)].set_index("item_id")
        common = ca.index.intersection(rf.index)
        fam["equivalence_care_vs_restart_fb"] = S.tost((ca.loc[common, primary] - rf.loc[common, primary]).to_numpy(),
                                                       ca.loc[common, "source_id"].to_numpy(), margin=0.03,
                                                       n_boot=args.n_boot)
        if bench == "hotpot":
            hp = {f"{m}_care_vs_{ref}": compare(sub, "care", bp, ref, rb, m)
                  for m in ["behaviour_ok", "correct", "abstain_ok", "covered", primary]
                  for ref, rb in [("base", 0), ("restart_fb", bp), ("cove_all", 0)]}
            adj = S.holm({k: v["p"] for k, v in hp.items() if k.startswith("behaviour_ok")})
            for k, v in adj.items():
                hp[k]["p_holm"] = v
            fam["hotpot"] = hp
        families[key] = fam
    tests["families"] = families
    json.dump(tests, open(out / "tests.json", "w"), indent=2, default=float)

    # ---------------------------------------------------------------- print the main table
    show = SUM[(SUM.scope == "all") & ((SUM.budget == bp) | (SUM.budget == 0))]
    cols = ["benchmark", "generator", "policy", "budget", "n", "hall_E1", "hall_E2", "hall_human", "covered",
            "correct", "cost_total", "latency_s"]
    print(show[[c for c in cols if c in show]].round(3).to_string(index=False))
    print("\nvalidation:", json.dumps({k: v["response_level"] for k, v in validation.items()}, indent=1, default=float))
    print("\nmissing evaluations:", dict(missing))
    for key, fam in families.items():
        print(f"\n== {key}")
        for k, v in fam["confirmatory"].items():
            print(f"  {k}: dHall={v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] p_holm={v['p_holm']:.4f} "
                  f"dCov={v['coverage']['diff']:+.3f} [{v['coverage']['lo']:+.3f}] -> {v['verdict']}")
    if mock:
        print("\nWARNING: MOCK pipeline test. Nothing above is a result.")


if __name__ == "__main__":
    main()
