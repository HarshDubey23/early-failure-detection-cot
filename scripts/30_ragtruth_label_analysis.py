"""Stage 30 (CPU): what RAGTruth's human labels say about recovering from hallucinations.

No model is run. Four questions decide which recovery action can work at all:

  A. How common are hallucinations, per generator and task?
  B. Difficulty: how much of "this response hallucinates" is predictable from the source alone
     (the other five generators' responses to the same source) and from the generator's identity?
     This is the RAG analogue of the math finding that early failure prediction is mostly a
     difficulty signal.
  C. Structure inside a response: is the hallucinated content one sentence (local repair can
     work) or several (backtracking/restart needed)? Do hallucinated sentences cluster beyond what
     their number implies (a runs test with the count held fixed, which removes response-level
     propensity), and does the first one sit earlier than chance?
  D. Evidence insufficiency: on QA sources where annotators accepted a refusal as correct
     ("Unable to answer based on given passages."), how often do the generators that answer
     anyway hallucinate? This is the case for abstention.

All of this is observational. Whether a later hallucination is *caused* by an earlier one (the
response is committed) is measured by resampling in stage 32, not here.

Usage:
  git clone https://github.com/ParticleMedia/RAGTruth data/external/RAGTruth
  python scripts/30_ragtruth_label_analysis.py --out results/recovery/ragtruth_labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import ragtruth as rt  # noqa: E402


def cluster_boot(df: pd.DataFrame, fn, cluster: str = "source_id", n_boot: int = 2000, seed: int = 0) -> tuple:
    """Point estimate and 95% percentile interval of fn(df), resampling clusters."""
    rng = np.random.default_rng(seed)
    idx = [np.asarray(v) for v in df.groupby(cluster).indices.values()]
    point = fn(df)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        boots.append(fn(df.iloc[np.concatenate([idx[i] for i in pick])]))
    boots = np.array([b for b in boots if b == b])
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def runs(h: np.ndarray) -> int:
    """Number of maximal runs of 1s."""
    h = np.asarray(h, dtype=int)
    return int(((h[1:] == 1) & (h[:-1] == 0)).sum() + (h[0] == 1)) if len(h) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="30: RAGTruth label structure")
    ap.add_argument("--ragtruth", default="data/external/RAGTruth")
    ap.add_argument("--out", default="results/recovery/ragtruth_labels")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resp, src = rt.load(args.ragtruth)

    rows = []
    for r in resp:
        s = src[r["source_id"]]
        spans = rt.sentence_spans(r["response"])
        lab = rt.sentence_labels(r["response"], r["labels"], spans)
        rows.append({"id": r["id"], "source_id": r["source_id"], "model": r["model"], "split": r["split"],
                     "task": s["task_type"], "quality": r["quality"], "halluc": int(len(r["labels"]) > 0),
                     "n_sent": len(spans), "sent_labels": lab, "n_spans": len(r["labels"]),
                     "refusal": rt.is_refusal(r["response"]),
                     "n_words": len(r["response"].split()),
                     "implicit_true": sum(bool(x.get("implicit_true")) for x in r["labels"]),
                     "due_to_null": sum(bool(x.get("due_to_null")) for x in r["labels"]),
                     "types": [x["label_type"] for x in r["labels"]]})
    df = pd.DataFrame(rows)
    good = df[df.quality == "good"].copy()
    report: dict = {"n_responses": len(df), "n_good": len(good)}

    # ---- A. rates -------------------------------------------------------------------------
    rates = (good.groupby(["split", "model", "task"]).halluc.agg(["mean", "size"]).reset_index())
    rates.to_csv(out / "A_rates.csv", index=False)
    test = good[good.split == "test"]
    report["A_test_rate_by_model"] = {m: round(float(g.halluc.mean()), 3) for m, g in test.groupby("model")}
    report["A_test_rate_by_task"] = {t: round(float(g.halluc.mean()), 3) for t, g in test.groupby("task")}
    report["A_test_rate_all"] = round(float(test.halluc.mean()), 3)

    # ---- B. difficulty: source and generator --------------------------------------------------
    by_src = good.groupby("source_id").halluc.agg(["sum", "size"])
    good = good.join(by_src, on="source_id")
    # NB: the leave-one-out source rate is biased against the response's own label when generator
    # effects are strong (e.g. Data2txt: if the other five hallucinated, the remaining response is
    # usually GPT's); read the source's contribution from generator+task vs generator+task+source.
    good["loo_src"] = np.where(good["size"] > 1, (good["sum"] - good.halluc) / (good["size"] - 1).clip(lower=1), np.nan)
    b = good.dropna(subset=["loo_src"]).copy()
    mt_rate = b.groupby(["model", "task"]).halluc.transform("mean")  # in-sample base rate (descriptive)
    b["mt_rate"] = mt_rate
    X_mt = pd.get_dummies(b[["model", "task"]]).astype(float).to_numpy()
    X_all = np.column_stack([X_mt, b.loo_src.to_numpy()])
    y = b.halluc.to_numpy()
    gkf = GroupKFold(n_splits=5)
    pred_mt, pred_all = np.zeros(len(b)), np.zeros(len(b))
    for tr, te in gkf.split(X_all, y, groups=b.source_id):
        pred_mt[te] = LogisticRegression(max_iter=2000).fit(X_mt[tr], y[tr]).predict_proba(X_mt[te])[:, 1]
        pred_all[te] = LogisticRegression(max_iter=2000).fit(X_all[tr], y[tr]).predict_proba(X_all[te])[:, 1]
    b["p_mt"], b["p_all"] = pred_mt, pred_all

    def auc(col):
        return lambda d: roc_auc_score(d.halluc, d[col]) if d.halluc.nunique() == 2 else np.nan

    B = {}
    for task, sub in [("all", b)] + list(b.groupby("task")):
        B[task] = {name: cluster_boot(sub, auc(col), n_boot=args.n_boot // 4, seed=args.seed)
                   for name, col in [("generator+task", "p_mt"), ("other_generators_same_source", "loo_src"),
                                     ("generator+task+source", "p_all")]}
        B[task]["n"] = int(len(sub))
    report["B_auroc"] = B

    # ---- C. structure inside hallucinated responses -------------------------------------------
    rng = np.random.default_rng(args.seed)
    hr = good[(good.halluc == 1)].copy()
    hr["n_h"] = hr.sent_labels.apply(sum)
    hr = hr[hr.n_h > 0]  # spans that fall outside every sentence would give 0 (should not happen)
    multi = hr[hr.n_sent >= 3]
    C = {"n_hallucinated": int(len(hr)), "n_with_3plus_sentences": int(len(multi))}
    C["share_one_sentence"] = cluster_boot(hr, lambda d: float((d.n_h == 1).mean()), n_boot=args.n_boot // 4)
    C["share_sentences_hallucinated"] = cluster_boot(hr, lambda d: float((d.n_h / d.n_sent).mean()), n_boot=args.n_boot // 4)
    C["share_at_most_half_sentences"] = cluster_boot(hr, lambda d: float((d.n_h <= d.n_sent / 2).mean()), n_boot=args.n_boot // 4)
    # permutation nulls with the count of hallucinated sentences held fixed, per response
    obs_runs, obs_first, null_runs, null_first = [], [], [], []
    for lab in multi.sent_labels:
        h = np.asarray(lab)
        n, k = len(h), int(h.sum())
        if k == n:
            continue
        obs_runs.append(runs(h))
        obs_first.append(int(np.argmax(h)) / (n - 1))
        pr, pf = [], []
        for _ in range(200):
            hp = np.zeros(n, dtype=int)
            hp[rng.choice(n, k, replace=False)] = 1
            pr.append(runs(hp))
            pf.append(int(np.argmax(hp)) / (n - 1))
        null_runs.append(np.mean(pr))
        null_first.append(np.mean(pf))
    obs_runs, null_runs = np.array(obs_runs), np.array(null_runs)
    obs_first, null_first = np.array(obs_first), np.array(null_first)

    def perm_p(diff: np.ndarray) -> float:  # sign-flip test of the mean per-response difference
        d = diff - 0.0
        stat = abs(d.mean())
        flips = rng.choice([-1, 1], size=(args.n_perm, len(d)))
        return float((np.abs((flips * d).mean(axis=1)) >= stat).mean())

    C["runs_test"] = {"n": int(len(obs_runs)), "observed_mean_runs": float(obs_runs.mean()),
                      "null_mean_runs": float(null_runs.mean()),
                      "p_sign_flip": perm_p(obs_runs - null_runs)}
    C["first_onset_relative_position"] = {"observed": float(obs_first.mean()), "null_given_count": float(null_first.mean()),
                                          "p_sign_flip": perm_p(obs_first - null_first)}
    C["by_task"] = {t: {"n": int(len(g)), "share_one_sentence": float((g.n_h == 1).mean()),
                        "mean_n_sent": float(g.n_sent.mean()), "mean_n_hallucinated_sent": float(g.n_h.mean())}
                    for t, g in hr.groupby("task")}
    report["C_structure"] = C

    # ---- D. evidence insufficiency (QA) -------------------------------------------------------
    qa = df[df.task == "QA"].copy()
    correct_ref = qa[(qa.quality == "good") & qa.refusal & (qa.halluc == 0)].source_id.unique()
    wrong_ref = qa[qa.quality == "incorrect_refusal"].source_id.unique()
    qa["src_group"] = np.where(qa.source_id.isin(correct_ref), "insufficient (a refusal was judged correct)",
                               np.where(qa.source_id.isin(wrong_ref), "sufficient (a refusal was judged wrong)",
                                        "no refusal"))
    ans = qa[(qa.quality == "good") & ~qa.refusal]
    D = {"n_qa_sources": int(qa.source_id.nunique()), "n_insufficient_sources": int(len(correct_ref)),
         "n_sufficient_by_wrong_refusal": int(len(wrong_ref))}
    for g, sub in ans.groupby("src_group"):
        D[g] = {"n_answering_responses": int(len(sub)),
                "halluc_rate": cluster_boot(sub, lambda d: float(d.halluc.mean()), n_boot=args.n_boot // 4)}
    refusals = qa[qa.refusal]
    D["refusal_counts_by_model"] = {m: {"correct": int(((g.quality == "good") & (g.halluc == 0)).sum()),
                                        "incorrect": int((g.quality == "incorrect_refusal").sum())}
                                    for m, g in refusals.groupby("model")}
    d2t = good[(good.task == "Data2txt") & (good.halluc == 1)]
    D["data2txt_hallucinated_responses_with_null_caused_span"] = float((d2t.due_to_null > 0).mean())
    D["share_spans_implicit_true"] = float(good.implicit_true.sum() / max(good.n_spans.sum(), 1))
    report["D_insufficiency"] = D

    json.dump(report, open(out / "summary.json", "w"), indent=2, default=float)
    good.drop(columns=["sent_labels", "types"]).to_csv(out / "responses.csv", index=False)
    print(json.dumps(report, indent=2, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
