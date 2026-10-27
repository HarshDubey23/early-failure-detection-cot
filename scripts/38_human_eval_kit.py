"""Stage 38 (CPU): a blinded human comparison kit for Run R, and the script that scores it.

No human result is produced here. The kit samples RAGTruth items where the controller (CARE) and
a reference policy return different texts, shows the source and the two outputs as A/B in random
order, and asks the annotator, for each output, which sentences are not supported by the source,
and which output they would rather receive. The key that maps A/B to policies is written to a
separate file that annotators must not open.

  python scripts/38_human_eval_kit.py make  --ref restart_fb --n 80 --out results/recovery/human_eval
  python scripts/38_human_eval_kit.py score --out results/recovery/human_eval   # after people fill it in

Two independent annotators should each fill a copy of the sheet (columns *_A_unsupported,
*_B_unsupported as comma-separated sentence numbers, preferred in {A, B, tie}); `score` reports
each policy's hallucination rate per annotator, the paired difference, and Cohen's kappa between
annotators on the response-level labels.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import candidates as C  # noqa: E402
from src.recovery import replay as R  # noqa: E402
from src.recovery import stats as S  # noqa: E402
from src.recovery.prompts import REFUSAL  # noqa: E402
from src.recovery.ragtruth import sentence_spans  # noqa: E402


def make(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = C.load_run(args.run_dir, args.items)
    D = {r["cand_id"]: r["sent_scores"] for r in C.read_jsonl(Path(args.run_dir) / "scores_D.jsonl")
         if r.get("sent_scores") is not None}
    texts = {c["cand_id"]: c["text"] for st in (1, 2) for c in C.build_candidates(run, D, args.tau, st)}
    per = pd.read_csv(args.per_item_outputs)
    a = per[(per.policy == "care") & (per.budget == args.budget)].set_index("item_id")
    b = per[(per.policy == args.ref) & (per.budget == (args.budget if args.ref.endswith("_fb") else 0))].set_index("item_id")
    common = sorted(i for i in a.index.intersection(b.index) if i in run["items"] and a.loc[i, "cid"] != b.loc[i, "cid"])
    rng = random.Random(args.seed)
    rng.shuffle(common)
    rows, key = [], []
    for n, iid in enumerate(common[: args.n]):
        it = run["items"][iid]
        outs = [("care", a.loc[iid, "cid"]), (args.ref, b.loc[iid, "cid"])]
        rng.shuffle(outs)
        rec = {"pair": n + 1, "task": it["task"], "source": it["prompt"]}
        for lab, (pol, cid) in zip("AB", outs):
            t = REFUSAL if cid == R.REFUSE else texts.get(cid, "")
            sents = [t[x:y].strip() for x, y in sentence_spans(t)]
            rec[f"{lab}_numbered"] = "\n".join(f"[{j + 1}] {s}" for j, s in enumerate(sents))
            rec[f"{lab}_unsupported"] = ""
            key.append({"pair": n + 1, "label": lab, "policy": pol, "cid": cid, "item_id": iid})
        rec["preferred"] = ""
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "sheet_blank.csv", index=False)
    pd.DataFrame(key).to_csv(out / "KEY_do_not_open.csv", index=False)
    # Commit only this hash; the key itself stays off the repository (.gitignore) until rating is done.
    import hashlib
    digest = hashlib.sha256((out / "KEY_do_not_open.csv").read_bytes()).hexdigest()
    (out / "KEY.sha256").write_text(f"{digest}  KEY_do_not_open.csv\n")
    with open(out / "sheet.html", "w", encoding="utf-8") as fh:
        fh.write("<meta charset='utf-8'><h1>Which sentences are not supported by the source?</h1>")
        for r in rows:
            fh.write(f"<hr><h2>Pair {r['pair']} ({r['task']})</h2><details><summary>Source and task</summary>"
                     f"<pre style='white-space:pre-wrap'>{html.escape(r['source'])}</pre></details>")
            for lab in "AB":
                fh.write(f"<h3>Output {lab}</h3><pre style='white-space:pre-wrap'>{html.escape(r[lab + '_numbered'])}</pre>")
    print(f"{len(rows)} pairs written to {out} (CARE vs {args.ref}); fill sheet_blank.csv copies, one per annotator")


def score(args) -> None:
    out = Path(args.out)
    key = pd.read_csv(out / "KEY_do_not_open.csv")
    sheets = sorted(out.glob("sheet_annotator_*.csv"))
    if not sheets:
        raise SystemExit("no filled sheets (sheet_annotator_<name>.csv) found")
    res, labels = {}, {}
    for sh in sheets:
        df = pd.read_csv(sh).fillna("")
        rows = []
        for _, r in df.iterrows():
            for lab in "AB":
                k = key[(key.pair == r.pair) & (key.label == lab)].iloc[0]
                bad = [x for x in str(r[f"{lab}_unsupported"]).replace(" ", "").split(",") if x]
                rows.append({"pair": r.pair, "policy": k.policy, "halluc": float(len(bad) > 0),
                             "preferred": float(str(r.preferred).strip().upper() == lab)})
        R_ = pd.DataFrame(rows)
        wide = R_.pivot(index="pair", columns="policy", values="halluc")
        pols = list(wide.columns)
        d = (wide[pols[0]] - wide[pols[1]]).to_numpy()
        res[sh.stem] = {"rates": {p: float(wide[p].mean()) for p in pols},
                        f"{pols[0]}_minus_{pols[1]}": S.boot_ci(d, np.arange(len(d))),
                        "preferred_share": R_.groupby("policy").preferred.mean().to_dict()}
        labels[sh.stem] = R_.set_index(["pair", "policy"]).halluc
    if len(labels) >= 2:
        names = list(labels)
        x, y = labels[names[0]].align(labels[names[1]], join="inner")
        res["agreement"] = {"annotators": names[:2], **S.binary_agreement(x.to_numpy(), y.to_numpy())}
    json.dump(res, open(out / "human_eval_results.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--items", nargs="*", default=["data/recovery/items.ragtruth.llama-2-7b-chat.jsonl"])
    ap.add_argument("--run-dir", default="results/recovery/run")
    ap.add_argument("--per-item-outputs", default="results/recovery/final/outputs_round0.csv")
    ap.add_argument("--ref", default="restart_fb")
    ap.add_argument("--budget", type=float, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/recovery/human_eval")
    args = ap.parse_args()
    make(args) if args.mode == "make" else score(args)


if __name__ == "__main__":
    main()
