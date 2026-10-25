"""Stage 36 (GPU): run the controller and the main baselines LIVE, one request at a time.

Measures wall-clock latency of the deployed system (src/recovery/controller.py) instead of
modelling it, on a small seeded subset of detector-flagged and unflagged items, and measures the
generator's single-request prefill and decode rates for the replay's latency model
(results/recovery/run/live_rates.json). The live outputs are not used for the hallucination
metrics (the replay covers all items); they show that the deployed loop does what the replay
assumes.

Usage (Kaggle, 2x T4):
  python scripts/36_live.py --items data/recovery/items.ragtruth.llama-2-7b-chat.jsonl \
      --generator llama-2-7b-chat --n-flagged 12 --n-unflagged 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recovery import candidates as C  # noqa: E402
from src.recovery import engine as E  # noqa: E402
from src.recovery.controller import RecoveryController  # noqa: E402

GEN_IDS = import_module("32_generate").GEN_IDS
PARAMS_B = {"llama-2-7b-chat": 6.74, "llama-3.1-8b-instruct": 8.03}


def main() -> None:
    ap = argparse.ArgumentParser(description="36: live latency")
    ap.add_argument("--items", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--run-dir", default="results/recovery/run")
    ap.add_argument("--policies", nargs="+", default=["care", "restart_fb", "backtrack_fb", "repairdel", "cove"])
    ap.add_argument("--budget", type=float, default=4)
    ap.add_argument("--n-flagged", type=int, default=12)
    ap.add_argument("--n-unflagged", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.78)
    ap.add_argument("--backend", choices=["real", "mock"], default="real")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run = C.load_run(run_dir, [args.items])
    dsc = {r["item_id"]: r["sent_scores"] for r in C.read_jsonl(run_dir / "scores_D.jsonl")
           if r.get("kind") == "draft" and r.get("sent_scores") is not None}
    ids = [i for i in run["items"] if i in run["drafts"] and i in dsc]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    flagged = [i for i in ids if max(dsc[i] or [0]) >= args.tau][: args.n_flagged]
    unflagged = [i for i in ids if max(dsc[i] or [0]) < args.tau][: args.n_unflagged]
    chosen = flagged + unflagged
    print(f"live: {len(flagged)} flagged + {len(unflagged)} unflagged items", flush=True)

    if args.backend == "real":
        gen = E.VLLMGenerator(GEN_IDS[args.generator], args.generator, tp=args.tp, gpu_mem=args.gpu_mem)
        det = E.LettuceDetector("KRLabsOrg/lettucedect-base-modernbert-en-v1", device="cuda:0")
    else:
        gen, det = E.MockGenerator(args.generator), E.MockDetector(noise=0.15, seed=1)  # same as stage 33 mock D
    w_d = 0.149 / PARAMS_B.get(args.generator, 7.0)

    # ---- single-request rates: decode (one long sample) and prefill (max_tokens = 1)
    rates = {"source": f"measured live, {args.generator}, batch size 1"}
    it0 = run["items"][chosen[0]]
    p = gen.fmt(it0["prompt"])
    gen.generate([p], 1, 0.7, 0.95, 8, [1])  # warm-up
    t = time.time()
    r, _ = gen.generate([p], 1, 0.7, 0.95, 1, [2])
    t_pre = time.time() - t
    t = time.time()
    r2, _ = gen.generate([p], 1, 0.7, 0.95, 256, [3])
    t_all = time.time() - t
    n_dec = r2[0]["samples"][0][1]
    rates["prefill"] = r[0]["prompt_tokens"] / max(t_pre, 1e-3)
    rates["decode"] = max(n_dec - 1, 1) / max(t_all - t_pre, 1e-3)
    t = time.time()
    det.score(it0["prompt"], run["drafts"][chosen[0]]["draft"])
    rates["detector_seconds_per_call"] = time.time() - t
    json.dump(rates, open(run_dir / f"live_rates.{args.generator}.json", "w"), indent=2)
    json.dump(rates, open(run_dir / "live_rates.json", "w"), indent=2)
    print("rates", rates, flush=True)

    out = open(run_dir / f"live.{args.generator}.jsonl", "a", encoding="utf-8")
    for iid in chosen:
        it = run["items"][iid]
        draft = run["drafts"][iid]["draft"]
        # end-to-end including the draft, measured once per item with the base policy
        base = RecoveryController(gen, det, policy="base", budget=args.budget, w_d=w_d).respond(it["prompt"], it["is_qa"])
        out.write(json.dumps({"item_id": iid, "policy": "base_generate_draft", "seconds": base["seconds"],
                              "flagged": iid in flagged}) + "\n")
        for pol in args.policies:
            ctl = RecoveryController(gen, det, policy=pol, budget=args.budget, tau=args.tau, w_d=w_d,
                                     temperature=it.get("draft_temperature") or 0.7)
            res = ctl.respond(it["prompt"], it["is_qa"], draft=draft)
            out.write(json.dumps({"item_id": iid, "policy": pol, "budget": args.budget, "flagged": iid in flagged,
                                  "seconds": res["seconds"], "cost": res["cost"], "action": res["action"],
                                  "budget_unit": res.get("budget_unit"), "text": res["text"],
                                  "backend": gen.backend}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{iid} {pol}: {res['action']} {res['seconds']:.1f}s cost {res['cost']:.0f}", flush=True)
    out.close()
    if args.backend == "real":
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
