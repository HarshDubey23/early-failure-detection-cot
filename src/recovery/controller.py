"""
WHAT: The recovery controller as a deployable system (online, one request at a time), and the
      fixed strategies it is compared with, on the same generator and detector interfaces.

WHY:  The offline replay (replay.py) evaluates policies at scale on a shared rollout bank. This
      module runs the same decision logic live, so that latency is measured rather than modelled
      and so the system can be used outside the experiment:

          ctl = RecoveryController(generator, detector, policy="care", budget=4)
          out = ctl.respond(prompt, is_qa=True)
          out["text"], out["action"], out["cost"], out["seconds"]

HOW:  draft -> detector D -> (no flagged sentence: done) -> policy:
        care          1. claim repair: rewrite every flagged sentence (one shared-prefix call each);
                         done if the repaired text passes D
                      2. resample, allocating the remaining budget between restart (k = 0) and
                         backtrack (k = just before the first flagged sentence) by Thompson
                         sampling on each boundary's estimated recoverability v_k per token
                      3. nothing passed D: abstain on QA, otherwise delete what D still flags
        restart_fb    resample from k = 0 until D-clean or the budget is spent; abstain / delete
        backtrack_fb  the same from just before the first flagged sentence
        repairdel     rewrite flagged sentences, then delete what D still flags
        cove          factored Chain-of-Verification (no detector)
      Costs follow replay.py: generator-equivalent tokens, prefix caching within a request.
"""
from __future__ import annotations

import os
import time

import numpy as np

from src.recovery import prompts as P
from src.recovery.candidates import drop_sentences, replace_sentences
from src.recovery.engine import seed_for
from src.recovery.ragtruth import sentence_spans
from src.recovery.replay import flagged_sentences, system_points


class RecoveryController:
    def __init__(self, generator, detector, policy: str = "care", budget: float = 4.0, tau: float = 0.5,
                 temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 600, w_d: float = 0.022,
                 seed: int = 0):
        self.g, self.d = generator, detector
        self.policy, self.b, self.tau = policy, budget, tau
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.w_d, self.seed = w_d, seed

    # ------------------------------------------------------------------ helpers
    def _gen(self, prompts: list, n: int = 1, greedy: bool = False, max_tokens: int | None = None, tag: str = ""):
        seeds = [seed_for(self.seed, tag, i, p[-64:]) for i, p in enumerate(prompts)]
        res, _ = self.g.generate(prompts, n, 0.0 if greedy else self.temperature, 1.0 if greedy else self.top_p,
                                 max_tokens or self.max_tokens, seeds)
        return res

    def _score(self, prompt: str, text: str) -> list:
        r = self.d.score(prompt, text)
        self.cost += self.w_d * (self.g.n_tokens(prompt) + self.g.n_tokens(text))
        return r.get("sent_scores") or []

    def _clean(self, scores: list) -> bool:
        return max(scores or [0.0]) < self.tau

    def _done(self, out: dict, text: str, action: str, t0: float) -> dict:
        return {**out, "text": text, "action": action, "cost": self.cost, "seconds": time.time() - t0}

    # ------------------------------------------------------------------ main entry
    def respond(self, prompt: str, is_qa: bool, draft: str | None = None) -> dict:
        t0 = time.time()
        self.cost = 0.0
        if draft is None:
            r = self._gen([self.g.fmt(prompt)], tag="draft")[0]
            draft, draft_tok, p0 = r["samples"][0][0].strip(), r["samples"][0][1], r["prompt_tokens"]
        else:
            draft_tok, p0 = self.g.n_tokens(draft), self.g.n_tokens(self.g.fmt(prompt))
        out = {"draft": draft}
        if self.policy == "base":
            return self._done(out, draft, "keep", t0)
        if self.policy == "cove":
            return self._done(out, self._cove(prompt, draft), "cove", t0)
        scores = self._score(prompt, draft)
        unit = p0 + draft_tok + self.cost  # draft plus its detector pass
        out["budget_unit"] = unit
        self.cost = 0.0  # the draft's detector pass is common to every D-triggered policy
        if self._clean(scores):
            return self._done(out, draft, "keep", t0)
        if self.policy == "repairdel":
            repaired, rsc = self._repair(prompt, draft, scores)
            return self._done(out, drop_sentences(repaired, flagged_sentences(rsc, self.tau)), "repair+delete", t0)

        budget = self.b * unit
        spans = sentence_spans(draft)
        k_back = max(system_points(scores, self.tau))
        arms = sorted({0, k_back})
        prefix = {k: (draft[: spans[k][0]] if k > 0 else "") for k in arms}
        rest = {k: max(8.0, draft_tok * (len(draft) - len(prefix[k])) / max(len(draft), 1)) for k in arms}
        pre_tok = {k: self.g.n_tokens(prefix[k]) if k else 0 for k in arms}
        paid = set()

        def marginal(k):
            return (0 if k in paid else pre_tok[k]) + rest[k] + self.w_d * (p0 + pre_tok[k] + rest[k])

        def affordable(k):
            return self.cost + marginal(k) <= budget

        def sample(k, n=2):
            """n continuations from boundary k in one request; returns (full text, D scores) pairs."""
            if not affordable(k):
                return []
            res = self._gen([self.g.fmt(prompt, prefix[k])], n=n, tag=f"k{k}c{self.cost:.0f}")[0]
            if res is None:
                return []
            if k not in paid:
                paid.add(k)
                self.cost += pre_tok[k]
            got = []
            for text, tok in res["samples"]:
                self.cost += tok
                full = (prefix[k] + text).strip()
                got.append((full, self._score(prompt, full)))
            return got

        repaired, rsc = None, None
        if self.policy == "care":
            repaired, rsc = self._repair(prompt, draft, scores)
            if self._clean(rsc):
                return self._done(out, repaired, "repair", t0)
        if self.policy in ("care", "restart_fb", "backtrack_fb"):
            rng = np.random.default_rng(seed_for(self.seed, "thompson", prompt[-64:]))
            fails = {k: 0 for k in arms}
            while True:
                if self.policy == "restart_fb":
                    choice = [0]
                elif self.policy == "backtrack_fb":
                    choice = [k_back]
                else:
                    choice = arms
                afford = [k for k in choice if affordable(k)]
                if not afford:
                    break
                v = {k: min(rng.beta(1.0, 1.0 + fails[k]), 1 - 1e-9) for k in afford}
                k = max(afford, key=lambda a: -np.log1p(-v[a]) / max(marginal(a), 1.0))
                got = sample(k)
                if not got:
                    break
                for full, sc in got:
                    if self._clean(sc):
                        return self._done(out, full, "restart" if k == 0 else "backtrack", t0)
                    fails[k] += 1
        else:
            raise ValueError(self.policy)
        if is_qa:
            return self._done(out, P.REFUSAL, "abstain", t0)
        if repaired is not None:
            return self._done(out, drop_sentences(repaired, flagged_sentences(rsc, self.tau)), "repair+delete", t0)
        return self._done(out, drop_sentences(draft, flagged_sentences(scores, self.tau)), "delete", t0)

    # ------------------------------------------------------------------ repair and CoVe
    def _repair(self, prompt: str, draft: str, scores: list) -> tuple:
        """Rewrite each flagged sentence (calls share the task + source + draft prefix)."""
        spans = sentence_spans(draft)
        fl = flagged_sentences(scores, self.tau)
        prompts = [self.g.fmt(P.repair_prompt(prompt, draft, j, draft[spans[j - 1][0]: spans[j - 1][1]])) for j in fl]
        res = self._gen(prompts, greedy=True, max_tokens=160, tag="repair")
        shared = self.g.n_tokens(os.path.commonprefix(prompts)) if len(prompts) > 1 else 0
        self.cost += shared
        rw = {}
        for j, r in zip(fl, res):
            if r is None:
                rw[j] = None
                continue
            self.cost += max(r["prompt_tokens"] - shared, 0) + r["samples"][0][1]
            rw[j] = P.parse_repair(r["samples"][0][0])
        repaired = replace_sentences(draft, rw)
        return repaired, self._score(prompt, repaired)

    def _cove(self, prompt: str, draft: str) -> str:
        r = self._gen([self.g.fmt(P.cove_plan_prompt(prompt, draft))], greedy=True, max_tokens=200, tag="plan")[0]
        if r is None:
            return draft
        self.cost += r["prompt_tokens"] + r["samples"][0][1]
        qs = P.parse_questions(r["samples"][0][0])
        qa = []
        if qs:
            res = self._gen([self.g.fmt(P.cove_answer_prompt(prompt, q)) for q in qs], greedy=True, max_tokens=80,
                            tag="ans")
            for q, a in zip(qs, res):
                if a is not None:
                    self.cost += a["prompt_tokens"] + a["samples"][0][1]
                    qa.append((q, a["samples"][0][0].strip()))
        r = self._gen([self.g.fmt(P.cove_revise_prompt(prompt, draft, qa))], greedy=True, tag="rev")[0]
        if r is None:
            return draft
        self.cost += r["prompt_tokens"] + r["samples"][0][1]
        return r["samples"][0][0].strip()
