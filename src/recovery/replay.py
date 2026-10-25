"""
WHAT: Offline replay of recovery policies over the rollout bank, at a matched compute budget.

WHY:  Every policy (fixed strategies, verify-and-revise, the commitment-aware controller and its
      ablations) chooses among texts that already exist in the bank, so all of them are scored on
      the same samples, paired item by item, without re-running the GPU for each policy. A policy
      only sees what it could see when deployed: the draft, detector D's sentence scores, and the
      D scores of the samples it has paid for. The evaluators (E1 judge, E2 detector, human labels)
      are read only to score the final output (and by the oracle reference policy).

HOW:  For each item and each of R random orders of the bank samples, a policy draws samples in
      that order, pays for them and stops by its own rule. Outcomes are averaged over the R orders
      (the expectation over sampling), then over items.

Cost (generator-equivalent tokens; FLOPs ~ parameters x tokens). Primary view, with prefix caching
(a prompt prefix already processed for the same item is not processed again):
  - draft: prompt P0 + generated tokens L, and one detector pass  (common to all policies)
  - a continuation from boundary k: the kept draft sentences (prefill, once per boundary; zero for
    a restart, whose prompt is cached from the draft), its generated tokens, and a detector pass
  - claim repair: the prefix shared by all repair calls of the item once, each call's own tail,
    the generated tokens, and a detector pass on the repaired text
  - CoVe: the shared prefix once, each call's tail and generated tokens
  - a detector pass costs w_D x (prompt + response tokens), w_D = params(D) / params(generator)
The no-cache view (every call pays its full prompt) is reported alongside. The recovery budget is
B = b x U with U = the cost of the draft and its detector pass; a new continuation from k may
start only while spent + its expected cost (prefill if unpaid + the length of the draft's
remainder + its detector pass) is within B.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

REFUSE = "<REFUSE>"


def flagged_sentences(scores: list, tau: float) -> list:
    return [j + 1 for j, s in enumerate(scores or []) if s >= tau]


def first_flag(scores: list, tau: float) -> int | None:
    """1-based index of the first sentence detector D flags, or None."""
    f = flagged_sentences(scores, tau)
    return f[0] if f else None


def system_points(scores: list, tau: float) -> list:
    """Boundaries the controller may resample from: 0 (restart), j*-1 (just before D's first
    flagged sentence) and their midpoint, when distinct.

    Example:
        >>> system_points([0.1, 0.2, 0.3, 0.1, 0.9, 0.2], 0.5)
        [0, 2, 4]
        >>> system_points([0.9, 0.1], 0.5)
        [0]
    """
    j = first_flag(scores, tau)
    if j is None:
        return []
    return sorted({0, j - 1, (j - 1) // 2})


@dataclass
class Sample:
    cid: str
    gen_tokens: float
    d_max: float
    det_tokens: float  # tokens one detector pass reads (prompt + response)


@dataclass
class Point:
    k: int
    prefill: float          # extra prefill with caching (kept draft sentences; 0 for k = 0)
    prefill_nocache: float  # generation prompt + kept sentences
    est_gen: float          # expected continuation length (the draft's remainder)
    est_det: float          # expected detector tokens of a candidate from k
    samples: list = field(default_factory=list)


@dataclass
class ItemCtx:
    iid: str
    source_id: str
    is_qa: bool
    flagged: bool
    draft_cid: str
    draft_dmax: float
    budget_unit: float            # cost of the draft and its detector pass
    points: dict                  # k -> Point (system points only)
    w_d: float
    delete_cid: str | None = None
    repair_cid: str | None = None
    repair_dmax: float = 1.0
    repairdel_cid: str | None = None
    repair_cost: float = 0.0      # cached view, including the detector pass on the repaired text
    repair_cost_nocache: float = 0.0
    repair_steps: list = field(default_factory=list)
    cove_cid: str | None = None
    cove_cost: float = 0.0
    cove_cost_nocache: float = 0.0
    cove_steps: list = field(default_factory=list)
    true_v: dict = field(default_factory=dict)  # k -> evaluator-clean share (oracle policy only)
    common: float = 0.0           # draft generation (prompt + tokens), paid by every policy
    det_draft: float = 0.0        # detector pass on the draft, paid by D-triggered policies
    draft_steps: list = field(default_factory=list)


@dataclass
class Outcome:
    cid: str
    cost: float = 0.0          # recovery cost (cached view), charged to the budget
    cost_nocache: float = 0.0
    gen_tokens: float = 0.0    # generated tokens in recovery
    calls: int = 0             # generator calls in recovery
    steps: list = field(default_factory=list)  # sequential steps: (prefill tokens, longest decode)
    action: str = "keep"


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Sampler:
    """Draws bank samples for one item in a fixed random order per boundary and keeps the bill."""

    def __init__(self, ctx: ItemCtx, rnd: int, budget: float, tau: float):
        self.ctx, self.budget, self.tau = ctx, budget, tau
        self.next = {k: 0 for k in ctx.points}
        self.order = {k: np.random.default_rng(_seed(ctx.iid, k, rnd)).permutation(len(p.samples))
                      for k, p in ctx.points.items()}
        self.paid_prefill: set = set()
        self.out = Outcome(cid=ctx.draft_cid)
        self.drawn: list = []  # (k, Sample)

    def marginal(self, k: int) -> float:
        p = self.ctx.points[k]
        return (0.0 if k in self.paid_prefill else p.prefill) + p.est_gen + self.ctx.w_d * p.est_det

    def can_draw(self, k: int) -> bool:
        p = self.ctx.points.get(k)
        if p is None or self.next[k] >= len(p.samples):
            return False
        return self.out.cost + self.marginal(k) <= self.budget + 1e-9

    def can_pay(self, x: float) -> bool:
        return self.out.cost + x <= self.budget + 1e-9

    def charge(self, cost: float, cost_nocache: float, steps: list, calls: int = 1) -> None:
        self.out.cost += cost
        self.out.cost_nocache += cost_nocache
        self.out.steps += list(steps)
        self.out.calls += calls

    def draw(self, ks: list) -> list:
        """One parallel step: one sample per entry of ks (repeats allowed), while affordable."""
        got, pre, dec = [], 0.0, 0.0
        for k in ks:
            if not self.can_draw(k):
                continue
            p = self.ctx.points[k]
            s = p.samples[self.order[k][self.next[k]]]
            self.next[k] += 1
            if k not in self.paid_prefill:
                self.paid_prefill.add(k)
                self.out.cost += p.prefill
                self.out.cost_nocache += p.prefill_nocache
                pre += p.prefill
            det = self.ctx.w_d * s.det_tokens
            self.out.cost += s.gen_tokens + det
            self.out.cost_nocache += s.gen_tokens + det
            self.out.gen_tokens += s.gen_tokens
            self.out.calls += 1
            dec = max(dec, s.gen_tokens)
            got.append((k, s))
            self.drawn.append((k, s))
        if got:
            self.out.steps.append((pre, dec))
        return got

    def clean(self, s: Sample) -> bool:
        return s.d_max < self.tau

    def best_drawn(self):
        """Least-flagged sample drawn so far (ties: later boundary), or None."""
        if not self.drawn:
            return None
        return min(self.drawn, key=lambda ks: (ks[1].d_max, -ks[0]))


def _sequential(smp: Sampler, k: int, batch: int = 2) -> bool:
    """Draw from boundary k in parallel batches until a D-clean sample or the budget/bank runs out."""
    while smp.can_draw(k):
        got = smp.draw([k] * batch)
        clean = [s for _, s in got if smp.clean(s)]
        if clean:
            smp.out.cid, smp.out.action = clean[0].cid, ("restart" if k == 0 else "backtrack")
            return True
    return False


def _thompson(ctx: ItemCtx, smp: Sampler, arms: list, rnd: int, batch: int = 2) -> bool:
    """Commitment-aware allocation between restarting (k = 0) and backtracking to just before the
    first flagged sentence. Each boundary k is an arm whose unknown value v_k is the probability that
    a continuation from k passes detector D, i.e. whether the response is still recoverable there.
    A Beta(1, 1) posterior on v_k is updated with every paid continuation; each step samples v~_k from
    it and draws `batch` continuations from the arm with the best value per token,
    -log(1 - v~_k) / marginal cost_k (the rate at which the arm buys P(at least one clean
    continuation)). Stops at the first D-clean continuation or when no arm is affordable."""
    rng = np.random.default_rng(_seed(ctx.iid, rnd, "thompson"))
    fails = {k: 0 for k in arms}
    while True:
        afford = [k for k in arms if smp.can_draw(k)]
        if not afford:
            return False
        draws = {k: min(rng.beta(1.0, 1.0 + fails[k]), 1 - 1e-9) for k in afford}
        k = max(afford, key=lambda a: -np.log1p(-draws[a]) / max(smp.marginal(a), 1.0))
        for kk, s in smp.draw([k] * batch):
            if smp.clean(s):
                smp.out.cid, smp.out.action = s.cid, ("restart" if kk == 0 else "backtrack")
                return True
            fails[kk] += 1


def _abstain_or_delete(ctx: ItemCtx, smp: Sampler, abstain_qa: bool = True, repaired: bool = False) -> None:
    """Nothing faithful was found within the budget. QA: abstain (the source may not contain the
    answer). Other tasks: drop the sentences D still flags, from the repaired text if a repair was
    paid for, otherwise from the draft. Neither costs generation."""
    if ctx.is_qa and abstain_qa:
        smp.out.cid, smp.out.action = REFUSE, "abstain"
    elif repaired and ctx.repairdel_cid is not None:
        smp.out.cid, smp.out.action = ctx.repairdel_cid, "repair+delete"
    elif ctx.delete_cid is not None:
        smp.out.cid, smp.out.action = ctx.delete_cid, "delete"
    else:
        smp.out.cid, smp.out.action = ctx.draft_cid, "keep"


def _keep_best(ctx: ItemCtx, smp: Sampler) -> None:
    """No fallback: output the least-flagged sample drawn, or the draft if it is less flagged."""
    best = smp.best_drawn()
    if best is not None and best[1].d_max < ctx.draft_dmax:
        smp.out.cid, smp.out.action = best[1].cid, ("restart" if best[0] == 0 else "backtrack")


def _care(ctx: ItemCtx, smp: Sampler, variant: str, rnd: int, k_back: int, pts: list) -> None:
    """The recovery controller: (1) claim repair, if its result passes D; (2) resample, allocating the
    remaining budget between restart and backtrack by their estimated recoverability; (3) abstain on
    QA, otherwise delete what D still flags."""
    repaired = False
    if variant != "norepair" and ctx.repair_cid is not None and smp.can_pay(ctx.repair_cost):
        smp.charge(ctx.repair_cost, ctx.repair_cost_nocache, ctx.repair_steps)
        repaired = True
        if ctx.repair_dmax < smp.tau:
            smp.out.cid, smp.out.action = ctx.repair_cid, "repair"
            return
    arms = sorted({0, k_back})
    if variant == "restart":
        found = _sequential(smp, 0)
    elif variant == "backtrack":
        found = _sequential(smp, k_back)
    elif variant == "oracle":  # not deployable: the evaluator's v_k picks the boundary
        found = _sequential(smp, max(arms, key=lambda k: (ctx.true_v.get(k, 0.0), k)))
    elif variant == "probe3":  # alternative estimator: one probe per system point, then commit
        got = smp.draw(sorted(pts, reverse=True))
        clean = [(k, s) for k, s in got if smp.clean(s)]
        if clean:
            k, s = max(clean, key=lambda ks: ks[0])
            smp.out.cid, smp.out.action = s.cid, ("restart" if k == 0 else "backtrack")
            return
        v_hat = {k: 1.0 - s.d_max for k, s in got}
        found = _sequential(smp, max(v_hat, key=lambda k: (v_hat[k], k)) if v_hat else k_back)
    else:
        found = _thompson(ctx, smp, arms, rnd)
    if not found:
        _abstain_or_delete(ctx, smp, abstain_qa=(variant != "noabstain"), repaired=repaired)


def run_policy(name: str, ctx: ItemCtx, rnd: int, b: float, tau: float) -> Outcome:
    """Output of policy `name` on one item for replay round `rnd` at budget b (x draft cost)."""
    smp = Sampler(ctx, rnd, b * ctx.budget_unit, tau)
    o = smp.out
    if name == "base":
        return o
    if name == "cove_all":  # unconditional verify-and-revise: no detector
        if ctx.cove_cid is not None:
            smp.charge(ctx.cove_cost, ctx.cove_cost_nocache, ctx.cove_steps, calls=len(ctx.cove_steps))
            o.cid, o.action = ctx.cove_cid, "cove"
        return o
    if not ctx.flagged:  # every other policy is triggered by detector D on the draft
        return o
    pts = sorted(ctx.points)
    k_back = max(pts) if pts else 0  # j* - 1
    if name == "cove_flag":
        if ctx.cove_cid is not None:
            smp.charge(ctx.cove_cost, ctx.cove_cost_nocache, ctx.cove_steps, calls=len(ctx.cove_steps))
            o.cid, o.action = ctx.cove_cid, "cove"
        return o
    if name == "abstain_flag":
        if ctx.is_qa:
            o.cid, o.action = REFUSE, "abstain"
        else:  # a summary or data description cannot "abstain"; drop the flagged sentences
            o.cid, o.action = (ctx.delete_cid or ctx.draft_cid), "delete"
        return o
    if name == "delete":
        o.cid, o.action = (ctx.delete_cid or ctx.draft_cid), "delete"
        return o
    if name in ("repair", "repairdel"):
        cid = ctx.repair_cid if name == "repair" else ctx.repairdel_cid
        if cid is not None:
            smp.charge(ctx.repair_cost, ctx.repair_cost_nocache, ctx.repair_steps)
            o.cid, o.action = cid, ("repair" if name == "repair" else "repair+delete")
        return o
    family, _, variant = name.partition("@")
    if family in ("restart", "backtrack", "restart_fb", "backtrack_fb"):
        k = 0 if family.startswith("restart") else k_back
        if not _sequential(smp, k):
            if family.endswith("_fb"):
                _abstain_or_delete(ctx, smp)
            else:
                _keep_best(ctx, smp)
        return o
    if family == "cascade_fb":  # fixed rule: half the budget before the flag, then restarts
        full = smp.budget
        smp.budget = full / 2
        if _sequential(smp, k_back):
            return o
        smp.budget = full
        if not _sequential(smp, 0):
            _abstain_or_delete(ctx, smp)
        return o
    if family == "care":
        _care(ctx, smp, variant, rnd, k_back, pts)
        return o
    raise ValueError(f"unknown policy {name}")


POLICIES_FIXED = ["base", "abstain_flag", "delete", "repair", "repairdel", "cove_all", "cove_flag"]
POLICIES_BUDGETED = ["restart", "backtrack", "restart_fb", "backtrack_fb", "cascade_fb", "care", "care@restart",
                     "care@backtrack", "care@norepair", "care@noabstain", "care@probe3", "care@oracle"]


def latency(steps: list, prefill_rate: float, decode_rate: float) -> float:
    """Seconds for sequential steps of (prefill tokens, longest decode in the step)."""
    return float(sum(pre / prefill_rate + dec / decode_rate for pre, dec in steps))
