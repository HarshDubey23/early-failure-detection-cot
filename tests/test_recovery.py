"""Tests for the hallucination-recovery system (src/recovery): segmentation, parsing, the cost
model, every policy's decision logic, and planted-effect simulations showing that the replay
(a) finds a benefit of choosing the resampling boundary when one exists and (b) does not create
one when it does not."""
from __future__ import annotations

import numpy as np

from src.recovery import candidates as C
from src.recovery import prompts as P
from src.recovery import ragtruth as rt
from src.recovery import replay as R
from src.recovery import stats as S

# ------------------------------------------------------------------ text handling


def test_sentence_spans_cover_text_and_respect_abbreviations():
    t = "Dr. Smith paid $3.50 at 5 p.m. on Monday. He left!\n\n- next item\nLast one"
    spans = rt.sentence_spans(t)
    assert spans[0][0] == 0 and spans[-1][1] == len(t)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert [t[a:b].strip() for a, b in spans] == ["Dr. Smith paid $3.50 at 5 p.m. on Monday.", "He left!",
                                                 "- next item", "Last one"]


def test_sentence_labels_and_mechanism_points():
    t = "One. Two. Three. Four."
    assert rt.sentence_labels(t, [{"start": 5, "end": 9}]) == [0, 1, 0, 0]
    mp = rt.mechanism_points(t, [{"start": 5, "end": 9}], "x", [0.5])
    assert mp["e"] == 2 and mp["points"] == [0, 1, 2, 3]
    clean = rt.mechanism_points(t, [], "x", [0.5, 0.75])
    assert clean["e"] is None and all(a >= 1 for a in clean["anchors"])
    assert set(clean["points"]) >= {0} | {a - 1 for a in clean["anchors"]}


def test_drop_and_replace_sentences():
    t = "A is red. B is blue. C is green."
    assert C.drop_sentences(t, [2]) == "A is red. C is green."
    assert C.replace_sentences(t, {1: "A is pink.", 3: None}) == "A is pink. B is blue."


def test_refusal_detection():
    assert C.is_refusal_text(P.REFUSAL)
    assert C.is_refusal_text("")
    assert not C.is_refusal_text("Unable to answer based on given passages. However, the store opens at 9 am "
                                 "and closes at 5 pm on weekdays, and it sells bread, milk and cheese.")


def test_judge_and_repair_parsing():
    assert P.parse_judge('{"unsupported": [2, 5, 9]}', 6) == [2, 5]
    assert P.parse_judge("Sure: {\"unsupported\": []}", 3) == []
    assert P.parse_judge("no json here", 3) is None
    assert P.parse_repair("DELETE") is None
    assert P.parse_repair('"The shop opens at 9."\nextra') == "The shop opens at 9."
    # real Llama-2 outputs from the Run R smoke test
    assert P.parse_repair("  Sure, here's the rewritten sentence:\n\nThe shop opens at 9.") == "The shop opens at 9."
    assert P.parse_repair("  Sure, here's the rewritten sentence:\n\nDELETE - not in the source.") is None
    assert P.parse_repair("  REWRITE SENTENCE 2:\nThe band has four members.") == "The band has four members."
    assert P.parse_repair("Sure, here's the rewritten sentence:\n\nREWRITE: It opened in 2022.") == "It opened in 2022."
    assert P.parse_repair("SENTENCE 3: It opened in 2022.") == "It opened in 2022."
    assert P.parse_final("Sure, here is the final response to the original task:\n\nDRAFT RESPONSE:\nA is B.\nC.") == "A is B.\nC."
    assert P.parse_final("FINAL RESPONSE:\nA is B.") == "A is B."
    assert P.parse_final("FINAL RESPONSE: A is B.") == "A is B."
    assert P.parse_final("A is B.") == "A is B."
    assert P.parse_questions("1. Who?\n2) When did it open?\nnot a question") == ["Who?", "When did it open?"]


def test_system_points():
    assert R.system_points([0.1, 0.2, 0.3, 0.1, 0.9], 0.5) == [0, 2, 4]
    assert R.system_points([0.1, 0.8], 0.5) == [0, 1]
    assert R.system_points([0.1, 0.2], 0.5) == []

# ------------------------------------------------------------------ policies on hand-built items


def _ctx(clean_at: dict, n_per_point: int = 8, is_qa: bool = False, iid: str = "it", rng=None,
         p0: float = 600.0, draft_tok: float = 200.0, w_d: float = 0.022) -> R.ItemCtx:
    """Item whose draft is flagged at sentence 3, so the system points are k = 0, 1, 2 (k_back = 2).
    clean_at[k] = number of D-clean samples among the n_per_point at boundary k. A continuation from
    k keeps k of 4 equal draft sentences."""
    pts = {}
    for k in (0, 1, 2):
        kept = draft_tok * k / 4
        gen = draft_tok - kept
        samples = []
        for i in range(n_per_point):
            dmax = 0.1 if i < clean_at.get(k, 0) else (float(rng.uniform(0.5, 1.0)) if rng is not None else 0.9)
            samples.append(R.Sample(cid=f"{iid}|k{k}|s{i}", gen_tokens=gen, d_max=dmax, det_tokens=p0 + draft_tok))
        pts[k] = R.Point(k=k, prefill=kept, prefill_nocache=p0 + kept, est_gen=gen, est_det=p0 + draft_tok,
                         samples=samples)
    unit = p0 + draft_tok + w_d * (p0 + draft_tok)
    return R.ItemCtx(iid=iid, source_id=iid, is_qa=is_qa, flagged=True, draft_cid=f"{iid}|draft", draft_dmax=0.9,
                     budget_unit=unit, points=pts, w_d=w_d, delete_cid=f"{iid}|delete", repair_cid=f"{iid}|repair",
                     repair_dmax=0.9, repairdel_cid=f"{iid}|repairdel", repair_cost=700.0, repair_cost_nocache=900.0,
                     repair_steps=[(100.0, 20.0)], cove_cid=f"{iid}|cove", cove_cost=900.0, cove_cost_nocache=3000.0,
                     cove_steps=[(300.0, 50.0)], common=p0 + draft_tok)


def test_fixed_policies_pick_the_right_boundary_and_fallback():
    ctx = _ctx({2: 8, 0: 0}, is_qa=True)
    o = R.run_policy("backtrack_fb", ctx, 0, 4, 0.5)
    assert o.cid.startswith("it|k2|") and o.action == "backtrack"
    o = R.run_policy("restart_fb", ctx, 0, 4, 0.5)
    assert o.cid == R.REFUSE and o.action == "abstain"
    ctx_sum = _ctx({2: 0, 0: 0}, is_qa=False)
    o = R.run_policy("restart_fb", ctx_sum, 0, 4, 0.5)
    assert o.cid == "it|delete" and o.action == "delete"
    o = R.run_policy("care", ctx_sum, 0, 4, 0.5)  # repair paid for, still flagged -> repair, then delete
    assert o.cid == "it|repairdel" and o.action == "repair+delete"
    o = R.run_policy("restart", ctx_sum, 0, 4, 0.5)  # no fallback, nothing better than the draft
    assert o.cid == "it|draft"


def test_care_repairs_first_and_resamples_when_repair_fails():
    ctx = _ctx({0: 8, 1: 8, 2: 8})
    ctx.repair_dmax = 0.1  # the repaired draft passes D: no resampling needed
    o = R.run_policy("care", ctx, 0, 4, 0.5)
    assert o.action == "repair" and o.cid == "it|repair" and o.gen_tokens == 0
    ctx = _ctx({0: 8, 1: 0, 2: 0})
    o = R.run_policy("care", ctx, 0, 4, 0.5)
    assert o.cid.startswith("it|k0|") and o.action == "restart"
    o = R.run_policy("care@norepair", ctx, 0, 4, 0.5)
    assert o.cost < R.run_policy("care", ctx, 0, 4, 0.5).cost


def test_unflagged_items_are_left_alone_and_cove_all_still_runs():
    ctx = _ctx({0: 8})
    ctx.flagged = False
    for pol in ["care", "restart_fb", "repairdel", "delete", "abstain_flag"]:
        assert R.run_policy(pol, ctx, 0, 4, 0.5).cid == "it|draft"
    assert R.run_policy("cove_all", ctx, 0, 0, 0.5).cid == "it|cove"


def test_budget_is_respected():
    rng = np.random.default_rng(0)
    for t in range(200):
        ctx = _ctx({k: int(rng.integers(0, 3)) for k in (0, 1, 2)}, iid=f"b{t}")
        for pol in ["care", "restart_fb", "backtrack_fb", "cascade_fb", "restart"]:
            for b in (1, 2, 4):
                o = R.run_policy(pol, ctx, t, b, 0.5)
                charged = o.cost - (ctx.repair_cost if o.action == "repair" else 0.0)
                assert charged <= b * ctx.budget_unit + 1e-6, (pol, b, charged)


def test_replay_orders_are_deterministic_and_differ_by_round():
    ctx = _ctx({0: 4})
    a = [R.run_policy("restart_fb", ctx, r, 8, 0.5).cid for r in range(20)]
    assert a == [R.run_policy("restart_fb", ctx, r, 8, 0.5).cid for r in range(20)]
    assert len(set(a)) > 1

# ------------------------------------------------------------------ planted-effect simulations


def _simulate(effect: bool, b: float, n_items: int = 300, seed: int = 0) -> dict:
    """Success = the policy found a D-clean answer (detector taken as truth here). effect=True:
    half the items are committed at the backtrack boundary (no clean continuation there) while
    restarts succeed half the time; the other half recover cheaply by backtracking. effect=False:
    every boundary has the same clean rate."""
    rng = np.random.default_rng(seed)
    pols = ["care@norepair", "restart_fb", "backtrack_fb", "cascade_fb"]
    res = {p: [] for p in pols}
    for i in range(n_items):
        if effect:
            clean = {0: 4, 1: 4, 2: 0} if i % 2 == 0 else {0: 4, 1: 4, 2: 7}
        else:
            c = int(rng.integers(1, 6))
            clean = {0: c, 1: c, 2: c}
        ctx = _ctx(clean, is_qa=True, iid=f"s{i}", rng=rng)
        for p in pols:
            res[p].append(np.mean([R.run_policy(p, ctx, r, b, 0.5).cid != R.REFUSE for r in range(8)]))
    return {p: float(np.mean(v)) for p, v in res.items()}


def test_planted_commitment_is_exploited():
    """When the backtrack boundary is committed for some items, online allocation recovers most of
    what backtracking alone loses on them."""
    r = _simulate(effect=True, b=2)
    assert r["care@norepair"] > r["backtrack_fb"] + 0.10
    assert r["care@norepair"] > r["restart_fb"] - 0.05


def test_no_planted_effect_no_manufactured_gain():
    """With no difference between boundaries the allocation cannot beat both fixed arms."""
    r = _simulate(effect=False, b=2)
    assert r["care@norepair"] <= max(r["restart_fb"], r["backtrack_fb"]) + 0.02


# ------------------------------------------------------------------ statistics


def test_holm_and_bootstrap():
    assert S.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    m, lo, hi = S.boot_ci(np.r_[np.zeros(100), np.ones(100)], np.arange(200), n_boot=500)
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi


def test_signflip_is_calibrated_under_the_null():
    rng = np.random.default_rng(1)
    ps = [S.signflip_p(rng.normal(0, 1, 60), np.arange(60), n_perm=400, seed=s) for s in range(150)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10


def test_cluster_bootstrap_widens_with_clustering():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(30), 10)
    vals = np.repeat(rng.normal(0, 1, 30), 10)
    _, lo_c, hi_c = S.boot_ci(vals, clusters, n_boot=800)
    _, lo_i, hi_i = S.boot_ci(vals, np.arange(300), n_boot=800)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_binary_agreement():
    a = S.binary_agreement([1, 1, 0, 0], [1, 0, 0, 0])
    assert a["precision"] == 0.5 and a["recall"] == 1.0 and a["accuracy"] == 0.75
