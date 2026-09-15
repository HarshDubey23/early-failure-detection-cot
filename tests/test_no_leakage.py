"""The critical test: every split in this repository must be question-disjoint.

Three layers of protection are tested:
1. `grouped_split` / `assert_no_leakage` themselves.
2. Every function that makes or consumes a split (E3 curves, E5 risk series,
   E7 transfer, E4 probes) - run on a fixture where each question has several
   correlated traces, then checked from the outside.
3. A source scan: no module may call a row-wise splitter or build its own
   GroupShuffleSplit outside `src/utils/seeds.py`.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.lead_time import stepwise_risk_series
from src.analysis.predictability import make_split, run_surface_curve
from src.analysis.transfer import leave_one_tier_out, transfer_table
from src.data.loaders import traces_to_frame
from src.data.synthetic import synthetic_traces
from src.models.probe import layer_depth_auroc
from src.utils.seeds import assert_no_leakage, grouped_split


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    # 40 questions x 3 correlated traces each, two tiers.
    return traces_to_frame(synthetic_traces(seed=0, n_questions=40, steps_range=[3, 8]))


def test_grouped_split_is_disjoint():
    groups = [f"q{i // 4}" for i in range(100)]  # 4 correlated rows per question
    tr, te = grouped_split(groups, test_size=0.25, seed=42, context="test")
    assert_no_leakage([groups[i] for i in tr], [groups[i] for i in te], "test")
    assert len(tr) + len(te) == len(groups)


def test_assert_no_leakage_catches_overlap():
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(["q1", "q2"], ["q2", "q3"], "test")


def test_repeated_seeds_stay_disjoint():
    groups = [f"q{i // 3}" for i in range(60)]
    for seed in (0, 1, 42):
        tr, te = grouped_split(groups, test_size=0.3, seed=seed, context="test")
        assert_no_leakage([groups[i] for i in tr], [groups[i] for i in te], "test")


def test_shared_e3_split_is_question_disjoint(corpus):
    train_idx, test_idx = make_split(corpus, seed=42, test_size=0.25, context="test")
    train_q = set(corpus["question_id"].iloc[train_idx])
    test_q = set(corpus["question_id"].iloc[test_idx])
    assert train_q.isdisjoint(test_q)
    curve, scores = run_surface_curve(corpus, "answer_incorrect", [0.5, 1.0], (train_idx, test_idx),
                                      seed=42, n_bootstrap=10)
    assert len(scores[1.0]) == len(test_idx)
    assert (curve["n_test"] == len(test_idx)).all()


def test_stepwise_risks_exist_only_for_test_questions(corpus):
    split = make_split(corpus, seed=42, test_size=0.25, context="test")
    series = stepwise_risk_series(corpus, "answer_incorrect", split, "surface", seed=42,
                                  min_train_traces=5)
    scored = corpus[corpus["trace_id"].isin(series["risks"].keys())]
    train_q = set(corpus["question_id"].iloc[split[0]])
    assert set(scored["question_id"]).isdisjoint(train_q)
    for trace_id, risks in series["risks"].items():
        n_steps = int(corpus.loc[corpus["trace_id"] == trace_id, "n_steps"].iloc[0])
        assert len(risks) == n_steps  # one risk per step, r_k uses steps 1..k only


def test_transfer_domains_share_no_questions(corpus):
    other = traces_to_frame(synthetic_traces(seed=1, n_questions=20, steps_range=[3, 8],
                                             dataset="other_domain"))
    assert set(corpus["question_id"]).isdisjoint(set(other["question_id"]))
    split = make_split(corpus, seed=42, test_size=0.25, context="test")
    table = transfer_table(corpus, other, "answer_incorrect", split, ["surface"], 42, 10, "src",
                           min_df=1)
    assert not table.empty
    loto = leave_one_tier_out(corpus, "answer_incorrect", "surface", 42, 10)
    assert set(loto["held_out_tier"]) == {"tierA", "tierB"}


def test_probe_split_keeps_all_states_of_a_question_together():
    rng = np.random.default_rng(0)
    rows = []
    for question in range(30):
        for sample in range(2):
            n_steps = 4
            correct = bool(rng.random() < 0.5)
            for step in range(1, n_steps + 1):
                rows.append({"question_id": f"q{question}", "sample": sample, "step": step,
                             "rel_pos": step / n_steps, "final_correct": correct})
    meta = pd.DataFrame(rows)
    states = rng.normal(size=(len(meta), 8))
    table = layer_depth_auroc(meta, {1: states}, [[0.0, 0.5], [0.5, 1.01]], seed=0, test_size=0.3,
                              n_bootstrap=10, C=1.0, max_iter=500, min_test_states=1)
    assert len(table) == 2  # grouped_split inside asserts disjointness; reaching here means it held


def test_no_row_wise_or_unguarded_splits_in_source():
    offenders = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\btrain_test_split\b|\bKFold\b|\bShuffleSplit\(", text):
            offenders.append(f"{path.name}: row-wise splitter")
        if "GroupShuffleSplit" in text and path.name != "seeds.py":
            offenders.append(f"{path.name}: splitter outside src/utils/seeds.py")
    assert offenders == [], offenders
