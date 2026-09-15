"""Loader mapping tests: indexing, grouping and labels on fixtures with known answers.

These pin down the three real-data bugs found when the pipeline first met the
live datasets: 0-indexed labels, repeated problems, and the answer-vs-process
label distinction.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import (
    DatasetSchemaError,
    bigbench_mistake_row_to_trace,
    extract_gsm8k_gold,
    extract_last_number,
    numbers_match,
    processbench_row_to_trace,
    question_id_from_text,
    traces_to_frame,
    zero_indexed_to_first_error,
)
from src.data.preprocessing import apply_p4_relative_onset


def pb_row(label, final_answer_correct, problem="Find x.", n_steps=4, row_id="math-7"):
    return {"id": row_id, "generator": "g", "problem": problem,
            "steps": [f"step {i}" for i in range(n_steps)],
            "final_answer_correct": final_answer_correct, "label": label}


def test_zero_indexed_label_becomes_one_indexed_e():
    assert zero_indexed_to_first_error(0) == 1
    assert zero_indexed_to_first_error(3) == 4
    assert zero_indexed_to_first_error(-1) is None
    assert zero_indexed_to_first_error(None) is None
    with pytest.raises(DatasetSchemaError):
        zero_indexed_to_first_error(-5)


def test_processbench_first_step_error_gives_rho_one_over_n():
    trace = processbench_row_to_trace(pb_row(label=0, final_answer_correct=False), "math", 0)
    frame = traces_to_frame([trace])
    assert trace.first_error_idx == 1
    assert frame["relative_onset"].iloc[0] == pytest.approx(0.25)  # never 0
    assert frame["steps_remaining"].iloc[0] == 3


def test_processbench_last_step_error_gives_rho_one():
    trace = processbench_row_to_trace(pb_row(label=3, final_answer_correct=False), "math", 0)
    assert traces_to_frame([trace])["relative_onset"].iloc[0] == pytest.approx(1.0)


def test_process_error_with_correct_answer_is_kept_distinct():
    frame = traces_to_frame([processbench_row_to_trace(pb_row(label=1, final_answer_correct=True),
                                                       "gsm8k", 0)])
    assert bool(frame["has_error"].iloc[0]) is True
    assert bool(frame["answer_incorrect"].iloc[0]) is False


def test_repeated_problem_shares_question_id_across_rows():
    a = processbench_row_to_trace(pb_row(-1, True, problem="Solve  2x = 4.", row_id="olympiadbench-1"),
                                  "olympiadbench", 0)
    b = processbench_row_to_trace(pb_row(2, False, problem="Solve 2x = 4. ", row_id="olympiadbench-2"),
                                  "olympiadbench", 1)
    assert a.trace_id != b.trace_id
    assert a.question_id == b.question_id


def test_question_ids_are_namespaced_by_dataset():
    assert question_id_from_text("a", "same text") != question_id_from_text("b", "same text")


def test_p4_rejects_unconverted_zero_index():
    with pytest.raises(ValueError, match="outside"):
        apply_p4_relative_onset(0, 5)


def test_bigbench_mistake_mapping():
    record = {"input": "sort: b a", "steps": ["look", "a b"], "answer": " a b",
              "target": "a b", "mistake_index": None}
    trace = bigbench_mistake_row_to_trace(record, "word_sorting", 0)
    assert trace.first_error_idx is None and trace.final_correct is True
    record_bad = {**record, "answer": "b a", "mistake_index": 1}
    trace_bad = bigbench_mistake_row_to_trace(record_bad, "word_sorting", 1)
    assert trace_bad.first_error_idx == 2 and trace_bad.final_correct is False
    with pytest.raises(DatasetSchemaError, match="Observed keys"):
        bigbench_mistake_row_to_trace({"input": "x"}, "word_sorting", 2)


def test_gsm8k_grading_helpers():
    assert extract_gsm8k_gold("so 3 + 4 = 7\n#### 1,007") == "1007"
    assert extract_last_number("so v = 60 / 2 = 30 km/h") == "30"
    assert numbers_match("30.0", "30")
    assert not numbers_match(None, "30")
