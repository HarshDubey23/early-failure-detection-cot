"""
WHAT: Dataset loaders emitting one uniform schema for all six benchmarks.

WHY:  Downstream code (onset analysis, probes, transfer) must be
      dataset-agnostic. A single `Trace` schema also makes the grouped-split
      assertion enforceable in one place: every trace carries `question_id`.

HOW:  Each `load_*` function prints and logs the live schema first, maps native
      columns to `Trace`, and raises `DatasetSchemaError` naming the dataset
      and observed columns when a column is missing. Network access happens
      only inside these functions, never at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


class DatasetSchemaError(RuntimeError):
    """Raised when a dataset's live schema lacks a required column."""


@dataclass
class Trace:
    trace_id: str
    question_id: str  # for grouped splitting - NEVER omit
    dataset: str
    subset: str  # difficulty tier where applicable
    question: str
    steps: list
    n_steps: int
    first_error_idx: int | None  # None when no error / not annotated
    final_correct: bool | None
    generator: str | None  # which model produced the trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_CANDIDATES = ["steps", "solution_steps", "reasoning_steps"]
_LABEL_CANDIDATES = ["label", "error_index", "first_error_step", "first_error_idx"]
_QUESTION_CANDIDATES = ["question", "problem", "input", "prompt"]


def find_column(columns: list, candidates: list) -> str | None:
    """Return the first candidate present in `columns`, else None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _as_step_texts(raw_steps) -> list:
    """Coerce a raw step list (str or dict entries) to plain strings."""
    texts = []
    for entry in raw_steps or []:
        if isinstance(entry, dict):
            for key in ("text", "content", "step", "value"):
                if key in entry:
                    texts.append(str(entry[key]))
                    break
            else:
                texts.append(str(entry))
        else:
            texts.append(str(entry))
    return texts


# ---------------------------------------------------------------------------
# Primary loader: ProcessBench
# ---------------------------------------------------------------------------

PROCESSBENCH_REPO = "Qwen/ProcessBench"


def load_processbench(split: str | None = None) -> list:
    """Load ProcessBench (`Qwen/ProcessBench`) across all configs.

    Returns:
        List of `Trace` with `first_error_idx` set (`-1` in source means
        error-free and becomes None) and `final_correct` derived from it.
    """
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(PROCESSBENCH_REPO)
    print(f"[ProcessBench] configs: {configs}")
    traces: list[Trace] = []
    for cfg in configs:
        ds = load_dataset(PROCESSBENCH_REPO, cfg)
        split_name = split or next(iter(ds.keys()))
        cols = ds[split_name].column_names
        print(f"[ProcessBench:{cfg}] split={split_name} columns={cols}")
        steps_col = find_column(cols, _STEP_CANDIDATES)
        label_col = find_column(cols, _LABEL_CANDIDATES)
        if steps_col is None or label_col is None:
            raise DatasetSchemaError(
                f"ProcessBench config '{cfg}': could not find step/label columns. "
                f"Observed columns: {cols}"
            )
        question_col = find_column(cols, _QUESTION_CANDIDATES)
        for i, row in enumerate(ds[split_name]):
            steps = _as_step_texts(row[steps_col])
            label = row[label_col]
            has_error = label is not None and int(label) != -1
            traces.append(
                Trace(
                    trace_id=f"processbench:{cfg}:{i}",
                    question_id=str(
                        row.get("id", row.get("question_id", f"{cfg}:{i}"))
                    ),
                    dataset="processbench",
                    subset=cfg,
                    question=str(row.get(question_col, "") if question_col else ""),
                    steps=steps,
                    n_steps=len(steps),
                    first_error_idx=int(label) if has_error else None,
                    final_correct=(not has_error),
                    generator=row.get("generator"),
                )
            )
    return traces


# ---------------------------------------------------------------------------
# Secondary loader: PRM800K (GitHub checkout, local files)
# ---------------------------------------------------------------------------

def load_prm800k(data_dir: str | Path, phase: str = "phase1_train") -> list:
    """Load PRM800K from a local GitHub checkout (`openai/prm800k`).

    PRM800K labels each step as correct/incorrect/neutral. The first step
    labelled incorrect becomes `first_error_idx`.

    Args:
        data_dir: Local checkout directory containing `phase1_train.jsonl`
            / `phase2_train.jsonl` style files.
        phase: File stem to read (without extension).

    Raises:
        DatasetSchemaError: When the expected file or fields are absent.
        FileNotFoundError: When the checkout directory is missing, with a
            message explaining how to obtain it.
    """
    import json
    from pathlib import Path as _Path

    data_dir = _Path(data_dir)
    candidate = data_dir / f"{phase}.jsonl"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"PRM800K checkout not found at {data_dir}. Clone "
            f"https://github.com/openai/prm800k and pass its data directory. "
            f"No data is fabricated in its place."
        )
    if not candidate.exists():
        raise DatasetSchemaError(
            f"PRM800K: expected file {candidate} not found. "
            f"Directory contains: {sorted(p.name for p in data_dir.iterdir())}"
        )
    traces = []
    for i, line in enumerate(candidate.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if "steps" not in record and "completions" not in record:
            raise DatasetSchemaError(
                f"PRM800K row {i}: missing step list. Observed keys: "
                f"{sorted(record.keys())}"
            )
        raw = record.get("steps") or record.get("completions")
        steps, first_error = [], None
        for j, step in enumerate(raw):
            text = step.get("text", "") if isinstance(step, dict) else str(step)
            steps.append(text)
            rating = (step.get("rating", 1) if isinstance(step, dict) else 1)
            if first_error is None and rating == 0:
                first_error = j + 1  # 1-indexed to match Trace convention
        traces.append(
            Trace(
                trace_id=f"prm800k:{phase}:{i}",
                question_id=str(record.get("question_id", record.get("id", i))),
                dataset="prm800k",
                subset=record.get("subset", phase),
                question=str(record.get("question", record.get("problem", ""))),
                steps=steps,
                n_steps=len(steps),
                first_error_idx=first_error,
                final_correct=(first_error is None),
                generator=record.get("generator", "human+model"),
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Trajectory-generation corpora (outcome labels only): GSM8K, MATH
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "test", limit: int | None = None) -> list:
    """Load GSM8K questions (`openai/gsm8k`, config `main`).

    GSM8K carries outcome labels only, so `first_error_idx` is None. It is used
    to generate trajectories from an open model for activation extraction.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    print(f"[GSM8K] splits={list(ds.keys())} columns={ds[split].column_names}")
    rows = ds[split]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [
        Trace(
            trace_id=f"gsm8k:{split}:{i}",
            question_id=f"gsm8k:{split}:{i}",
            dataset="gsm8k",
            subset=split,
            question=str(row["question"]),
            steps=[],
            n_steps=0,
            first_error_idx=None,
            final_correct=None,
            generator=None,
        )
        for i, row in enumerate(rows)
    ]


def load_math(split: str = "test", limit: int | None = None) -> list:
    """Load MATH / MATH-500 competition problems (harder trajectory generation).

    Tries `HuggingFaceH4/MATH-500` first, then `hendrycks/competition_math`.
    Records which identifier resolved in the trace `subset` field.
    """
    from datasets import load_dataset

    last_error = None
    for repo in ("HuggingFaceH4/MATH-500", "hendrycks/competition_math"):
        try:
            ds = load_dataset(repo)
            print(f"[MATH] resolved {repo}: splits={list(ds.keys())}")
            split_name = split if split in ds else next(iter(ds.keys()))
            rows = ds[split_name]
            if limit is not None:
                rows = rows.select(range(min(limit, len(rows))))
            cols = rows.column_names
            qcol = find_column(cols, _QUESTION_CANDIDATES + ["problem"])
            traces = []
            for i, row in enumerate(rows):
                traces.append(
                    Trace(
                        trace_id=f"math:{repo}:{i}",
                        question_id=f"math:{repo}:{i}",
                        dataset="math",
                        subset=repo,
                        question=str(row.get(qcol, "") if qcol else ""),
                        steps=[],
                        n_steps=0,
                        first_error_idx=None,
                        final_correct=None,
                        generator=None,
                    )
                )
            return traces
        except Exception as exc:  # noqa: BLE001 - must record, not substitute
            last_error = exc
            print(f"[MATH] identifier did not resolve: {repo} ({exc})")
    raise DatasetSchemaError(
        "MATH: neither HuggingFaceH4/MATH-500 nor hendrycks/competition_math "
        f"resolved. Last error: {last_error}. Recorded as unverified in "
        "docs/BENCHMARKS.md; no substitute dataset is used."
    )


# ---------------------------------------------------------------------------
# Cross-domain corpora: BIG-Bench Mistake, StrategyQA
# ---------------------------------------------------------------------------

def load_bigbench_mistake(data_dir: str | Path) -> list:
    """Load BIG-Bench mistake-detection traces from a local checkout.

    Expected layout: JSONL files with `steps` and `first_mistake` fields
    (see Google Research `bigbench` repo, mistake-detection tasks).
    """
    import json
    from pathlib import Path as _Path

    data_dir = _Path(data_dir)
    files = sorted(data_dir.glob("*.jsonl"))
    if not data_dir.exists() or not files:
        raise FileNotFoundError(
            f"BIG-Bench Mistake data not found under {data_dir}. Place the "
            f"task JSONL files there (see docs/BENCHMARKS.md). Nothing is "
            f"substituted in their place."
        )
    traces = []
    for path in files:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            if "steps" not in record:
                raise DatasetSchemaError(
                    f"BIG-Bench Mistake {path.name} row {i}: missing 'steps'. "
                    f"Observed keys: {sorted(record.keys())}"
                )
            steps = _as_step_texts(record["steps"])
            first = record.get("first_mistake", record.get("first_error_idx"))
            traces.append(
                Trace(
                    trace_id=f"bbm:{path.stem}:{i}",
                    question_id=str(record.get("question_id", f"{path.stem}:{i}")),
                    dataset="bigbench_mistake",
                    subset=path.stem,
                    question=str(record.get("question", "")),
                    steps=steps,
                    n_steps=len(steps),
                    first_error_idx=int(first) if first is not None else None,
                    final_correct=(first is None),
                    generator=record.get("generator"),
                )
            )
    return traces


def load_strategyqa(split: str = "train", limit: int | None = None) -> list:
    """Load StrategyQA implicit multi-hop questions (decomposition only).

    No first-error annotations exist, so `first_error_idx` is None. Used only
    for cross-domain transfer of the outcome detector (E7).
    """
    from datasets import load_dataset

    last_error = None
    for repo in ("wics/strategy-qa", "bigbio/strategy_qa"):
        try:
            ds = load_dataset(repo)
            split_name = split if split in ds else next(iter(ds.keys()))
            print(f"[StrategyQA] resolved {repo}: columns={ds[split_name].column_names}")
            rows = ds[split_name]
            if limit is not None:
                rows = rows.select(range(min(limit, len(rows))))
            cols = rows.column_names
            qcol = find_column(cols, _QUESTION_CANDIDATES)
            traces = []
            for i, row in enumerate(rows):
                decomposition = row.get("decomposition", row.get("facts", [])) or []
                steps = _as_step_texts(decomposition)
                traces.append(
                    Trace(
                        trace_id=f"strategyqa:{i}",
                        question_id=str(row.get("qid", row.get("id", i))),
                        dataset="strategyqa",
                        subset=split_name,
                        question=str(row.get(qcol, "") if qcol else ""),
                        steps=steps,
                        n_steps=len(steps),
                        first_error_idx=None,
                        final_correct=None,
                        generator=None,
                    )
                )
            return traces
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[StrategyQA] identifier did not resolve: {repo} ({exc})")
    raise DatasetSchemaError(
        "StrategyQA: no known identifier resolved. "
        f"Last error: {last_error}. Recorded as unverified in BENCHMARKS.md."
    )


def traces_to_frame(traces: list) -> object:
    """Convert a trace list to the processed dataframe used by all analysis.

    Columns: trace_id, question_id, dataset, subset, n_steps, first_error_idx,
    has_error, relative_onset, steps_remaining, question, steps, generator.
    """
    import pandas as pd

    records = []
    for trace in traces:
        has_error = trace.first_error_idx is not None
        records.append(
            {
                "trace_id": trace.trace_id,
                "question_id": trace.question_id,
                "dataset": trace.dataset,
                "subset": trace.subset,
                "question": trace.question,
                "steps": trace.steps,
                "n_steps": trace.n_steps,
                "first_error_idx": trace.first_error_idx,
                "has_error": has_error,
                # P4 normalisation lives in preprocessing; mirrored here for
                # convenience on already-processed traces.
                "relative_onset": (
                    trace.first_error_idx / trace.n_steps
                    if has_error and trace.n_steps
                    else float("nan")
                ),
                "steps_remaining": (
                    trace.n_steps - trace.first_error_idx if has_error else float("nan")
                ),
                "final_correct": trace.final_correct,
                "generator": trace.generator,
            }
        )
    return pd.DataFrame(records)


def to_records(traces: list) -> list:
    """Serialise traces to plain dicts (for caching to JSONL)."""
    return [asdict(t) for t in traces]
