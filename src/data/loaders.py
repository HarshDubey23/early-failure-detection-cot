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
    first_error_idx: int | None  # 1-INDEXED e in {1..n}; None when no error / not annotated
    final_correct: bool | None
    generator: str | None  # which model produced the trace
    gold_answer: str | None = None  # only for generation corpora (GSM8K) - used to grade


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


def question_id_from_text(dataset: str, question_text: str) -> str:
    """Stable question identifier derived from the question text itself.

    Why not the dataset's own row id: ProcessBench has 3,400 row ids but only
    2,842 distinct problems (several generators solved the same problem).
    Grouping by row id would let the same problem sit in train AND test, which
    is exactly the leakage rule P6 forbids. Hashing whitespace-normalised
    question text makes every copy of a problem share one group.

    Example:
        >>> question_id_from_text("pb", "What is  2+2?") == question_id_from_text("pb", "What is 2+2?")
        True
    """
    import hashlib

    normalised = " ".join(str(question_text).split())
    digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{digest}"


def zero_indexed_to_first_error(label, error_free_code: int = -1) -> int | None:
    """Convert a 0-indexed first-error label to the paper's 1-indexed `e`.

    ProcessBench and BIG-Bench Mistake both store the first erroneous step as
    a 0-INDEXED position (0 = the first step is wrong), with -1 / null for
    error-free traces. The paper defines e in {1, ..., n} so that
    rho = e / n lies in (0, 1]. Using the raw label as e would shift every
    onset one step earlier and allow rho = 0.

    Example:
        >>> zero_indexed_to_first_error(0)
        1
        >>> zero_indexed_to_first_error(-1) is None
        True
    """
    if label is None:
        return None
    label = int(label)
    if label == error_free_code:
        return None
    if label < 0:
        raise DatasetSchemaError(f"Unexpected negative first-error label {label}.")
    return label + 1


# ---------------------------------------------------------------------------
# Primary loader: ProcessBench
# ---------------------------------------------------------------------------

PROCESSBENCH_REPO = "Qwen/ProcessBench"
PROCESSBENCH_REQUIRED = ["id", "generator", "problem", "steps", "final_answer_correct", "label"]


def processbench_row_to_trace(row: dict, split_name: str, row_index: int) -> Trace:
    """Map one ProcessBench row to the uniform `Trace` schema.

    Args:
        row: Native row with id, generator, problem, steps, final_answer_correct, label.
        split_name: The difficulty tier (gsm8k / math / olympiadbench / omnimath).
        row_index: Position in the split (for a unique trace id).

    Returns:
        Trace with a 1-indexed `first_error_idx` and `final_correct` taken from
        the dataset's own `final_answer_correct` field (NOT inferred from the
        process label: 521 traces contain a process error yet reach the
        correct final answer).

    Example:
        >>> t = processbench_row_to_trace({"id": "gsm8k-0", "generator": "g",
        ...     "problem": "p", "steps": ["a", "b"], "final_answer_correct": False,
        ...     "label": 1}, "gsm8k", 0)
        >>> (t.first_error_idx, t.final_correct, t.subset)
        (2, False, 'gsm8k')
    """
    steps = _as_step_texts(row["steps"])
    return Trace(
        trace_id=f"processbench:{split_name}:{row.get('id', row_index)}",
        question_id=question_id_from_text("processbench", row["problem"]),
        dataset="processbench",
        subset=split_name,
        question=str(row["problem"]),
        steps=steps,
        n_steps=len(steps),
        first_error_idx=zero_indexed_to_first_error(row["label"]),
        final_correct=bool(row["final_answer_correct"]),
        generator=row.get("generator"),
    )


def load_processbench() -> list:
    """Load every split of every config of ProcessBench (`Qwen/ProcessBench`).

    The live layout (verified 2026-09-15) is ONE config, `default`, with FOUR
    splits named after the difficulty tiers: gsm8k (400), math (1000),
    olympiadbench (1000), omnimath (1000). Each split becomes a `subset`.

    Returns:
        List of `Trace`.

    Raises:
        DatasetSchemaError: When a split lacks a required column (the error
            lists the columns actually observed).
    """
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(PROCESSBENCH_REPO)
    print(f"[ProcessBench] configs: {configs}")
    traces: list[Trace] = []
    for cfg in configs:
        dataset_dict = load_dataset(PROCESSBENCH_REPO, cfg)
        print(f"[ProcessBench:{cfg}] splits: {list(dataset_dict.keys())}")
        for split_name, split in dataset_dict.items():
            cols = split.column_names
            print(f"[ProcessBench:{cfg}/{split_name}] rows={len(split)} columns={cols}")
            missing = [c for c in PROCESSBENCH_REQUIRED if c not in cols]
            if missing:
                raise DatasetSchemaError(
                    f"ProcessBench {cfg}/{split_name}: missing columns {missing}. "
                    f"Observed columns: {cols}"
                )
            for i, row in enumerate(split):
                traces.append(processbench_row_to_trace(row, split_name, i))
    return traces


# ---------------------------------------------------------------------------
# Secondary loader: PRM800K (GitHub checkout, local files)
# ---------------------------------------------------------------------------

def load_prm800k(data_dir: str | Path, phase: str = "phase1_train") -> list:
    """Load PRM800K from a local GitHub checkout (`openai/prm800k`).

    PRM800K rates each step +1 (good), 0 (neutral) or -1 (bad). The first
    step rated -1 becomes `first_error_idx` (1-indexed).

    UNVERIFIED MAPPING: PRM800K was not downloaded in this build, so the
    flattened `steps`/`completions` layout assumed here has not been checked
    against the real files (which nest completions under `label.steps`).
    The function raises `DatasetSchemaError` rather than guessing if it differs.

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
            # PRM800K rates each step -1 (bad), 0 (neutral) or +1 (good); only -1 is an error.
            if first_error is None and rating == -1:
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
    to generate trajectories from an open model for activation extraction; the
    gold answer (the number after `####`) is kept in `gold_answer` so every
    generated trajectory can be graded.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    cols = ds[split].column_names
    print(f"[GSM8K] splits={list(ds.keys())} columns={cols}")
    if "question" not in cols or "answer" not in cols:
        raise DatasetSchemaError(f"GSM8K: expected question/answer columns. Observed: {cols}")
    rows = ds[split]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [
        Trace(
            trace_id=f"gsm8k:{split}:{i}",
            question_id=question_id_from_text("gsm8k", row["question"]),
            dataset="gsm8k",
            subset=split,
            question=str(row["question"]),
            steps=[],
            n_steps=0,
            first_error_idx=None,
            final_correct=None,
            generator=None,
            gold_answer=extract_gsm8k_gold(row["answer"]),
        )
        for i, row in enumerate(rows)
    ]


def extract_gsm8k_gold(answer_text: str) -> str | None:
    """Return the gold number after `####` in a GSM8K answer string.

    Example:
        >>> extract_gsm8k_gold("3 + 4 = 7 #### 1,007")
        '1007'
    """
    import re

    match = re.search(r"####\s*(-?[\d,\.]+)", answer_text or "")
    return match.group(1).replace(",", "").strip() if match else None


def extract_last_number(generated_text: str) -> str | None:
    """Return the last number in a generated solution (the prototype's grader).

    Example:
        >>> extract_last_number("so v = 60 / 2 = 30 km/h")
        '30'
    """
    import re

    numbers = re.findall(r"-?\d+(?:\.\d+)?", (generated_text or "").replace(",", ""))
    return numbers[-1] if numbers else None


def numbers_match(predicted: str | None, gold: str | None) -> bool:
    """Numeric equality with float tolerance, so '30.0' matches '30'."""
    if predicted is None or gold is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()


def load_math(split: str = "test", limit: int | None = None) -> list:
    """Load MATH / MATH-500 competition problems (harder trajectory generation).

    Tries `HuggingFaceH4/MATH-500` first, then `hendrycks/competition_math`.
    Records which identifier resolved in the trace `subset` field. Note: on
    2026-09-15 `hendrycks/competition_math` is marked `disabled` on the Hub,
    so only MATH-500 resolves (see docs/BENCHMARKS.md). Not used by any
    experiment in the current build.
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

BIGBENCH_MISTAKE_BASE_URL = "https://raw.githubusercontent.com/WHGTyen/BIG-Bench-Mistake/main"
BIGBENCH_MISTAKE_TASKS = [
    "dyck_languages",
    "logical_deduction",
    "multistep_arithmetic",
    "tracking_shuffled_objects",
    "word_sorting",
]


def download_bigbench_mistake(data_dir: str | Path) -> list:
    """Download the five BIG-Bench Mistake JSONL files from the authors' repo.

    Source: github.com/WHGTyen/BIG-Bench-Mistake (Apache-2.0), released with
    Tyen et al. (2024). Network access happens only when this is called.

    Returns:
        Paths of the files written.
    """
    import urllib.request

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task in BIGBENCH_MISTAKE_TASKS:
        target = data_dir / f"{task}.jsonl"
        url = f"{BIGBENCH_MISTAKE_BASE_URL}/{task}.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            target.write_bytes(response.read())
        written.append(target)
    return written


def _normalise_answer(text) -> str:
    """Collapse whitespace so '(A) ' and '(A)' compare equal."""
    return " ".join(str(text).split()).strip()


def bigbench_mistake_row_to_trace(record: dict, task: str, row_index: int) -> Trace:
    """Map one BIG-Bench Mistake record to `Trace`.

    Native fields: input, steps, answer, target, mistake_index. The dataset
    README states `mistake_index` is 0-INDEXED (null = no mistake), so it is
    converted to the paper's 1-indexed e. `final_correct` is exact match of
    the whitespace-normalised extracted answer against the target - the
    matching rule the dataset authors describe.

    Example:
        >>> t = bigbench_mistake_row_to_trace({"input": "q", "steps": ["a", "b"],
        ...     "answer": "(A)", "target": "(B)", "mistake_index": 0}, "word_sorting", 0)
        >>> (t.first_error_idx, t.final_correct)
        (1, False)
    """
    for key in ("input", "steps", "answer", "target", "mistake_index"):
        if key not in record:
            raise DatasetSchemaError(
                f"BIG-Bench Mistake {task} row {row_index}: missing '{key}'. "
                f"Observed keys: {sorted(record.keys())}"
            )
    steps = _as_step_texts(record["steps"])
    return Trace(
        trace_id=f"bbm:{task}:{row_index}",
        question_id=question_id_from_text(f"bbm-{task}", record["input"]),
        dataset="bigbench_mistake",
        subset=task,
        question=str(record["input"]),
        steps=steps,
        n_steps=len(steps),
        first_error_idx=zero_indexed_to_first_error(record["mistake_index"]),
        final_correct=_normalise_answer(record["answer"]) == _normalise_answer(record["target"]),
        generator="PaLM-2-L (Unicorn)",
    )


def load_bigbench_mistake(data_dir: str | Path) -> list:
    """Load BIG-Bench Mistake traces from local JSONL files (one per task).

    Raises:
        FileNotFoundError: when the JSONL files are absent (run `make data`).
    """
    import json

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.jsonl")) if data_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"BIG-Bench Mistake data not found under {data_dir}. Run `make data` "
            f"(downloads from {BIGBENCH_MISTAKE_BASE_URL}). Nothing is "
            f"substituted in their place."
        )
    traces = []
    for path in files:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        print(f"[BIG-Bench Mistake:{path.stem}] rows={len(records)} "
              f"keys={sorted(records[0].keys()) if records else []}")
        for i, record in enumerate(records):
            traces.append(bigbench_mistake_row_to_trace(record, path.stem, i))
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

    Columns: trace_id, question_id, dataset, subset, question, steps, n_steps,
    first_error_idx (1-indexed e), has_error (a process error exists),
    relative_onset (rho = e / n, via P4), steps_remaining (n - e),
    final_correct, answer_incorrect (the paper's y), generator.
    """
    import pandas as pd

    from .preprocessing import apply_p4_relative_onset

    records = []
    for trace in traces:
        has_error = trace.first_error_idx is not None
        relative_onset = apply_p4_relative_onset(trace.first_error_idx, trace.n_steps)
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
                "relative_onset": (
                    relative_onset if relative_onset is not None else float("nan")
                ),
                "steps_remaining": (
                    trace.n_steps - trace.first_error_idx if has_error else float("nan")
                ),
                "final_correct": trace.final_correct,
                # The paper's y: 1 when the final answer disagrees with gold.
                "answer_incorrect": (
                    (not trace.final_correct) if trace.final_correct is not None else None
                ),
                "generator": trace.generator,
            }
        )
    return pd.DataFrame(records)


def to_records(traces: list) -> list:
    """Serialise traces to plain dicts (for caching to JSONL)."""
    return [asdict(t) for t in traces]
