"""Stage 02: preprocess (P1-P6). Thin CLI wrapper - no logic here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import loaders
from src.data.loaders import Trace
from src.data.preprocessing import apply_all
from src.utils.io import Manifest, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def synthetic_traces(cfg) -> list:
    """Seeded synthetic corpus for `make smoke` (labelled SYNTHETIC)."""
    rng = np.random.default_rng(cfg.seed)
    traces = []
    for q in range(cfg.data.synthetic_n_questions):
        n_traces = 3
        for s in range(n_traces):
            n_steps = int(rng.integers(cfg.data.synthetic_steps_range[0],
                                       cfg.data.synthetic_steps_range[1] + 1))
            # Failing traces are longer on average (known construction for tests).
            fails = bool(rng.random() < 0.5)
            if fails:
                n_steps += 2
                first_error = int(rng.integers(1, n_steps + 1))
            else:
                first_error = None
            steps = [
                f"synthetic step {i + 1} with "
                + ("error marker xyz " if fails and i + 1 == first_error else "")
                + "reasoning text filler"
                for i in range(n_steps)
            ]
            traces.append(Trace(
                trace_id=f"synthetic:q{q}:{s}", question_id=f"synthetic-q{q}",
                dataset="synthetic", subset="smoke", question=f"smoke question {q}",
                steps=steps, n_steps=n_steps, first_error_idx=first_error,
                final_correct=(not fails), generator="synthetic",
            ))
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="02: preprocess P1-P6")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("02_preprocess", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    if cfg.data.synthetic:
        traces = synthetic_traces(cfg)
        log.info("synthetic=true: generated seeded smoke corpus.")
    else:
        raw = Path(cfg.data.raw_dir) / "processbench.jsonl"
        if not raw.exists():
            print_summary_block("02_preprocess",
                                [f"FAILED: {raw} missing. Run make data first."])
            manifest.log("02_preprocess", args.config, "FAILED", {"missing": str(raw)})
            sys.exit(1)
        traces = []
        for i, line in enumerate(raw.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                record = json.loads(line)
                traces.append(Trace(**record))

    cleaned = apply_all(traces, cfg.preprocessing.min_steps,
                        cfg.preprocessing.error_free_code)
    frame = loaders.traces_to_frame(cleaned)
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "synthetic_" if cfg.data.synthetic else ""
    out = out_dir / f"{prefix}processed_traces.csv"
    frame.to_csv(out, index=False)
    atomic_write_text(
        out_dir / f"{prefix}preprocessing_report.json",
        json.dumps({
            "n_raw": len(traces), "n_clean": len(cleaned),
            "class_balance": float(frame["has_error"].mean()) if len(frame) else None,
            "missing_after_clean": {
                c: int(frame[c].isna().sum()) for c in frame.columns
            },
        }, indent=2),
    )
    summary = [
        f"Raw traces      : {len(traces)}",
        f"After P1-P5     : {len(cleaned)}",
        f"Erroneous       : {100 * frame['has_error'].mean():.1f}%" if len(frame) else "n/a",
        f"Wrote           : {out}",
        "Note: err_idx/relative_onset NaN is structural (error-free traces).",
    ]
    log.info("preprocessed %d -> %d", len(traces), len(cleaned))
    manifest.log("02_preprocess", args.config, "OK",
                 {"n_raw": len(traces), "n_clean": len(cleaned),
                  "synthetic": cfg.data.synthetic})
    print_summary_block("02_preprocess", summary)


if __name__ == "__main__":
    main()
