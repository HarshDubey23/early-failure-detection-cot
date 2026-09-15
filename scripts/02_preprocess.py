"""Stage 02: preprocess (P1-P6) -> processed trace files + data-quality report. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import loaders
from src.data.loaders import Trace
from src.data.preprocessing import apply_all, quality_report
from src.data.synthetic import synthetic_traces
from src.utils.io import Manifest, atomic_write_text, write_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="02: preprocess P1-P6")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("02_preprocess", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    if cfg.data.synthetic:
        primary_raw = synthetic_traces(cfg.seed, cfg.data.synthetic_n_questions,
                                       cfg.data.synthetic_steps_range, "synthetic")
        transfer_raw = synthetic_traces(cfg.seed + 1, cfg.data.synthetic_n_questions // 2,
                                        cfg.data.synthetic_steps_range, "synthetic_transfer")
    else:
        raw_path = Path(cfg.data.raw_dir) / "processbench.jsonl"
        if not raw_path.exists():
            manifest.log("02_preprocess", args.config, "FAILED", {"missing": str(raw_path)})
            print_summary_block("02_preprocess", [f"FAILED: {raw_path} missing. Run make data."])
            sys.exit(1)
        primary_raw = [Trace(**json.loads(line)) for line in
                       raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        transfer_raw = loaders.load_bigbench_mistake(cfg.data.bigbench_mistake_dir)

    summary, details = [], {}
    for name, raw in (("processed_traces", primary_raw), ("processed_transfer", transfer_raw)):
        cleaned = apply_all(raw, cfg.preprocessing.min_steps, cfg.preprocessing.error_free_code)
        frame = loaders.traces_to_frame(cleaned)
        write_traces_frame(frame, Path(cfg.data.processed_dir) / f"{name}.jsonl")
        report = quality_report(raw, cleaned, frame)
        atomic_write_text(Path(cfg.outputs.tables_dir) / f"{name}_quality_report.json",
                          json.dumps(report, indent=2))
        details[name] = {"n_raw": report["n_raw"], "n_clean": report["n_clean"]}
        summary += [
            f"[{name}] raw traces            : {report['n_raw']}",
            f"[{name}] after P1-P5           : {report['n_clean']} "
            f"(dropped by P5: {report['dropped_by_p5']})",
            f"[{name}] distinct questions    : {report['n_questions']}",
            f"[{name}] process error (e set) : {100 * report['frac_has_error']:.1f}%",
            f"[{name}] final answer incorrect: {100 * report['frac_answer_incorrect']:.1f}%",
            f"[{name}] unexpected missing    : {report['unexpected_missing']}",
        ]
    summary.append("Note: first_error_idx / relative_onset are empty by design for error-free traces.")
    for line in summary:
        log.info(line)
    manifest.log("02_preprocess", args.config, "OK", {**details, "synthetic": cfg.data.synthetic})
    print_summary_block("02_preprocess", summary)


if __name__ == "__main__":
    main()
