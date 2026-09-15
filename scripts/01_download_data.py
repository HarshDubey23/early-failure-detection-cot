"""Stage 01: download (and cache) datasets. Thin CLI wrapper - no logic here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import loaders
from src.utils.io import Manifest, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="01: download datasets")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("01_download_data", cfg.outputs.logs_dir)
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(cfg.outputs.manifest)
    summary, status, details = [], "OK", {}

    if cfg.data.synthetic:
        # Smoke mode: no network. Real downloads are skipped by design.
        log.info("synthetic=true: skipping network downloads (smoke test).")
        summary.append("Synthetic mode: no datasets downloaded (smoke test).")
        details["synthetic"] = True
    else:
        # ProcessBench: verify identifier resolves, cache one JSONL of traces.
        try:
            traces = loaders.load_processbench()
            out = raw_dir / "processbench.jsonl"
            lines = "\n".join(
                json.dumps(r) for r in loaders.to_records(traces)
            )
            atomic_write_text(out, lines)
            msg = f"ProcessBench: {len(traces)} traces cached -> {out}"
            log.info(msg)
            summary.append(msg)
            details["processbench_n"] = len(traces)
        except Exception as exc:  # noqa: BLE001 - must record, then continue
            msg = f"ProcessBench FAILED: {exc}"
            log.error(msg)
            summary.append(msg)
            status = "FAILED"
            details["processbench_error"] = str(exc)[:500]
        # GSM8K questions (for trajectory generation): small, cached too.
        try:
            questions = loaders.load_gsm8k(limit=cfg.probes.n_problems)
            out = raw_dir / "gsm8k_questions.jsonl"
            atomic_write_text(out, "\n".join(
                json.dumps(loaders.to_records([q])[0]) for q in questions))
            msg = f"GSM8K: {len(questions)} questions cached -> {out}"
            log.info(msg)
            summary.append(msg)
            details["gsm8k_n"] = len(questions)
        except Exception as exc:  # noqa: BLE001
            msg = f"GSM8K FAILED: {exc}"
            log.error(msg)
            summary.append(msg)
            status = "FAILED" if status == "OK" else status
            details["gsm8k_error"] = str(exc)[:500]
        for name in ("PRM800K", "MATH", "BIG-Bench Mistake", "StrategyQA"):
            summary.append(f"{name}: loader implemented; see docs/BENCHMARKS.md "
                           f"for identifier verification status.")
    manifest.log("01_download_data", args.config, status, details)
    print_summary_block("01_download_data", summary)
    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
