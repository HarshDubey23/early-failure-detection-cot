"""Stage 01: download + cache datasets and verify every identifier. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import loaders
from src.data.verification import verify_identifiers
from src.utils.io import Manifest, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="01: download datasets")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("01_download_data", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary, details, failures = [], {}, []

    if cfg.data.synthetic:
        log.info("synthetic=true: skipping network downloads (smoke test).")
        manifest.log("01_download_data", args.config, "OK", {"synthetic": True})
        print_summary_block("01_download_data", ["Synthetic mode: no datasets downloaded."])
        return

    # 1) Primary corpus: ProcessBench, all splits, cached as JSONL of Trace records.
    try:
        traces = loaders.load_processbench()
        out = raw_dir / "processbench.jsonl"
        atomic_write_text(out, "\n".join(json.dumps(r, ensure_ascii=False)
                                         for r in loaders.to_records(traces)))
        summary.append(f"ProcessBench      : {len(traces)} traces cached -> {out}")
        details["processbench_n"] = len(traces)
    except Exception as exc:  # noqa: BLE001 - record, then fail the stage
        failures.append(f"ProcessBench FAILED: {exc}")

    # 2) Cross-domain corpus with first-mistake labels: BIG-Bench Mistake.
    try:
        files = loaders.download_bigbench_mistake(cfg.data.bigbench_mistake_dir)
        n_bbm = len(loaders.load_bigbench_mistake(cfg.data.bigbench_mistake_dir))
        summary.append(f"BIG-Bench Mistake : {n_bbm} traces in {len(files)} task files")
        details["bigbench_mistake_n"] = n_bbm
    except Exception as exc:  # noqa: BLE001
        failures.append(f"BIG-Bench Mistake FAILED: {exc}")

    # 3) GSM8K questions (with gold answers) for GPU trajectory generation (stage 06).
    try:
        questions = loaders.load_gsm8k(limit=cfg.probes.n_problems)
        out = raw_dir / "gsm8k_questions.jsonl"
        atomic_write_text(out, "\n".join(json.dumps(r, ensure_ascii=False)
                                         for r in loaders.to_records(questions)))
        summary.append(f"GSM8K             : {len(questions)} questions cached -> {out}")
        details["gsm8k_n"] = len(questions)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"GSM8K FAILED: {exc}")

    # 4) Identifier verification for every dataset named in the brief (no bulk download).
    verification = verify_identifiers()
    out = Path(cfg.outputs.tables_dir) / "dataset_identifier_verification.json"
    atomic_write_text(out, json.dumps(verification, indent=2))
    for row in verification:
        summary.append(f"  verify {row['identifier']:<32} -> {row['status']}")
    summary += failures
    for line in summary:
        log.info(line)
    status = "FAILED" if failures else "OK"
    manifest.log("01_download_data", args.config, status, details)
    print_summary_block("01_download_data", summary)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
