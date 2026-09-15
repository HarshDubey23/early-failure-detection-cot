"""Stage 10: build RESULTS.md + paper_values.json from saved tables. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.reporting.report import build_paper_values, render_results_markdown
from src.utils.io import Manifest, atomic_write_text
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="10: build report")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("10_build_report", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    stage_status = {}
    for row in manifest.read_all():
        stage_status[row["stage"]] = row["status"]  # last run of each stage wins
    stage_status["10_build_report"] = "OK"
    values = build_paper_values(cfg)
    atomic_write_text(Path(cfg.outputs.results_dir) / "paper_values.json", json.dumps(values, indent=2))
    markdown = render_results_markdown(cfg, values, stage_status)
    atomic_write_text(cfg.outputs.report, markdown)
    manifest.log("10_build_report", args.config, "OK", {"report": cfg.outputs.report})
    summary = [f"Wrote {cfg.outputs.report} and {Path(cfg.outputs.results_dir) / 'paper_values.json'}"]
    summary += [f"{key}: {entry['value']}" for key, entry in values["placeholders"].items()]
    summary += [f"Stage {stage}: {status}" for stage, status in sorted(stage_status.items())]
    for line in summary:
        log.info(line)
    print_summary_block("10_build_report", summary)


if __name__ == "__main__":
    main()
