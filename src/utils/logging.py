"""
WHAT: Structured per-run logging (one log file per script invocation).

WHY:  The researcher must be able to reconstruct what ran without re-running
      it. Print statements alone are lost; a timestamped log line per event in
      `results/logs/` plus the terminal summary block keeps evidence.

HOW:  `get_logger` returns a stdlib logger writing to both stderr and a file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(stage: str, logs_dir: str | Path) -> logging.Logger:
    """Create a logger that writes to `results/logs/<stage>.log` and stderr."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"choke_points.{stage}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(logs_dir / f"{stage}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def print_summary_block(title: str, lines: list[str]) -> None:
    """Print the end-of-script human-readable summary block to stdout."""
    print("=" * 72)
    print(title)
    print("=" * 72)
    for line in lines:
        print(line)
    print("=" * 72)
