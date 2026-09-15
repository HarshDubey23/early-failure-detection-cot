"""
WHAT: Atomic file writes and manifest tracking for every pipeline stage.

WHY:  A crashed run must never leave a half-written CSV that a later stage
      mistakes for a complete result (ground rule 1: every number must be
      computed at runtime from real data). Atomic writes plus a manifest row
      per stage make "what ran, with which config" auditable.

HOW:  `atomic_write_*` helpers write to a temp file then rename. `Manifest`
      appends one JSON row per stage with config hash, git status, timing, and
      status (OK / SKIPPED_* / FAILED).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def config_hash(config_path: str | Path) -> str:
    """Hash the raw config file bytes so runs are traceable to exact settings."""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write text atomically (temp file + rename) so crashes leave no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_dataframe(df, path: str | Path, **kwargs) -> None:
    """Write a pandas DataFrame to CSV atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_traces_frame(df, path: str | Path) -> None:
    """Write a processed-trace DataFrame as JSON lines (atomic).

    JSONL keeps each trace's `steps` as a real list and booleans as booleans,
    so nothing has to be re-parsed from a CSV string later.
    """
    atomic_write_text(path, df.to_json(orient="records", lines=True, force_ascii=False))


def read_traces_frame(path: str | Path):
    """Read a processed-trace JSONL written by `write_traces_frame`.

    Raises:
        FileNotFoundError: naming the stage that produces the file.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the preprocessing stage (make preprocess).")
    df = pd.read_json(path, orient="records", lines=True, dtype=False)
    for column in ("has_error", "final_correct", "answer_incorrect"):
        if column in df.columns and df[column].notna().all():
            df[column] = df[column].astype(bool)
    return df


def git_commit() -> str | None:
    """Current git commit hash (None outside a repository), for the manifest."""
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=10)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class Manifest:
    """Append-only JSON-lines manifest of pipeline stages (one JSON object per line)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        config_path: str | Path | None = None,
        status: str = "OK",
        details: dict | None = None,
    ) -> None:
        """Append one manifest row.

        Args:
            stage: Stage name, e.g. "03_run_eda".
            config_path: Config file used (hashed, not copied).
            status: OK, SKIPPED_GPU_ABSENT, NOT_RUN, or FAILED.
            details: Free-form extra fields (key numbers, file lists).
        """
        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "status": status,
            "config_hash": config_hash(config_path) if config_path else None,
            "config_path": str(config_path) if config_path else None,
            "git_commit": git_commit(),
            "details": details or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def read_all(self) -> list:
        """Return all manifest rows (empty list when nothing ran yet)."""
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
