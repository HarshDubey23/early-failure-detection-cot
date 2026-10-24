"""Stage 12a: write data/processed/processed_traces.jsonl straight from Hugging Face.

A light alternative to stages 01-02 for GPU notebooks (Kaggle, Colab): it needs only
`datasets` and the repository's loader, not the pinned analysis stack. Stage 12 reads only
the fields written here. P5 (drop traces with fewer than two steps) is applied; ProcessBench
has none.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import loaders  # noqa: E402


def main() -> None:
    traces = [t for t in loaders.load_processbench() if t.n_steps >= 2]
    out = Path("data/processed/processed_traces.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for record in loaders.to_records(traces):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(traces)} traces -> {out}")


if __name__ == "__main__":
    main()
