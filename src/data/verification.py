"""
WHAT: Checks that every dataset identifier named in the brief actually resolves.

WHY:  Brief §3/§13: "Verify each dataset identifier resolves before building on
      it. If it does not exist, report that fact rather than substituting a
      guess." A Hub repo can exist but be DISABLED (e.g. taken down), which a
      naive existence check misses, so the Hub's `disabled` flag is read too.

HOW:  One metadata request per identifier (HuggingFace Hub API or GitHub API).
      Nothing is downloaded here. Network access happens only when
      `verify_identifiers` is called - never at import time.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

IDENTIFIERS = [
    # (dataset, host, identifier, role in this repository)
    ("ProcessBench", "huggingface", "Qwen/ProcessBench", "primary: E1-E6, E8"),
    ("GSM8K", "huggingface", "openai/gsm8k", "questions for GPU generation (E4)"),
    ("MATH-500", "huggingface", "HuggingFaceH4/MATH-500", "loader only; not used by an experiment"),
    ("MATH (full)", "huggingface", "hendrycks/competition_math", "loader fallback only"),
    ("StrategyQA", "huggingface", "wics/strategy-qa", "loader only; E7 needs generated traces"),
    ("PRM800K", "github", "openai/prm800k", "loader only; not used by an experiment"),
    ("BIG-Bench Mistake", "github", "WHGTyen/BIG-Bench-Mistake", "E7 cross-domain transfer"),
]


def _get_json(url: str) -> tuple:
    request = urllib.request.Request(url, headers={"User-Agent": "choke-points-llm"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https hosts
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def verify_identifiers() -> list:
    """Return one verification record per identifier.

    Status values: RESOLVES, DISABLED_ON_HUB (repo page exists but its data
    has been taken down), NOT_FOUND (HTTP 404/401), ERROR (network failure).
    """
    records = []
    for dataset, host, identifier, role in IDENTIFIERS:
        url = (f"https://huggingface.co/api/datasets/{identifier}" if host == "huggingface"
               else f"https://api.github.com/repos/{identifier}")
        try:
            code, payload = _get_json(url)
            if code == 200 and payload is not None and payload.get("disabled"):
                status = "DISABLED_ON_HUB"
            elif code == 200:
                status = "RESOLVES"
            else:
                status = f"NOT_FOUND (HTTP {code})"
        except OSError as exc:
            status = f"ERROR ({exc.__class__.__name__})"
        records.append({"dataset": dataset, "host": host, "identifier": identifier,
                        "role": role, "status": status,
                        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return records
