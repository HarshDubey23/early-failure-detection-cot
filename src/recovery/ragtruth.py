"""
WHAT: RAGTruth loading, sentence segmentation and span-to-sentence labels.

WHY:  The recovery system acts at sentence boundaries: it keeps a verified prefix and
      resamples (backtracks), regenerates from scratch (restarts), rewrites one sentence
      (repairs) or declines to answer (abstains). RAGTruth gives human hallucination spans at
      character level; this module turns them into per-sentence labels so that "the first
      hallucinated sentence" plays the role ProcessBench's first error played in the math study.

HOW:  Pure functions over the two RAGTruth JSONL files (github.com/ParticleMedia/RAGTruth,
      MIT licence for the data release). No model code here, so it runs on any CPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REFUSAL = "Unable to answer based on given passages."
_REFUSAL_RE = re.compile(r"unable to answer", re.I)
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "no", "inc", "ltd", "co", "corp", "e.g", "i.e",
           "u.s", "u.k", "a.m", "p.m", "approx", "est", "dept", "mt", "ft", "jan", "feb", "mar", "apr", "jun",
           "jul", "aug", "sep", "sept", "oct", "nov", "dec", "fig", "al"}


def load(root: str | Path) -> tuple:
    """(responses, sources): lists of dicts / {source_id: dict} from the RAGTruth release."""
    root = Path(root)
    resp = [json.loads(line) for line in open(root / "dataset" / "response.jsonl", encoding="utf-8")]
    src = {}
    for line in open(root / "dataset" / "source_info.jsonl", encoding="utf-8"):
        r = json.loads(line)
        src[r["source_id"]] = r
    return resp, src


def sentence_spans(text: str) -> list:
    """[(start, end), ...] covering `text` (each segment owns its trailing whitespace).

    Splits at newlines and after . ! ? (optionally followed by a closing quote or bracket) when
    the next non-space character starts a new sentence. Common abbreviations, initials and
    decimals do not split.

    Example:
        >>> t = "Dr. Smith arrived at 3.5 p.m. on Monday. He left.\\n- Item two"
        >>> [t[a:b].strip() for a, b in sentence_spans(t)]
        ['Dr. Smith arrived at 3.5 p.m. on Monday.', 'He left.', '- Item two']
    """
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        cut = None
        if ch == "\n":
            j = i
            while j < n and text[j].isspace():
                j += 1
            cut = j
        elif ch in ".!?":
            j = i + 1
            while j < n and text[j] in "\"')]”’":
                j += 1
            if j < n and text[j] in " \t":
                k = j
                while k < n and text[k] in " \t":
                    k += 1
                nxt = text[k] if k < n else ""
                word = re.findall(r"([A-Za-z][A-Za-z.]*)$", text[start:i])
                w = word[0].lower().rstrip(".") if word else ""
                is_abbrev = ch == "." and (w in _ABBREV or (len(w) == 1 and w.isalpha()))
                if nxt and (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘(*-•") and not is_abbrev:
                    cut = k
        if cut is not None and text[start:cut].strip():
            spans.append((start, cut))
            start = cut
            i = cut
            continue
        i += 1
    if start < n and text[start:].strip():
        spans.append((start, n))
    elif spans and start < n:  # trailing whitespace only: give it to the last sentence
        spans[-1] = (spans[-1][0], n)
    return spans


def sentence_labels(text: str, labels: list, spans: list | None = None) -> list:
    """Per sentence: 1 if any hallucination span overlaps it, else 0.

    Example:
        >>> t = "A is true. B is made up. C is fine."
        >>> sentence_labels(t, [{"start": 11, "end": 23}])
        [0, 1, 0]
    """
    spans = spans if spans is not None else sentence_spans(text)
    out = [0] * len(spans)
    for lab in labels:
        a, b = int(lab["start"]), int(lab["end"])
        for j, (s, e) in enumerate(spans):
            if a < e and b > s:
                out[j] = 1
    return out


def mechanism_points(text: str, labels: list, item_id: str, placebo_rels: list, n_anchor: int = 2) -> dict:
    """Sentence boundaries to resample for the commitment study of one human-labelled response.

    Hallucinated response, first hallucinated sentence e (1-based): k in {0, e-2, e-1, e, e+1}
    (k = number of kept sentences; k = e keeps the hallucinated sentence). Clean response
    (placebo): k = 0 and {a-1, a} for `n_anchor` anchors a drawn from the relative positions e/n
    of the hallucinated responses (position-matched), seeded by the item id.

    Example:
        >>> t = "A one. B two. C three. D four. E five."
        >>> mechanism_points(t, [{"start": 14, "end": 22}], "x", [0.5])
        {'e': 3, 'points': [0, 1, 2, 3, 4], 'anchors': []}
    """
    import random as _random

    lab = sentence_labels(text, labels)
    n = len(lab)
    if any(lab):
        e = lab.index(1) + 1
        pts = sorted(k for k in {0, e - 2, e - 1, e, e + 1} if 0 <= k <= n - 1)
        return {"e": e, "points": pts, "anchors": []}
    rng = _random.Random(item_id)
    anchors = sorted({min(max(1, round(r * n)), n - 1) for r in rng.sample(sorted(placebo_rels), min(n_anchor, len(placebo_rels)))}) \
        if n >= 2 and placebo_rels else []
    pts = sorted({0} | {a - 1 for a in anchors} | set(anchors))
    return {"e": None, "points": pts, "anchors": anchors}


def is_refusal(text: str) -> bool:
    """A response that declines to answer (the instruction RAGTruth QA prompts give)."""
    return bool(_REFUSAL_RE.search(text)) and len(text.split()) <= 20


def generation_prompt(source: dict, model: str) -> str:
    """The prompt string RAGTruth used. Llama-2 / Mistral responses were produced with
    `<s>[INST] {prompt} [/INST]`; the chat template of the serving library is applied
    elsewhere, so this returns the bare instruction."""
    return source["prompt"]


def context_text(source: dict) -> str:
    """Evidence the response must stay faithful to (passages, structured data or article)."""
    info = source["source_info"]
    if source["task_type"] == "QA":
        return info["passages"]
    if source["task_type"] == "Summary":
        return info
    return json.dumps(info, ensure_ascii=False) if not isinstance(info, str) else info


def question_text(source: dict) -> str | None:
    info = source["source_info"]
    return info.get("question") if isinstance(info, dict) else None
