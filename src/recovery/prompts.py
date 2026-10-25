"""
WHAT: Every prompt the recovery system and its baselines send to a model.

WHY:  Prompts are part of the method. Keeping them in one module, versioned with the code,
      makes the comparison auditable: the commitment-aware controller, the fixed strategies and
      the verify-and-revise baseline all use the same generator calls and the same repair text.

HOW:  Plain string builders. Formatting for a particular chat model happens in
      `format_for_model`, which reproduces how RAGTruth produced its responses for Llama-2
      ("<s>[INST] {prompt} [/INST]"; the tokenizer adds <s>) and uses the tokenizer's chat
      template for every other model.
"""
from __future__ import annotations

import json
import re

REFUSAL = "Unable to answer based on given passages."

RAW_INST_MODELS = ("llama-2", "Llama-2", "mistral-7B-instruct", "Mistral-7B-Instruct-v0.1")


def format_for_model(tok, model_name: str, user_prompt: str, prefix: str = "") -> str:
    """Full prompt string, followed by `prefix` (an assistant prefill) when given."""
    if any(t in model_name for t in RAW_INST_MODELS):
        head = f"[INST] {user_prompt.strip()} [/INST]"
        return head + (" " + prefix if prefix else "")
    messages = [{"role": "user", "content": user_prompt}]
    head = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return head + prefix


# ---------------------------------------------------------------------------------------------
# Generation prompts for new benchmark items (RAGTruth items reuse the released prompt)
# ---------------------------------------------------------------------------------------------

def qa_prompt(question: str, passages: list) -> str:
    """RAGTruth's QA instruction, so that detectors trained on RAGTruth see a familiar format."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten"}
    body = "".join(f"passage {i + 1}:{p}\n\n" for i, p in enumerate(passages))
    n = words.get(len(passages), str(len(passages)))
    return (f"Briefly answer the following question:\n{question}\n"
            f"Bear in mind that your response should be strictly based on the following {n} passages:\n"
            f"{body}In case the passages do not contain the necessary information to answer the question, "
            f"please reply with: \"{REFUSAL}\"\noutput:")


# ---------------------------------------------------------------------------------------------
# Claim repair (rewrite one flagged sentence)
# ---------------------------------------------------------------------------------------------

def repair_prompt(source_prompt: str, draft: str, j: int, sentence: str) -> str:
    """The task, source and full draft come first, so every flagged sentence of one draft shares
    the same prompt prefix (served once from the prefix cache); only the tail differs."""
    return (
        f"ORIGINAL TASK (with the source):\n{source_prompt}\n\n"
        f"RESPONSE TO EDIT:\n{draft.strip()}\n\n"
        "You are editing this response so that it is fully supported by the source.\n"
        f"SENTENCE {j} (may contain information that is not in the source): {sentence.strip()}\n\n"
        f"Rewrite SENTENCE {j} so that every claim in it is stated in, or directly follows from, the source. "
        "Keep it as close to the original as possible. If no version of it is supported by the source, reply "
        "with exactly: DELETE\n"
        "Reply with the rewritten sentence only.")


def parse_repair(text: str) -> str | None:
    """Rewritten sentence, or None for DELETE / empty output."""
    t = (text or "").strip()
    t = re.sub(r"^(rewritten sentence|sentence|answer)\s*:\s*", "", t, flags=re.I).strip()
    t = t.split("\n")[0].strip().strip('"').strip()
    if not t or t.upper().startswith("DELETE"):
        return None
    return t


# ---------------------------------------------------------------------------------------------
# Chain-of-Verification (factored), adapted to grounded generation: the standard
# verification-and-revision baseline (Dhuliawala et al., 2023)
# ---------------------------------------------------------------------------------------------

def cove_plan_prompt(source_prompt: str, draft: str, max_q: int = 5) -> str:
    return (
        f"ORIGINAL TASK (with the source):\n{source_prompt}\n\nDRAFT RESPONSE:\n{draft.strip()}\n\n"
        f"Write up to {max_q} short verification questions that check the specific factual claims in the "
        "DRAFT RESPONSE (names, numbers, dates, attributes, events). One question per line, numbered 1., 2., ... "
        "Write nothing else.")


def parse_questions(text: str, max_q: int = 5) -> list:
    qs = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*(?:\d+[\.\)]|[-*•])\s*(.+)", line)
        if m and len(m.group(1).strip()) > 3:
            qs.append(m.group(1).strip())
    return qs[:max_q]


def cove_answer_prompt(source_prompt: str, question: str) -> str:
    # The draft is NOT shown (factored CoVe): the answer cannot copy the draft's mistake.
    return (f"ORIGINAL TASK (with the source):\n{source_prompt}\n\n"
            f"Using ONLY the source above, answer this question in one sentence. If the source does not "
            f"say, answer exactly: Not stated in the source.\nQUESTION: {question}")


def cove_revise_prompt(source_prompt: str, draft: str, qa: list) -> str:
    checks = "\n".join(f"Q: {q}\nA: {a.strip()}" for q, a in qa) or "(no checks)"
    return (
        f"ORIGINAL TASK (with the source):\n{source_prompt}\n\nDRAFT RESPONSE:\n{draft.strip()}\n\n"
        f"VERIFICATION (answered from the source only):\n{checks}\n\n"
        "Write the final response to the ORIGINAL TASK. Keep everything in the draft that the source supports, "
        "correct anything the verification contradicts, and remove anything the source does not state. "
        "Reply with the final response only.")


# ---------------------------------------------------------------------------------------------
# LLM judge (evaluator E1): which sentences are unsupported by the source?
# ---------------------------------------------------------------------------------------------

JUDGE_SYSTEM = ("You are a meticulous fact-checker for retrieval-augmented generation. You decide whether each "
                "sentence of a response is supported by the given source material.")


def judge_prompt(source_prompt: str, sentences: list) -> str:
    numbered = "\n".join(f"[{i + 1}] {s.strip()}" for i, s in enumerate(sentences))
    return (
        f"SOURCE MATERIAL AND TASK GIVEN TO THE WRITER:\n<<<\n{source_prompt}\n>>>\n\n"
        f"RESPONSE, ONE NUMBERED SENTENCE PER LINE:\n{numbered}\n\n"
        "A sentence is UNSUPPORTED if any claim in it contradicts the source material, or adds a fact that is "
        "not stated in and cannot be directly inferred from the source material (even if it is true in the "
        "real world). Sentences that only introduce, structure or politely close the response, and sentences "
        "saying that the source does not contain the answer, are supported unless they assert a new fact.\n"
        "Answer with JSON only, in exactly this form: {\"unsupported\": [list of sentence numbers]}. "
        "Use an empty list if every sentence is supported.")


def parse_judge(text: str, n: int) -> list | None:
    """Sorted 1-based sentence numbers judged unsupported; None if the output cannot be parsed."""
    if text is None:
        return None
    m = re.search(r"\{[^{}]*\"unsupported\"\s*:\s*\[([^\]]*)\][^{}]*\}", text, flags=re.S)
    if not m:
        m = re.search(r"unsupported\W*\[([^\]]*)\]", text, flags=re.S | re.I)
    if not m:
        return None
    nums = []
    for tok in re.findall(r"\d+", m.group(1)):
        k = int(tok)
        if 1 <= k <= n:
            nums.append(k)
    return sorted(set(nums))


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)
