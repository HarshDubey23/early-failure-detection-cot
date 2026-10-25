"""
WHAT: Model wrappers for the recovery system: a generator (vLLM or mock), a hallucination
      detector (LettuceDetect or mock) and an LLM judge (vLLM or mock).

WHY:  The same three interfaces serve the offline experiment (rollout bank, scoring) and the
      live controller (`src/recovery/controller.py`), so the system that is timed live is the
      system that is evaluated at scale. The mock classes let the whole pipeline run on a CPU
      in tests; anything they produce is tagged backend="mock" and is never a result.

HOW:  vLLM settings follow the ones that worked on Kaggle's 2x T4 in Runs A-C: float16,
      eager mode, no custom all-reduce. Prompts are passed as token ids so no second BOS is
      added. Detector scores are per-sentence probabilities of containing hallucinated content.
"""
from __future__ import annotations

import hashlib
import random
import re
import time

from src.recovery import prompts as P
from src.recovery.ragtruth import sentence_spans

# ---------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------


def make_llm(**kw):
    """vLLM engine with the T4-safe settings of Runs A-C (eager, no custom all-reduce), with
    prefix caching when this vLLM version accepts the flag."""
    from vllm import LLM

    kw.update(enforce_eager=True, disable_custom_all_reduce=True)
    try:
        return LLM(enable_prefix_caching=True, **kw)
    except TypeError:
        return LLM(**kw)


class VLLMGenerator:
    def __init__(self, model_id: str, name: str, tp: int = 2, dtype: str = "half", max_model_len: int = 4096,
                 gpu_mem: float = 0.88, seed: int = 0):
        from transformers import AutoTokenizer

        self.name, self.model_id = name, model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.max_model_len = max_model_len
        self.llm = make_llm(model=model_id, tensor_parallel_size=tp, dtype=dtype, seed=seed,
                            max_model_len=max_model_len, gpu_memory_utilization=gpu_mem)
        self.backend = "vllm"

    def fmt(self, user_prompt: str, prefix: str = "") -> str:
        return P.format_for_model(self.tok, self.name, user_prompt, prefix)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts: list, n: int, temperature: float, top_p: float, max_tokens: int, seeds: list,
                 stop: list | None = None) -> list:
        """For each prompt string: list of (text, n_generated_tokens) or None if it does not fit.
        Also returns wall-clock seconds for the whole call (for throughput accounting)."""
        from vllm import SamplingParams

        keep, inputs, params = [], [], []
        for i, p in enumerate(prompts):
            ids = self.tok(p, add_special_tokens=True)["input_ids"]
            # a chat template already carries BOS; drop a duplicate
            if len(ids) > 1 and ids[0] == ids[1] == self.tok.bos_token_id:
                ids = ids[1:]
            room = self.max_model_len - len(ids)
            if room < 64:
                continue
            keep.append((i, len(ids)))
            inputs.append({"prompt_token_ids": ids})
            params.append(SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=-1,
                                         max_tokens=min(max_tokens, room), seed=seeds[i], stop=stop))
        out = [None] * len(prompts)
        t0 = time.time()
        if inputs:
            res = self.llm.generate(inputs, params, use_tqdm=False)
            for (i, n_prompt), r in zip(keep, res):
                out[i] = {"prompt_tokens": n_prompt, "samples": [(c.text, len(c.token_ids)) for c in r.outputs]}
        return out, time.time() - t0


class MockGenerator:
    """Fabricates text from the source prompt: for TESTING THE PIPELINE ONLY.

    Each sampled sentence is copied from the source (supported) or invented (contains the
    marker 'ZZFAKE'), with a hallucination probability that is higher after an invented
    sentence, so the analysis code has commitment-like structure to find.
    """

    backend = "mock"

    def __init__(self, name: str = "mock"):
        self.name = name
        self.model_id = "mock"

    def fmt(self, user_prompt: str, prefix: str = "") -> str:
        return f"{user_prompt}\x00{prefix}"

    def n_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate(self, prompts, n, temperature, top_p, max_tokens, seeds, stop=None):
        out = []
        for p, s in zip(prompts, seeds):
            user, prefix = p.split("\x00")
            rng = random.Random(s)
            words = re.findall(r"[A-Za-z]{4,}", user)[-300:] or ["source"]
            samples = []
            if "RESPONSE TO EDIT" in user:
                samples = [("DELETE" if rng.random() < 0.3 else f"The source mentions {rng.choice(words)}.", 8)
                           for _ in range(n)]
            elif "verification questions" in user:
                samples = [("1. Is it stated?\n2. Which number?", 12) for _ in range(n)]
            elif "QUESTION:" in user:
                samples = [(f"The source says {rng.choice(words)}.", 6) for _ in range(n)]
            else:
                bad_prefix = "ZZFAKE" in prefix
                for _ in range(n):
                    sents, bad = [], bad_prefix
                    for _ in range(rng.randint(1, 5)):
                        p_bad = 0.6 if bad else 0.15
                        if rng.random() < p_bad:
                            sents.append(f"It ZZFAKE {rng.choice(words)} {rng.randint(1, 99)}.")
                            bad = True
                        else:
                            sents.append(f"The text discusses {rng.choice(words)} and {rng.choice(words)}.")
                    txt = " ".join(sents)
                    samples.append((txt, self.n_tokens(txt)))
            out.append({"prompt_tokens": self.n_tokens(p), "samples": samples})
        return out, 0.001 * len(prompts)


def seed_for(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------------------------
# Detector (system detector D and evaluator E2): per-sentence hallucination probability
# ---------------------------------------------------------------------------------------------


class LettuceDetector:
    """KRLabsOrg/lettucedect-*-modernbert-en-v1 (MIT), trained on RAGTruth train.

    Input: the full task prompt (instruction + source) and the response; output: hallucinated
    character spans with confidences. A sentence's score is the highest confidence among spans
    overlapping it (0 if none)."""

    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 4096, batch_size: int = 8):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.model_path, self.device, self.max_length, self.bs = model_path, device, max_length, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
        self.backend = "lettucedetect-transformers"

    def _encode(self, prompts, answers, trunc):
        return self.tok(prompts, answers, truncation=trunc, max_length=self.max_length, padding=True,
                        return_offsets_mapping=True, return_tensors="pt")

    def score_batch(self, rows: list) -> list:
        """Token-level P(hallucinated) for the answer tokens of (prompt, answer) pairs, the input
        format LettuceDetect was trained on; sentence score = highest token probability in it."""
        torch = self.torch
        out = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]) + len(rows[i]["text"]))
        for s in range(0, len(order), self.bs):
            idx = [i for i in order[s: s + self.bs]]
            todo = [i for i in idx if rows[i]["text"].strip()]
            for i in idx:
                if not rows[i]["text"].strip():
                    out[i] = {"sent_scores": [], "spans": [], "n_sent": 0}
            if not todo:
                continue
            prompts, answers = [rows[i]["prompt"] for i in todo], [rows[i]["text"] for i in todo]
            try:
                enc = self._encode(prompts, answers, "only_first")
            except Exception:  # noqa: BLE001 - answer alone longer than max_length
                enc = self._encode(prompts, answers, True)
            offsets = enc.pop("offset_mapping")
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float()
            probs = torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()
            for b, i in enumerate(todo):
                seq = enc.sequence_ids(b)
                toks = [(int(offsets[b, t, 0]), int(offsets[b, t, 1]), float(probs[b, t]))
                        for t in range(len(seq)) if seq[t] == 1 and offsets[b, t, 1] > offsets[b, t, 0]]
                out[i] = token_probs_to_sentences(rows[i]["text"], toks)
        return out

    def score(self, source_prompt: str, response: str, context: str | None = None,
              question: str | None = None) -> dict:
        return self.score_batch([{"prompt": source_prompt, "text": response}])[0]


def token_probs_to_sentences(response: str, toks: list, thr: float = 0.5) -> dict:
    """toks: (char start, char end, P(hallucinated)) for answer tokens. Sentence score = max token
    probability inside it; spans = maximal runs of tokens with probability >= thr."""
    sents = sentence_spans(response)
    scores = [0.0] * len(sents)
    for a, b, p in toks:
        for j, (s, e) in enumerate(sents):
            if a < e and b > s:
                scores[j] = max(scores[j], p)
    spans, cur = [], None
    for a, b, p in toks:
        if p >= thr:
            if cur and a <= cur["end"] + 1:
                cur["end"], cur["confidence"] = b, max(cur["confidence"], p)
            else:
                cur = {"start": a, "end": b, "confidence": p}
                spans.append(cur)
        else:
            cur = None
    for sp in spans:
        sp["confidence"] = round(sp["confidence"], 4)
    return {"sent_scores": [round(x, 4) for x in scores], "spans": spans, "n_sent": len(sents),
            "covered_chars": max([b for _, b, _ in toks] or [0])}


def spans_to_sentences(response: str, spans: list) -> dict:
    sents = sentence_spans(response)
    scores = [0.0] * len(sents)
    clean_spans = []
    for sp in spans or []:
        a, b = int(sp.get("start", 0)), int(sp.get("end", 0))
        conf = float(sp.get("confidence", sp.get("prob", 1.0)))
        clean_spans.append({"start": a, "end": b, "confidence": round(conf, 4)})
        for j, (s, e) in enumerate(sents):
            if a < e and b > s:
                scores[j] = max(scores[j], conf)
    return {"sent_scores": [round(x, 4) for x in scores], "spans": clean_spans, "n_sent": len(sents)}


class MockDetector:
    """Flags sentences containing 'ZZFAKE' with noise: for TESTING THE PIPELINE ONLY."""

    backend = "mock"

    def __init__(self, noise: float = 0.15, seed: int = 0):
        self.noise, self.seed = noise, seed
        self.model_path = "mock"

    def score(self, source_prompt, response, context=None, question=None):
        rng = random.Random(seed_for(self.seed, response))
        spans = []
        for a, b in sentence_spans(response):
            bad = "ZZFAKE" in response[a:b]
            if (bad and rng.random() > self.noise) or (not bad and rng.random() < self.noise / 3):
                spans.append({"start": a, "end": b, "confidence": rng.uniform(0.6, 0.99)})
        return spans_to_sentences(response, spans)

    def score_batch(self, rows):
        return [self.score(r["prompt"], r["text"]) for r in rows]


# ---------------------------------------------------------------------------------------------
# LLM judge (evaluator E1)
# ---------------------------------------------------------------------------------------------


class VLLMJudge:
    def __init__(self, model_id: str, tp: int = 2, dtype: str = "half", max_model_len: int = 6144,
                 gpu_mem: float = 0.88, quantization: str | None = None):
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.max_model_len = max_model_len
        self.llm = make_llm(model=model_id, tensor_parallel_size=tp, dtype=dtype, max_model_len=max_model_len,
                            gpu_memory_utilization=gpu_mem, quantization=quantization, seed=0)
        self.backend = "vllm"

    def judge(self, rows: list) -> list:
        """rows: dicts with prompt (source prompt) and text; returns parsed verdicts."""
        from vllm import SamplingParams

        inputs, meta = [], []
        for i, r in enumerate(rows):
            sents = [r["text"][a:b] for a, b in sentence_spans(r["text"])]
            if not sents:
                meta.append((i, 0, True))
                continue
            msgs = [{"role": "system", "content": P.JUDGE_SYSTEM},
                    {"role": "user", "content": P.judge_prompt(r["prompt"], sents)}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > self.max_model_len - 128:
                meta.append((i, len(sents), False))
                continue
            inputs.append({"prompt_token_ids": list(ids)})
            meta.append((i, len(sents), None))
        params = SamplingParams(n=1, temperature=0.0, max_tokens=96)
        res = self.llm.generate(inputs, params, use_tqdm=False) if inputs else []
        out, it = [None] * len(rows), iter(res)
        for i, n, flag in meta:
            if flag is True:
                out[i] = {"unsupported": [], "n_sent": 0, "ok": True, "raw": ""}
            elif flag is False:
                out[i] = {"unsupported": None, "n_sent": n, "ok": False, "raw": "too_long"}
            else:
                txt = next(it).outputs[0].text
                parsed = P.parse_judge(txt, n)
                out[i] = {"unsupported": parsed, "n_sent": n, "ok": parsed is not None, "raw": txt[:300]}
        return out


class MockJudge:
    backend = "mock"
    model_id = "mock"

    def judge(self, rows):
        out = []
        for r in rows:
            sp = sentence_spans(r["text"])
            bad = [j + 1 for j, (a, b) in enumerate(sp) if "ZZFAKE" in r["text"][a:b]]
            out.append({"unsupported": bad, "n_sent": len(sp), "ok": True, "raw": ""})
        return out
