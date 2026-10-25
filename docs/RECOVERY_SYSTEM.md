# Recovering from hallucinations: system, experiment and how to run it

Status: **built and tested on CPU with fake models; preregistered
([`PREREGISTRATION_RECOVERY.md`](PREREGISTRATION_RECOVERY.md)); GPU run (Run R) not yet done.**
No hallucination-reduction number exists yet. Everything below describes what will be measured
and how.

## Why this follows from the math study

Runs A-C showed three things on ProcessBench:

- The human-annotated first error is often not where a reasoning trace becomes unrecoverable.
- Where to resample a failing trace from depends on the model.
- At a matched budget, restarting is very hard to beat.

Retrieval-augmented generation differs in one way that matters. Whether a response is faithful to
its source can be checked while the system runs, by a detector reading the source. So the system
can measure, at inference time, how recoverable the response still is from each sentence boundary.
That is the counterfactual value v_k of the math study, estimated online. It can then decide what
to do.

## The system (`src/recovery/controller.py`)

```
draft ──► detector D ──► no flagged sentence ─────────────────────────────► return draft
                    │
                    └─► flagged at sentence j*
                          1. claim repair: rewrite each flagged sentence ─► passes D? return
                          2. resample within the budget, choosing between
                             restart (k = 0) and backtrack (k = j* − 1) by Thompson sampling
                             on each boundary's recoverability per token ─► passes D? return
                          3. QA: abstain ("Unable to answer based on given passages.")
                             other tasks: delete the sentences D still flags
```

- D is LettuceDetect-base, a RAGTruth-trained token classifier. Inference is in
  `src/recovery/engine.py`.
- Every step's cost is counted in generator-equivalent tokens (`src/recovery/replay.py`).
- The same decision logic runs offline in replay and online in the live controller.

## The experiment

| Stage | Script | Hardware |
|---|---|---|
| label analysis of RAGTruth (no model) | `scripts/30_ragtruth_label_analysis.py` | CPU, minutes |
| benchmark items (RAGTruth test; HotpotQA with controlled evidence) | `scripts/31_build_items.py` | CPU |
| drafts, rollout bank, repairs, Chain-of-Verification | `scripts/32_generate.py` | GPU (vLLM) |
| detector D, evaluators E2 and E1 | `scripts/33_score.py` | GPU |
| replay of every policy at matched budgets, tests | `scripts/34_replay.py` | CPU |
| commitment of hallucinations vs human labels | `scripts/35_mechanism.py` | CPU |
| live latency of the deployed loop | `scripts/36_live.py` | GPU |
| check of the detector implementation | `scripts/37_check_detector.py` | GPU |
| Kaggle driver (2x T4, resumable, archived per part) | `scripts/kaggle_recovery.sh` | GPU |

The comparisons:
- base model;
- Chain-of-Verification (verification and revision);
- claim repair alone;
- restart and backtrack at the same budget;
- the controller with its parts removed;
- an oracle reference that knows the true v_k.

Each output is scored by:
- an LLM judge from a different model family (E1), checked against RAGTruth's human labels;
- LettuceDetect-large (E2).

## Running it

```bash
# CPU: tests and the full pipeline with fake models (no network, no GPU)
python -m pytest tests/test_recovery.py -q
MOCK=1 ARCHIVE=/tmp/recovery_mock.tar.gz bash scripts/kaggle_recovery.sh

# CPU: what RAGTruth's human labels say (needs the dataset from GitHub)
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth data/external/RAGTruth
python scripts/30_ragtruth_label_analysis.py

# Kaggle, 2x T4: set SMOKE=1 or SMOKE=0 in scripts/recovery_mode, then in the notebook
git clone -b hallucination-recovery https://github.com/HarshDubey23/early-failure-detection-cot
cd early-failure-detection-cot && bash scripts/kaggle_recovery.sh

# CPU, after downloading /kaggle/working/recovery_results.tar.gz and extracting it here
python scripts/34_replay.py --items data/recovery/items.*.jsonl --out results/recovery/final
python scripts/35_mechanism.py --items data/recovery/items.ragtruth.llama-2-7b-chat.jsonl
```

## What the RAGTruth labels already show (descriptive, no model run)

From `results/recovery/ragtruth_labels/summary.json`. These are human labels on 17,617
good-quality responses from six generators.

- Hallucination is concentrated by generator and task. Test-split rates are:
  - by generator: 9-10% for GPT-4 and GPT-3.5, 50% for Llama-2-7B-chat, 58% for Mistral-7B;
  - by task: 64% on Data2txt against 18-23% on QA and summaries.
- Generator and task alone predict whether a response hallucinates with AUROC 0.83. Adding the
  source raises it to 0.85.
- Hallucinated content is mostly local:
  - 96% of hallucinated responses have at most half of their sentences hallucinated;
  - 55% have all of it in one sentence (84% for summaries).
- Hallucinated sentences cluster beyond what their number implies. This uses a runs test with the
  count held fixed: 1.34 runs observed against 1.57 expected, p < 0.001. The first one sits later
  than chance.
- On QA sources where annotators accepted a refusal as correct, the generators that answered anyway
  hallucinated 46% of the time, against 28% on sources with no refusal.

These observations motivate repairing or backtracking rather than always restarting, and abstaining
when evidence is missing. Whether those actions actually reduce hallucinations at a matched budget
is what Run R tests.
