# E10: where does a reasoning trajectory actually become doomed? (GPU)

Status: **code written and tested on a planted-effect mock; not yet run on a real model.**
Nothing in this file is a result.

## Question

ProcessBench tells us where the first *wrong step* is (e). Stage 11 shows that this is often
not where the outcome is decided: a quarter of first errors are not fatal, fatality does not
depend on where the error sits, and text detectors cannot tell fatal from non-fatal errors. The
missing measurement is causal: if we restart the model from just before step k, how often does
it still succeed?

For each trace and each prefix k = 0..n−1 we sample m continuations **from the trace's own
generator** and estimate

    v_k = P(final answer correct | question, first k steps)

v_0 is the from-scratch solve rate, i.e. problem difficulty for that model, so the analysis
controls for difficulty by construction rather than by a baseline.

## What is new relative to existing work

- Thought Anchors (Bogdan et al., 2025) and The Point of No Return (Merrill & Srivastava, 2026)
  resample from prefixes to find decisive sentences; the latter studies deception and names
  incorrect solutions as future work. Neither compares the counterfactual decision point with
  **human first-error annotations**.
- Zhang et al. (2025, "Lessons of developing PRMs") note that Monte Carlo rollouts and human
  labels disagree and treat the disagreement as label noise. Here the disagreement is the object
  of study, measured on-policy, and turned into a backtracking decision.
- Afzal et al. (2025) and Lugoloobi et al. (2026) predict success from hidden states; Bulut
  (2026) shows such pooled AUROCs are largely difficulty. v_0 removes that confound.

## Pre-registered hypotheses and how each is read (stage 13 output)

For failing traces that are not capability-limited (v_0 > δ):

| Hypothesis | Measured by | Supports "errors are not choke points" if |
|---|---|---|
| H1 the annotated first error is the decisive step | share with argmax drop at k = e, vs the 1/n chance rate | share is well below 1 |
| H2 latent commitment: success has collapsed before the written error | share "latent-commitment" (c < e) | non-trivial share |
| H3 recoverable errors: success survives the first error | share "recoverable-error" (c > e); mean drop at e for recovered traces | large share; small drop |
| H4 where to backtrack | success of resampling from before e vs before the decisive step vs restart, sample-split so the step is chosen and scored on different continuations | decisive-step backtracking beats first-error backtracking |

Noise floor: the largest per-trace drop on clean traces. Report every share at
δ ∈ {1/16, 1/8, 1/4}. A drop at e only counts if it exceeds the clean-trace noise floor.

Any outcome is publishable. If H1 holds (the first error is the decisive step), human first-error
labels are validated as intervention points, which is useful for PRM-guided search. If it fails,
PRM-guided backtracking is aimed at the wrong step, and stage 13 says how much success that costs.

## Settings

| Setting | Traces (clean / fatal / recovered) | Prefixes | Rollouts at m = 16 |
|---|---|---:|---:|
| Pilot: Qwen2.5-1.5B-Instruct | 53 / 51 / 12 | 704 | 11,264 |
| Main: Qwen2.5-Math-7B-Instruct | 140 / 210 / 47 | 3,312 | 52,992 |
| Replication: Qwen2.5-7B-Instruct | 116 / 172 / 63 | 2,801 | 44,816 |

Sampling: temperature 0.7, top-p 0.95, max 1,536 new tokens, the Qwen math system prompt.
Gold answers come from the source datasets (GSM8K, MATH-500 / hendrycks_math, OlympiadBench,
Omni-MATH), matched by problem text; traces without a gold answer are skipped and counted in
the log. Answers are compared with HuggingFace `math-verify` (string fallback).

Caveats to state in the paper: ProcessBench re-segmented the original solutions into
paragraphs, so prefixes are near-, not exactly, on-policy; temperature-0.7 continuations
measure the model's sampling distribution, not the greedy decode that produced the trace.

## Running it (Kaggle, 2x T4, or Colab)

```bash
git clone https://github.com/HarshDubey23/early-failure-detection-cot && cd early-failure-detection-cot
pip install -r requirements.txt vllm math-verify
python scripts/01_download_data.py --config configs/default.yaml   # needs internet (HF)
python scripts/02_preprocess.py --config configs/default.yaml
make commitment-mock                                               # 1 minute, CPU: checks the pipeline
make commitment GENERATOR=Qwen2.5-1.5B-Instruct M=16 DTYPE=half    # pilot, one T4
make commitment GENERATOR=Qwen2.5-Math-7B-Instruct M=16 TP=2 DTYPE=half   # main, 2x T4
```

T4s have no bfloat16, hence `DTYPE=half`. Stage 12 appends to its JSONL after every batch and
skips finished prefixes on restart, so a session that times out can be resumed. Run the pilot
first and read `results/commitment/<generator>.log` for the measured throughput before
committing GPU hours to the 7B runs; if time is short, start the 7B model with
`make commitment ... M=8` or `EXTRA="--max-per-group 60"`.

Outputs: `results/commitment/<generator>.jsonl` (every continuation's answer and correctness),
`results/commitment/tables_<generator>/{per_trace.csv, commitment_categories.csv, summary.txt}`.
