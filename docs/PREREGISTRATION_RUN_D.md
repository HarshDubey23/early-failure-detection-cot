# Preregistration: Run D (long-reasoning models)

Written on 2026-10-27, **before any Run D outcome data are analysed**. The commit that adds this
file is the timestamp; the full run is launched from a later commit, and every deviation is reported.
A pilot on MATH-500 problems 1-20 (dataset order) was run before this file only to measure speed,
token use and output format. Those 20 problems are excluded from every analysis below.

## 1. Why this run

Runs A-C used instruction models on ProcessBench solutions that were written elsewhere, so their
prefixes were only near on-policy. The paper's Proposition 1 holds exactly for on-policy traces, and
the field's question today is about long-reasoning models that already backtrack inside their own
reasoning. Run D asks, for two such models: is a failed partial reasoning trace worth more than a
fresh start, and does resampling from any point of it beat restarting at matched compute?

## 2. Data collected (`scripts/60_reasoning_rollouts.py`, driver `scripts/kaggle_run_d.py`)

- **Models.** Qwen/Qwen3-1.7B in thinking mode (temperature 0.6, top-p 0.95, top-k 20) and
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (temperature 0.6, top-p 0.95), each with its recommended
  settings; float16 on 2x T4 with vLLM.
- **Problems.** MATH-500 problems 21-500 in dataset order (480 problems).
- **Traces.** One trace per problem per model, at most 6,144 generated tokens. The final answer is the
  last `\boxed{}` after `</think>`, compared with the gold answer by math-verify. A trace **fails** if
  that answer is wrong or missing (including traces cut at the token budget).
- **Cuts.** For each failed trace, fixed token positions t in {0, 256, 512, 1024, 1536, 2048, 3072,
  4096, 5120} of its reasoning, each moved back to the last paragraph break before t. A cut therefore
  depends only on text already written (a stopping time). If the reasoning ended before t, the cut is
  the end of the trace and its value is 0 (the trace failed).
- **Continuations.** m continuations from every cut, with the same sampling settings and the same
  total budget (prefix plus continuation at most 6,144 tokens). v_t is the share that are correct.
  m is the largest of {8, 12, 16} that the pilot's measured throughput fits into the available GPU
  time; this choice uses no outcome data.

## 3. Hypotheses and tests (per model; pooled over models as secondary)

Intervals: 95% problem-level bootstrap, 2,000 resamples. Sample splitting: 50 random halves.

| ID | Hypothesis | Test | Confirmed if |
|---|---|---|---|
| D1 | Proposition 1 on exactly on-policy traces: for every fixed cut t, a failed trace cut at t is worth no more than a fresh start, mean(v_t - v_0) <= 0 | one-sided bootstrap test of a positive mean at each of the 8 cuts, Holm-adjusted | **refuted** if any Holm-adjusted p < 0.05; supported otherwise |
| D2 | At matched compute, no resampling point of a failed trace beats restart by 5 points or more | P(solved) within B restart-equivalents of generated tokens; the cut is chosen on half A, scored on half B, minus restart on half B; B in {1, 2, 4} | upper 95% bound at B = 4 is below +0.05 |
| D3 | (descriptive) how many failures are decided before the first token, and where the largest drop in v falls | share with v_0 <= 1/8; histogram of the cut after which v falls most | reported, no test |
| D4 | (sensitivity) D1 and D2 without the traces that hit the token budget | as D1, D2 | reported |

The theory predicts D1 holds. D2 is the practical question; a positive answer (a cut that beats
restart by 5 points or more) would mean that reasoning models change the paper's conclusion, and it
will be reported as such.

## 4. Analysis code

`scripts/61_reasoning_analysis.py`, committed with this file and tested on mock data, computes every
quantity above. Any change after the data arrive will be listed here as a deviation.

## 5. Deviations, fixed before the confirmatory launch (no outcome data seen)

- **Pilot facts used.** The pilot (Qwen3-1.7B, 8 problems, m = 4, 4,096-token budget) ran at about
  130 generated tokens per second per T4 in eager mode. Both pilot failures hit the token budget, and
  most continuations did too. No value curve from the pilot is used anywhere.
- **m = 8.** This is the smallest value in the preregistered set. At the pilot's throughput, even m = 8
  does not fit all 480 problems into one Kaggle session.
- **Coverage set by time, not by outcome.** Each model runs in its own Kaggle session (2x T4, about
  6.5 hours). Stage 1 (one trace per problem) covers all 480 problems. Stage 2 (value curves) then
  processes failed traces in dataset order, 8 at a time, until the time limit. The analysed failures
  are therefore the first failures in dataset order, not a selection by outcome. The number reached is
  reported.
- **Speed settings only.** CUDA graphs replace eager mode, with eager kept as the fallback, and memory
  use is raised from 0.90 to 0.92. These change speed, not sampling.
- The token budget stays at 6,144, and the cuts, sampling settings, hypotheses and analysis code are
  unchanged.

## 6. Outcomes

### 6.1 Qwen3-1.7B, first session (2026-10-27, 6.6 h on 2x T4): stage 1 complete, stage 2 empty

- **Stage 1 (480 problems).** Accuracy 70.8% (340/480): by level 95%, 93%, 81%, 71%, 41%.
  Correct traces used 3,008 tokens on average.
- **Failures are almost all budget exhaustion.** Of 140 failures, 133 (95%) hit the 6,144-token
  budget without closing the reasoning block. Only 7 ended with a wrong final answer, and two of
  those are grading false negatives: the gold answer is a `pmatrix` vector and the model wrote the
  same vector as a tuple.
- **Stage 2 wrote no rows.** Throughput was about 92 generated tokens per second per T4, below the
  pilot's eager-mode rate (TRITON attention, since FlashAttention 2 needs compute capability 8). The
  first chunk (8 failed traces x 7 cuts x 8 continuations = 560 requests) was 61% done when the time
  limit stopped it, and a chunk is written only when it completes, so nothing was kept.
- **What this means for the hypotheses.** For a trace that fails by running out of budget, a later
  cut leaves less budget for its continuations, so a fall in v_t is partly mechanical. D1 and D2 on
  such traces test the budget more than the reasoning. D4 (wrong-answer failures only) is the
  informative subgroup, but it has at most 7 traces for this model.
- **Changes for the next session (before any stage-2 data).** Stage-1 traces are committed and
  reused, so a session goes entirely to stage 2. Chunks are 2 failed traces, so a time-limit kill
  loses at most one chunk.

### 6.2 DeepSeek-R1-Distill-Qwen-1.5B (2026-10-27/28, 6.6 h on 2x T4): stage 1 complete, stage 2 for 40 failures

- **Stage 1 (480 problems).** Accuracy 72.7% (349/480): by level 93%, 92%, 84%, 72%, 46%. Correct
  traces used 2,456 tokens on average. Of 131 failures, 120 (92%) hit the 6,144-token budget; 11
  ended with a wrong answer, at least one of them a grading false negative (`E` for `\text{east}`).
- **Stage 2.** The first 40 failures in dataset order were completed before the time limit (9 cuts,
  8 continuations each, 10.5 million generated tokens). 36 of the 40 are budget truncations.
  `scripts/61_reasoning_analysis.py --exclude-first 0` (the run already skipped problems 1-20).
- **D1: supported.** No cut is worth more than a fresh start. Mean v(t) - v(0) is +0.000 at t = 256 and
  falls steadily to -0.212 [-0.303, -0.128] at t = 5,120; no Holm-adjusted p is below 0.05.
- **D2: the preregistered rule is not met, but the result is fragile.** With the preregistered estimator
  (Jeffreys posterior pass rate), the best cut beats restart by +0.133 [+0.067, +0.197] at a 4x budget,
  so "restart not beaten" fails. Per attempt, no cut beats restart (D1); the whole gain comes from
  cheaper attempts. A late cut leaves a short budget, so its continuations cost as little as 1,106
  tokens against 5,174 for a restart, and buy up to about 19 attempts from 4 observed samples. The
  Jeffreys prior gives a cut with 0 of 4 successes about a 10% chance per attempt, and many such
  attempts add up. A plug-in sensitivity (pass rate 1 - (1 - c/n)^N, added after the data, so
  exploratory) gives +0.031 [-0.045, +0.114]: positive but not significant. The honest reading: on
  this model, resampling late in a failed trace may pay **only through shorter attempts**, as
  Proposition 2 allows, and our samples are too few to tell whether it does.
- **D3.** 60% [45%, 75%] of the 40 failures are capability-limited (v(0) <= 1/8). Mean v(0) is 0.256.
  Among the rest, the largest drop falls early: 9 of 16 at or before t = 1,024.
- **D4.** Only 4 wrong-answer failures have value curves, too few to test.

