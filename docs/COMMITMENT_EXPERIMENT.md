# E10: where does a reasoning trajectory actually become doomed? (GPU)

Status: **run on three models (Kaggle, 2x T4, 2026-10-24); raw rollouts in `results/commitment/`.**
The numbers to cite are in "Final analysis" at the end (stage 13 with repeated sample splitting);
the run log above it keeps the numbers as first read off the notebook, for the record.

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

Planned (before running):

| Setting | Traces (clean / fatal / recovered) | Prefixes | Rollouts at m = 16 |
|---|---|---:|---:|
| Pilot: Qwen2.5-1.5B-Instruct | 53 / 51 / 12 | 704 | 11,264 |
| Main: Qwen2.5-Math-7B-Instruct | 140 / 210 / 47 | 3,312 | 52,992 |
| Replication: Qwen2.5-7B-Instruct | 116 / 172 / 63 | 2,801 | 44,816 |

Run (what the archive contains): Qwen2.5-1.5B-Instruct 53 / 51 / 12, 704 prefixes, m = 16 (11,264
rollouts); Qwen2-1.5B-Instruct 15 / 67 / 6, 446 prefixes, m = 16 (7,136); Qwen2.5-Math-7B-Instruct
60 / 207 / 47 (clean capped at 60; 3 traces without gold skipped), 2,646 prefixes, m = 12 (31,752).
Total 50,152 scored continuations. Qwen2.5-7B-Instruct was not run (GPU time).

Sampling: temperature 0.7, top-p 0.95, max 1,536 new tokens (1,024 for the 7B model), the Qwen math
system prompt.
Gold answers come from the source datasets (GSM8K, MATH-500 / hendrycks_math, OlympiadBench,
Omni-MATH), matched by problem text; traces without a gold answer are skipped and counted in
the log. Answers are compared with HuggingFace `math-verify` (string fallback).

Caveats to state in the paper: ProcessBench re-segmented the original solutions into
paragraphs, so prefixes are near-, not exactly, on-policy; temperature-0.7 continuations
measure the model's sampling distribution, not the greedy decode that produced the trace.

## Running it (Kaggle, 2x T4)

Tested on Kaggle (2x Tesla T4, 2026-10-24): transformers 5.16 and vLLM 0.32 (torch 2.13 cu130)
both run; vLLM is about 4-5x faster and is the default for the real runs.

```bash
git clone -b novelty-upgrade https://github.com/HarshDubey23/early-failure-detection-cot && cd early-failure-detection-cot
bash scripts/kaggle_commitment.sh setup                       # math-verify + ProcessBench from HF
bash scripts/kaggle_commitment.sh vllm                        # installs vLLM, removes the mismatched torchaudio
BACKEND=vllm bash scripts/kaggle_commitment.sh smoke Qwen2.5-1.5B-Instruct          # 3 traces, m = 4
BACKEND=vllm bash scripts/kaggle_commitment.sh run Qwen2.5-1.5B-Instruct 16         # one process per GPU
BACKEND=vllm TP=2 bash scripts/kaggle_commitment.sh run Qwen2.5-Math-7B-Instruct 16 # 7B over both T4s
```

T4s have no bfloat16, so everything runs in float16; vLLM runs eager with custom all-reduce off
(compatibility settings for Turing GPUs). Prompts that do not fit the context window are recorded
as skipped and their traces dropped from the analysis. Stage 12 appends to its JSONL after every
batch and skips finished prefixes on restart, so a session that times out can be resumed.

Outputs: `results/commitment/<generator>.jsonl` (every continuation's answer and correctness),
`results/commitment/tables_<generator>/{per_trace.csv, commitment_categories.csv, summary.txt}`.

## Run log

**Run A — interactive pilot, 2026-10-24, Kaggle 2x T4, vLLM 0.32.** Qwen2.5-1.5B-Instruct on all
116 of its ProcessBench traces (53 clean, 51 fatal, 12 recovered), m = 16, temperature 0.7,
max 1,536 new tokens, ~20 minutes. Numbers below were read off the stage-13 summary printed in the
notebook; the raw JSONL of this interactive session was not kept (the session was stopped to start
Run B, which repeats this configuration with independent sampling seeds and saves everything).

| Quantity (fatal traces, δ = 1/8) | Value | 95% CI |
|---|---:|---|
| capability-limited (v_0 ≤ δ) | 0.353 | 0.235–0.490 |
| latent commitment (c < e) | 0.176 | 0.078–0.294 |
| commits at the annotated error (c = e) | 0.412 | 0.275–0.549 |
| recoverable error (c > e) | 0.059 | 0.000–0.137 |
| first error is the decisive step (not capability-limited, n = 33) | 0.606 | 0.455–0.758 (chance 0.181) |
| mean drop in v at the first error, fatal | +0.234 | +0.165 to +0.311 |
| mean drop in v at the first error, recovered | −0.099 | −0.318 to +0.089 |
| noise floor: largest per-trace drop, clean | +0.093 | +0.065 to +0.125 |
| backtrack: restart from scratch | 0.489 | 0.390–0.583 |
| backtrack: just before the annotated first error | 0.386 | 0.280–0.492 |
| backtrack: just before the decisive step (sample-split) | 0.508 | 0.413–0.602 |
| backtrack: best prefix (oracle, in-sample) | 0.574 | 0.490–0.655 |
| decisive step minus first error | +0.121 | +0.049 to +0.205 |

Event study (mean v at k − e = −1, 0, +1): fatal 0.257 → 0.023 → 0.014; recovered 0.667 → 0.766 →
0.790; clean (random anchor) 0.868 → 0.923 → 0.966.

**Run B — background "Save Version" run, started 2026-10-24 13:13 IST.** `scripts/kaggle_full.sh`:
Qwen2.5-1.5B-Instruct (m = 16, independent replicate of Run A), Qwen2-1.5B-Instruct (m = 16),
Qwen2.5-Math-7B-Instruct (TP = 2, m = 12, clean traces capped at 60, 8.5-hour limit). Outputs:
`/kaggle/working/commitment_results.tar.gz` in the notebook version's output.

**Run B, small models (finished 2026-10-24 13:55 IST; read from the version log).** Same analysis
code as Run A (selection of "not capability-limited" on all samples; the corrected held-out
selection is applied when the archive is re-analysed).

| Quantity | Qwen2.5-1.5B (replicate) | Qwen2-1.5B |
|---|---|---|
| fatal traces: capability-limited | 0.333 [0.196, 0.471] | 0.522 [0.403, 0.642] |
| latent commitment (c < e) | 0.216 [0.098, 0.333] | 0.134 [0.060, 0.224] |
| commits at the error (c = e) | 0.373 [0.255, 0.510] | 0.239 [0.134, 0.343] |
| recoverable error (c > e) | 0.078 [0.020, 0.157] | 0.104 [0.045, 0.179] |
| first error is the decisive step | 0.529 [0.353, 0.706], n = 34, chance 0.184 | 0.344 [0.188, 0.531], n = 32, chance 0.238 |
| mean drop at the first error: fatal / recovered | +0.238 / −0.099 | +0.140 / +0.208 |
| clean noise floor | +0.081 | +0.208 |
| backtrack: restart / before error / before decisive | 0.438 / 0.382 / 0.500 | 0.621 / 0.406 / 0.566 |
| decisive step minus first error | **+0.118 [+0.033, +0.202]** | **+0.160 [+0.051, +0.277]** |
| event study, fatal: v at r = −1, 0, +1 | 0.270, 0.032, 0.019 | 0.193, 0.053, 0.024 |

Run A and Run B agree for Qwen2.5-1.5B (decisive minus first error +0.121 vs +0.118). For Qwen2-1.5B
the annotated first error is the decisive step barely more often than chance.

**Run B, Qwen2.5-Math-7B-Instruct (finished 2026-10-24 22:25 IST; read from the version log).**
TP = 2, m = 12, max 1,024 new tokens, clean traces capped at 60. All 2,646 prefixes of the 314
selected traces finished after 17,688 s (log: "2646/2646 prefixes ... done"); the vLLM process then
failed to exit and was stopped by the 8.5-hour `timeout` (exit 124), which did not affect the data.
The 3 fatal traces not run are the ones without a gold answer.

| Quantity | Qwen2.5-Math-7B |
|---|---|
| fatal: capability-limited / latent / at error / recoverable | 0.657 / 0.063 / 0.179 / 0.101 |
| first error is the decisive step | 0.423 [0.310, 0.535], n = 71, chance 0.153 |
| mean drop at the first error: fatal / recovered | +0.109 / −0.215 |
| clean noise floor | +0.078 |
| backtrack: restart / before error / before decisive / oracle | 0.538 / 0.432 / 0.561 / 0.641 |
| decisive step minus first error | **+0.129 [+0.061, +0.200]** |
| mean continuation tokens: restart / before error / before decisive | 633 / 469 / 458 |
| event study, fatal: v at r = −1, 0, +1 | 0.175, 0.066, 0.044 |
| event study, recovered | 0.374, 0.589, 0.670 |

## Final analysis (cite these)

Re-analysis of the downloaded archive with the final stage-13 code
(`python scripts/13_commitment_analysis.py --rollouts results/commitment/<gen>_s*.jsonl --out results/commitment/tables_<gen>`).
Two changes against the run log: (1) a trace enters the backtracking pool only if half A of its
continuations says it is not capability-limited, and (2) the A/B split is repeated 50 times and the
pool means are averaged, with CIs from a trace bootstrap over all splits. On the planted-effect mock
(first error decisive by construction) the estimator gives decisive minus first error +0.001, so it
does not manufacture a gap. The "first error is decisive" share is unchanged; a position-matched null
(each trace gets the relative error positions of the other traces) replaces the 1/n chance rate.

| Quantity | Qwen2.5-1.5B | Qwen2-1.5B | Qwen2.5-Math-7B |
|---|---|---|---|
| traces clean / fatal / recovered | 53 / 51 / 12 | 15 / 67 / 6 | 60 / 207 / 47 |
| fatal: capability-limited (δ = 1/8) | 0.333 | 0.522 | 0.657 |
| fatal: latent / at error / recoverable | 0.216 / 0.373 / 0.078 | 0.134 / 0.239 / 0.104 | 0.063 / 0.179 / 0.101 |
| first error is the decisive step | 0.529 [0.353, 0.706], n = 34 | 0.344 [0.188, 0.531], n = 32 | 0.423 [0.310, 0.535], n = 71 |
| position-matched null | 0.171 | 0.214 | 0.159 |
| same, without last-step errors | 0.515 (n = 33) | 0.296 (n = 27) | 0.418 (n = 67) |
| backtrack pool (mean over splits) | 31.7 | 28.5 | 75.9 |
| success: restart | 0.481 [0.415, 0.551] | 0.621 [0.511, 0.722] | 0.470 [0.400, 0.538] |
| success: before first error | 0.375 [0.277, 0.478] | 0.406 [0.281, 0.533] | 0.402 [0.336, 0.471] |
| success: before decisive step | 0.493 [0.421, 0.566] | 0.601 [0.502, 0.695] | 0.511 [0.442, 0.573] |
| success: best prefix (oracle) | 0.574 | 0.688 | 0.622 |
| **restart − first error** | **+0.106 [+0.027, +0.187]** | **+0.215 [+0.095, +0.339]** | **+0.068 [+0.007, +0.133]** |
| **decisive − first error** | **+0.117 [+0.059, +0.183]** | **+0.196 [+0.082, +0.316]** | **+0.108 [+0.051, +0.165]** |
| decisive − restart | +0.011 [−0.035, +0.061] | −0.020 [−0.057, +0.020] | +0.040 [−0.008, +0.089] |
| continuation tokens: restart / before error / before decisive | 374 / 253 / 308 | 203 / 172 / 173 | 650 / 477 / 463 |
| decisive − first error, all fatal traces (no selection) | +0.087 [+0.045, +0.133] | +0.096 [+0.045, +0.154] | +0.055 [+0.030, +0.082] |
| event study, fatal: v at r = −1 → 0 | 0.270 → 0.032 | 0.193 → 0.053 | 0.175 → 0.066 |

Reading:

- H1 (first error = decisive step): holds for about half of the fatal traces that are not
  capability-limited, 2.5-3x the null for the two Qwen2.5 models; for Qwen2-1.5B the interval includes
  the null.
- H2 / H3: latent commitment is 18-32% and recoverable errors 12-30% of non-capability-limited
  fatal traces.
- H4: decisive-step backtracking beats first-error backtracking in all three models (pre-registered
  test passes). Unplanned but robust: **restarting from scratch also beats first-error backtracking in
  all three models**, and decisive-step backtracking only ties with restarting while using 15-29% fewer
  tokens. Because v_k is a martingale under the generating policy, E[v_k | fail] = E[v_k(1 − v_k)] / P(fail)
  is non-increasing for fixed k or stopping times, so restart is a strong baseline for any target not
  chosen with information about where success is lost (the step before the error is not a stopping
  time, so this is an explanation, not a proof of the result).

## Run C (confirmatory and robustness run, preregistered)

Plan: [`PREREGISTRATION_RUN_C.md`](PREREGISTRATION_RUN_C.md), committed in `459ff22` before any Run C
data existed. Driver: `scripts/kaggle_run_c.sh` (Kaggle, 2x T4).

- Smoke test 1 (Kaggle version 2): step signals and PRM scoring worked; vLLM jobs after the first
  failed because `os._exit` left vLLM engine processes holding GPU memory. Fixed in `f47b3f9`.
- Smoke test 2 (version 3): skipped the setup step by mistake (no data, no vLLM); no result.
- Smoke test 3 (version 4): every part finished (probes for all four generators, three replicate
  batches for both 1.5B models, Llama-3.1-8B rollouts).
- Full run (version 5, Kaggle script version 361921902) started from `17c83cc` on 2026-10-24 at
  about 19:55 UTC.
