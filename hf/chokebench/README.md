---
license: mit
language:
- en
pretty_name: ChokeBench v1
size_categories:
- n<1K
source_datasets:
- Qwen/ProcessBench
tags:
- reasoning
- chain-of-thought
- process-reward-model
- verifier
- llm-evaluation
- benchmark
- math
configs:
- config_name: default
  data_files: chokebench_v1.jsonl
---

# ChokeBench v1

**Does your verifier find the step where a reasoning failure is actually decided?**

ProcessBench grades a verifier on finding the *human-labelled first error*. ChokeBench also asks it to find the
**choke point**: the step where the generator's own chance of success collapses. Ground truth comes from
resampling the generator 64 times from every prefix of every failed trace, so it measures what each step is
worth as a place to restart from, not what a human annotator thought was wrong.

Headline finding: a process reward model finds **73%** of human first errors but only **45%** of choke points,
and the human label itself is the choke point only about **half** the time.

- Paper and code: [github.com/HarshDubey23/early-failure-detection-cot](https://github.com/HarshDubey23/early-failure-detection-cot)
- Authors: Harsh Dubey, Harshit Saini, Prashant Jain, Himanshu Kumar, Gaurav Kaushal, Rohit Chaudhary
  (Dept. of CSE (AI&ML), KIET Group of Institutions)

## What is in it

469 failed maths traces from [ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench), each run on one
of four generators: Qwen2.5-1.5B-Instruct, Qwen2-1.5B-Instruct, Qwen2.5-Math-7B-Instruct and Llama-3.1-8B-Instruct.
Scoring uses the open ones (not capability-limited, prefix-0 success rate above 1/8).

| Field | Meaning |
|---|---|
| `trace_id` | ProcessBench item, e.g. `processbench:gsm8k:gsm8k-156` |
| `generator` | model that produced the rollouts |
| `n_steps` | steps in the trace |
| `human_first_error` | 1-based step of the expert-labelled first error |
| `successes`, `continuations` | per prefix k = 0..n-1: correct continuations out of those sampled |
| `mean_continuation_tokens` | per prefix: mean tokens generated from there |
| `prm` | per step: Qwen2.5-Math-PRM-7B score |
| `capability_limited` | true if the generator rarely solves the problem even from scratch |

No problem text is included; join on `trace_id` with ProcessBench if you need it.

## Grade your own localiser

Name one step per trace and score it with [`chokebench_score.py`](chokebench_score.py) (NumPy only, runs in seconds):

```bash
python chokebench_score.py chokebench_v1.jsonl submission.json   # {"<trace_id>|<generator>": step, ...}
```

It reports how often you hit the human first error, the posterior probability that you hit the choke point
(Jeffreys posterior on every prefix's success rate, so sampling noise is not counted as a miss), and the gain of
resampling from your step versus a plain restart. This scorer skips the paper's sample-split weighting, so its
numbers sit within about 0.01 of the official ones below.

## Official results (paper protocol, n = 230, 95% bootstrap CIs)

| Localiser | finds human first error | finds choke point | gain vs restart, per attempt |
|---|---|---|---|
| Human first error | 1.00 | 0.51 [0.45, 0.56] | -0.066 |
| PRM: largest drop | 0.73 | 0.45 [0.39, 0.51] | -0.103 |
| PRM: first step < 0.5 | 0.68 | 0.38 [0.32, 0.44] | -0.077 |
| PRM: minimum | 0.61 | 0.39 [0.34, 0.45] | -0.158 |
| Learned (nested CV, held-out model) | 0.68 | 0.39 [0.33, 0.44] | -0.085 |
| Random step | 0.18 | 0.18 | -0.224 |
| First step (= restart) | 0.14 | 0.28 | 0 |

Even an oracle that names the choke point for free gains only +0.028 over restart at a four-attempt budget.

## Licence

MIT. Derived from ProcessBench (Apache-2.0); the generator and PRM outputs follow their model licences
(Qwen, Llama 3.1 Community License).

## Citation

```bibtex
@misc{dubey2026chokepoints,
  title        = {Errors Are Not Choke Points: Counterfactual Localisation of {LLM} Reasoning
                  Failures, and Why Restart Is Hard to Beat},
  author       = {Dubey, Harsh and Saini, Harshit and Jain, Prashant and Kumar, Himanshu and
                  Kaushal, Gaurav and Chaudhary, Rohit},
  year         = {2026},
  howpublished = {\url{https://github.com/HarshDubey23/early-failure-detection-cot}}
}
```
