# ChokeBench predictive validity (stage 54, exploratory)

232 open failed traces, 1821 (trace, localiser) pairs. Half A of each prefix's continuations labels the decisive step, half B scores the gain of resampling from the named step (50 splits); trace fixed effects.

| Model | gain per attempt when the named step is the decisive step | ... when it is the human first error |
|---|---|---|
| decisive step only | +0.404 [+0.368, +0.442] | |
| human first error only | | +0.153 [+0.109, +0.196] |
| both | +0.396 [+0.354, +0.442] | +0.014 [-0.026, +0.050] |

Difference (decisive minus human, joint model): +0.382 [+0.313, +0.458].

## Held-out model: do benchmark scores on three generators rank localisers by gain on the fourth?

| Held-out generator | localisers | Spearman, ChokeBench score | Spearman, ProcessBench-style score |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 7 | +0.32 | +0.18 |
| Qwen2-1.5B-Instruct | 7 | +0.61 | +0.32 |
| Qwen2.5-Math-7B-Instruct | 7 | +0.32 | +0.04 |
| Llama-3.1-8B-Instruct | 7 | +0.82 | +0.36 |
| **mean** | | **+0.52** | **+0.22** |

Mean difference +0.29 [+0.08, +0.46] (trace bootstrap within generator, 500 resamples). Seven localisers per generator, so each single rho is noisy.

Reading: the coefficients compare named steps within a trace, not a named step with restart. Naming the decisive step avoids most of the loss other steps incur; it does not by itself beat restart (stage 49).

exploratory; localiser-level ChokeBench accuracies and gains (stage 49) had been seen before this analysis.
