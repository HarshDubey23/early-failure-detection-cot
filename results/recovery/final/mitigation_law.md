# Detect-restart-abstain: the law and the data (exploratory)

Human-labelled RAGTruth drafts: n = 448; resampled continuations judged by E1: n = 7125.

| Quantity | Estimate [95% CI] |
|---|---|
| base hallucination rate h (people) | 50.2% [45.5, 54.9] |
| detector D: false-alarm rate alpha | 20.6% [15.4, 26.1] |
| detector D: miss rate beta | 28.4% [22.5, 34.7] |
| **predicted hallucination among delivered answers, pi** | **26.6% [21.0, 32.2]** |
| predicted reduction, h - pi | 23.7% [19.4, 28.2] |
| check: E1-implied human-scale rate, D-passed continuations | 24.8% [-5.0, 43.5] |
| control: E1-implied rate, D-flagged continuations | 116.4% [99.9, 144.9] |

Cost of each RAG mitigation (RAGTruth, 450 sources; tokens per item, all calls counted):

| Policy | tokens | cost vs base | E1 | E2 | coverage |
|---|---|---|---|---|---|
| Base model (no mitigation) | 948 | 1.00x | 0.809 | 0.460 | 1.000 |
| Detect, delete flagged sentences | 968 | 1.02x | 0.796 | 0.131 | 0.998 |
| Detect, repair claims | 1,624 | 1.71x | 0.800 | 0.189 | 0.998 |
| Detect, restart, abstain (DRA) | 1,293 | 1.36x | 0.767 | 0.129 | 0.989 |
| Detect, backtrack, abstain | 1,279 | 1.35x | 0.759 | 0.134 | 0.971 |
| CARE (choke-point controller) | 1,677 | 1.77x | 0.762 | 0.153 | 0.965 |
| Detect, abstain on QA | 968 | 1.02x | 0.673 | 0.104 | 0.871 |
| Chain-of-Verification (n=192) | 3,558 | 3.75x | 0.974 | 0.724 | 1.000 |

Failed maths solutions solved by plain restart (P(solved), FLOP-matched budget in restart attempts):

| Generator | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.266 | 0.376 | 0.492 | 0.606 |
| Qwen2-1.5B-Instruct | 0.302 | 0.401 | 0.514 | 0.616 |
| Qwen2.5-1.5B-Instruct | 0.360 | 0.512 | 0.657 | 0.771 |
| Qwen2.5-Math-7B-Instruct | 0.206 | 0.328 | 0.437 | 0.556 |
