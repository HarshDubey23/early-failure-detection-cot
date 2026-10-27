# Checks of the restart-loop law (Proposition 3), exploratory

## 1. More retries buy coverage, not faithfulness

RAGTruth, 450 sources, same items at budget 1 and budget 4 (paired). Delivered = answered outputs.

| Loop | coverage b=1 -> b=4 | change | delivered hallucination (E1) b=1 -> b=4 | change | (E2) b=1 -> b=4 | change |
|---|---|---|---|---|---|---|
| restart_fb | 0.972 -> 0.996 | +0.024 [+0.015, +0.034] | 0.761 -> 0.776 | +0.015 [+0.009, +0.022] | 0.125 -> 0.131 | +0.006 [-0.001, +0.014] |
| backtrack_fb | 0.960 -> 0.975 | +0.015 [+0.008, +0.023] | 0.772 -> 0.782 | +0.010 [+0.005, +0.016] | 0.134 -> 0.138 | +0.004 [-0.002, +0.011] |

## 2. Operating curve predicted from 449 human-labelled drafts (base rate 0.503)

| Gate | tau | predicted hallucination of delivered answers | flag rate q | draws per answered input | coverage with 2 draws |
|---|---|---|---|---|---|
| D (LettuceDetect-base) | 0.1 | 0.213 [0.151, 0.277] | 0.624 | 2.66 | 0.611 |
| D (LettuceDetect-base) | 0.2 | 0.245 [0.187, 0.302] | 0.546 | 2.20 | 0.702 |
| D (LettuceDetect-base) | 0.3 | 0.255 [0.200, 0.311] | 0.510 | 2.04 | 0.740 |
| D (LettuceDetect-base) | 0.4 | 0.259 [0.201, 0.314] | 0.492 | 1.97 | 0.758 |
| D (LettuceDetect-base) | 0.5 | 0.266 [0.209, 0.320] | 0.463 | 1.86 | 0.785 |
| D (LettuceDetect-base) | 0.6 | 0.271 [0.216, 0.325] | 0.425 | 1.74 | 0.819 |
| D (LettuceDetect-base) | 0.7 | 0.272 [0.219, 0.326] | 0.403 | 1.68 | 0.837 |
| D (LettuceDetect-base) | 0.8 | 0.285 [0.232, 0.339] | 0.383 | 1.62 | 0.853 |
| D (LettuceDetect-base) | 0.9 | 0.316 [0.269, 0.366] | 0.339 | 1.51 | 0.885 |
| E2 (LettuceDetect-large) | 0.1 | 0.145 [0.091, 0.214] | 0.708 | 3.43 | 0.498 |
| E2 (LettuceDetect-large) | 0.2 | 0.180 [0.123, 0.233] | 0.592 | 2.45 | 0.649 |
| E2 (LettuceDetect-large) | 0.3 | 0.188 [0.133, 0.245] | 0.539 | 2.17 | 0.710 |
| E2 (LettuceDetect-large) | 0.4 | 0.211 [0.158, 0.264] | 0.494 | 1.98 | 0.756 |
| E2 (LettuceDetect-large) | 0.5 | 0.235 [0.185, 0.291] | 0.459 | 1.85 | 0.790 |
| E2 (LettuceDetect-large) | 0.6 | 0.253 [0.206, 0.308] | 0.419 | 1.72 | 0.825 |
| E2 (LettuceDetect-large) | 0.7 | 0.274 [0.216, 0.328] | 0.374 | 1.60 | 0.860 |
| E2 (LettuceDetect-large) | 0.8 | 0.298 [0.249, 0.354] | 0.334 | 1.50 | 0.888 |
| E2 (LettuceDetect-large) | 0.9 | 0.343 [0.295, 0.396] | 0.267 | 1.36 | 0.929 |
| D or E2 flags | 0.1 | 0.127 [0.071, 0.191] | 0.773 | 4.40 | 0.403 |
| D or E2 flags | 0.2 | 0.150 [0.090, 0.198] | 0.673 | 3.05 | 0.548 |
| D or E2 flags | 0.3 | 0.167 [0.116, 0.222] | 0.612 | 2.58 | 0.625 |
| D or E2 flags | 0.4 | 0.190 [0.137, 0.255] | 0.579 | 2.38 | 0.665 |
| D or E2 flags | 0.5 | 0.216 [0.156, 0.274] | 0.537 | 2.16 | 0.712 |
| D or E2 flags | 0.6 | 0.240 [0.189, 0.291] | 0.481 | 1.93 | 0.769 |
| D or E2 flags | 0.7 | 0.242 [0.190, 0.294] | 0.448 | 1.81 | 0.800 |
| D or E2 flags | 0.8 | 0.262 [0.213, 0.312] | 0.414 | 1.71 | 0.828 |
| D or E2 flags | 0.9 | 0.292 [0.239, 0.351] | 0.367 | 1.58 | 0.865 |

## 3. Each loop against the base model, same items (budget 2; exploratory except CARE under E1, which is H0)

| Loop | change in hallucination, E1 judge | change, E2 |
|---|---|---|
| restart_fb | -0.042 [-0.063, -0.022] | -0.331 [-0.373, -0.291] |
| care | -0.047 [-0.070, -0.024] | -0.307 [-0.351, -0.264] |
