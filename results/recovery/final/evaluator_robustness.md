# Part 3 under five evaluators (RAGTruth, TEST half of the sources)

E1-calibrated flags an output when the judge calls more than 25% of its sentences unsupported; the threshold was chosen on the 216 DEV drafts (grid fixed in advance) and never reads a LettuceDetect score. Human-agreement kappa on the 232 TEST drafts: E1 (any sentence): 0.238; E1-calibrated (share > 0.25): 0.189; E2 (LettuceDetect-large >= 0.5): 0.535; dual (E1 and E2 >= 0.3; tau chosen on all drafts): 0.560; D (system detector >= 0.5): 0.476.

E2, the dual rule and D share LettuceDetect's training data with the detector the system resamples against, so they are expected to favour detector-gated policies.

Hallucination rate (lower is better), and difference from restart_fb at budget 2 with 95% bootstrap CI:

| Policy | coverage | tokens | E1 judge | E1 calibrated | E2 | dual rule | D (system's own detector) |
|---|---|---|---|---|---|---|---|
| base@0 (n=233) | 1.000 | 980 | 0.807 (+0.017 [-0.013, +0.047]) | 0.511 (+0.013 [-0.043, +0.069]) | 0.472 (+0.326 [+0.266, +0.386]) | 0.498 (+0.309 [+0.249, +0.373]) | 0.446 (+0.438 [+0.373, +0.502]) |
| delete@0 (n=233) | 0.996 | 1,001 | 0.790 (+0.000 [-0.030, +0.030]) | 0.519 (+0.021 [-0.030, +0.077]) | 0.172 (+0.026 [-0.013, +0.064]) | 0.223 (+0.034 [-0.009, +0.082]) | 0.077 (+0.069 [+0.039, +0.103]) |
| repairdel@0 (n=233) | 0.996 | 1,624 | 0.798 (+0.009 [-0.021, +0.039]) | 0.498 (+0.000 [-0.060, +0.060]) | 0.210 (+0.064 [+0.021, +0.112]) | 0.288 (+0.099 [+0.052, +0.150]) | 0.069 (+0.060 [+0.030, +0.094]) |
| abstain_flag@0 (n=233) | 0.884 | 1,001 | 0.678 (-0.112 [-0.159, -0.069]) | 0.421 (-0.077 [-0.133, -0.017]) | 0.150 (+0.004 [-0.030, +0.039]) | 0.197 (+0.009 [-0.034, +0.052]) | 0.056 (+0.047 [+0.021, +0.077]) |
| restart_fb@2 (n=233) | 0.996 | 1,320 | 0.790 | 0.498 | 0.146 | 0.189 | 0.009 |
| backtrack_fb@2 (n=233) | 0.996 | 1,279 | 0.803 (+0.013 [-0.013, +0.043]) | 0.528 (+0.030 [-0.026, +0.082]) | 0.193 (+0.047 [+0.009, +0.086]) | 0.262 (+0.073 [+0.026, +0.120]) | 0.004 (-0.004 [-0.013, +0.000]) |
| care@2 (n=233) | 0.971 | 1,708 | 0.777 (-0.013 [-0.043, +0.017]) | 0.468 (-0.030 [-0.082, +0.021]) | 0.193 (+0.047 [+0.009, +0.086]) | 0.249 (+0.060 [+0.017, +0.103]) | 0.021 (+0.013 [-0.004, +0.030]) |
| care@restart@2 (n=233) | 0.968 | 1,710 | 0.773 (-0.017 [-0.047, +0.009]) | 0.451 (-0.047 [-0.099, +0.000]) | 0.185 (+0.039 [+0.008, +0.073]) | 0.236 (+0.047 [+0.013, +0.086]) | 0.021 (+0.013 [-0.004, +0.034]) |
| cove_flag_clean@0 (n=98) | 1.000 | – | 0.714 (+0.031 [-0.031, +0.092]) | 0.439 (+0.041 [-0.051, +0.133]) | 0.418 (+0.265 [+0.184, +0.357]) | 0.429 (+0.235 [+0.153, +0.317]) | 0.306 (+0.296 [+0.214, +0.388]) |

Sign of the difference from restart_fb under each evaluator (+ worse than restart, − better, 0 no difference detected):

| Policy | E1 judge | E1 calibrated | E2 | dual rule | D (system's own detector) | same under all? |
|---|---|---|---|---|---|---|
| base@0 | 0 | 0 | + | + | + | no |
| delete@0 | 0 | 0 | 0 | 0 | + | no |
| repairdel@0 | 0 | 0 | + | + | + | no |
| abstain_flag@0 | - | - | 0 | 0 | + | no |
| backtrack_fb@2 | 0 | 0 | + | + | 0 | no |
| care@2 | 0 | 0 | + | + | 0 | no |
| care@restart@2 | 0 | 0 | + | + | 0 | no |
| cove_flag_clean@0 | 0 | 0 | + | + | + | no |
