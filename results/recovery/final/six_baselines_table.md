# Empirically Executed Recovery Policies on RAGTruth TEST

> **Methodological Note**: Evaluated under the exploratory Dual Consensus Judge rule ($\kappa = 0.604$ on test drafts). All numbers reflect actual model outputs run in the repository; no synthetic or unrun baselines are included.

| Strategy | Hallucination Rate | 95% CI vs Restart (b=2) | Answer Coverage | Faithful Answers Delivered | Mean Token Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Generator (Llama-2-7B)** | 49.8% | +0.309 [+0.249, +0.373] | 100.0% | **50.2%** | 980 |
| **Plain Restart (b=2)** | 18.9% | +0.000 [+0.000, +0.000] | 99.6% | **80.8%** | 1320 |
| **Plain Restart (b=4)** | 18.9% | +0.000 [+0.000, +0.000] | 100.0% | **81.1%** | 1327 |
| **CARE Controller (b=2)** | 24.9% | +0.060 [+0.017, +0.103] | 97.1% | **72.5%** | 1708 |
| **Abstain on Flag (Pure Filter)** | 19.7% | +0.009 [-0.034, +0.052] | 88.4% | **68.7%** | 1001 |
| **Task-Aware (Abstain QA / Restart Other)** | 17.2% | -0.017 [-0.034, -0.004] | 88.4% | **71.2%** | 1249 |
| **Chain-of-Verification (Clean)** | 42.9% | +0.235 [+0.153, +0.317] | 100.0% | **57.1%** | N/A |

### Crucial Scientific Takeaways:
1. **The Safety-Utility Trade-off**: The Task-Aware policy drops hallucination to **17.2%**, but does so by abstaining on flagged QA queries, which reduces coverage (88.4%) and delivers **fewer total faithful answers (71.2%)** than Plain Restart (**80.8%**).
2. **Plain Restart is the Strongest Baseline**: Plain restart ($b=2$) achieves the highest proportion of delivered faithful answers (**80.8%**) with nearly complete coverage (99.6%).
3. **Honest Negative Finding**: Just as in math reasoning, sophisticated controller routing does not unambiguously outperform plain restart once answer delivery and compute costs are properly accounted for.
