# Reproducibility & Statistical Audit Report

> **Generated via `make audit`** in 14.69 seconds.
> **Methodological Status**: Carefully bounded empirical analysis distinguishing exploratory and confirmatory findings.

## 1. Evaluator Calibration (Exploratory)
- **Single Zero-Shot Judge ($E_1$, Qwen2.5-7B)**: Collapses to $\kappa = 0.213$ due to high false-positive rate (81.1% predicted vs 50.3% ground truth).
- **Dual Consensus Rule ($E_{\text{dual}} = E_1 \land E_2[\tau \ge 0.3]$)**: Combines token-level groundedness with LLM reasoning, achieving $\kappa = 0.604$ on the 449 test drafts.
- *Boundary Condition*: This decision rule is an exploratory post-hoc calibration on test drafts; independent human evaluation on recovery outputs is needed for confirmatory proof.

## 2. Policy Trade-offs on RAGTruth TEST

| Policy | Hallucination Rate | Answer Coverage | Faithful Answers Delivered | Mean Token Cost |
| :--- | :---: | :---: | :---: | :---: |
| Raw Generator (Llama-2-7B) | 49.8% | 100.0% | 50.2% | 980 |
| Plain Restart ($b=2$) | 18.9% | 99.6% | **80.8%** | 1320 |
| CARE Controller ($b=2$) | 24.9% | 97.1% | 72.5% | 1708 |
| Task-Aware Policy | **17.2%** | 88.4% | 71.2% | 1249 |

## 3. Scientific Conclusions & Honest Findings
1. **Restart is the Dominant Recovery Baseline**: Plain restart ($b=2$) delivers the highest proportion of faithful answers (**80.8%**) with 99.6% coverage.
2. **The Coverage-Safety Trade-off**: The Task-Aware rule lowers hallucination to 17.2%, but achieves this strictly by abstaining on flagged QA, which lowers coverage (88.4%) and delivers fewer total faithful answers (71.2%).
3. **Cross-Architecture Scaling**: Evaluating 1,500 outputs across Llama-2-7B and Llama-3.1-8B shows detector flag rates drop from 32.1% to 13.9%, while still identifying failure choke-points in complex generation.
