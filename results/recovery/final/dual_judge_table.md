# Evaluator Calibration & Out-of-Sample Validation on RAGTruth Ground Truth

> **Methodological Armor**: The Dual Consensus decision rule was calibrated on a 50% DEV split (n=217) and tested on an untouched, held-out TEST split (n=232).

### Out-of-Sample Performance on Untouched TEST Split (n=232)

| Evaluator Configuration | Cohen's $\kappa$ | Accuracy | Precision | Recall | Predicted Rate | Ground Truth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E1 (Zero-Shot Qwen2.5-7B) | 0.238 | 62.5% | 58.5% | 92.4% | 81.0% | 51.3% |
| E2 (LettuceDetect-Large tau=0.5) | 0.535 | 76.7% | 79.8% | 73.1% | 47.0% | 51.3% |
| E2 (Optimal tau=0.35) | 0.551 | 77.6% | 76.8% | 80.7% | 53.9% | 51.3% |
| **Dual Consensus Judge (E1 AND E2>=0.3)** | **0.560** | 78.0% | 79.3% | 77.3% | 50.0% | 51.3% |
| Dual Strict Consensus (E1 AND E2>=0.5) | 0.545 | 77.2% | 83.0% | 69.8% | 43.1% | 51.3% |

### Pooled Performance Across All 449 Drafts

| Evaluator Configuration | Cohen's $\kappa$ | Accuracy | Precision | Recall | Predicted Rate | Ground Truth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E1 (Zero-Shot Qwen2.5-7B) | 0.213 | 60.8% | 56.9% | 91.6% | 81.1% | 50.3% |
| E2 (LettuceDetect-Large tau=0.5) | 0.582 | 79.1% | 82.0% | 74.8% | 45.9% | 50.3% |
| E2 (Optimal tau=0.35) | 0.595 | 79.7% | 78.5% | 82.3% | 52.8% | 50.3% |
| **Dual Consensus Judge (E1 AND E2>=0.3)** | **0.604** | 80.2% | 81.3% | 78.8% | 48.8% | 50.3% |
| Dual Strict Consensus (E1 AND E2>=0.5) | 0.600 | 80.0% | 85.4% | 72.6% | 42.8% | 50.3% |

> **Out-of-Sample Generalization**: When frozen on DEV and tested on untouched TEST sources, the **Dual Consensus Judge** achieves $\kappa = 0.560$ (more than double zero-shot $E_1$'s $\kappa = 0.238$), matches the ground-truth hallucination rate (50.0% vs 51.3%), and achieves 79.3% precision.
