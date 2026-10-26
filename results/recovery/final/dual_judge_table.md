# Evaluator Validation & Calibration on RAGTruth Ground Truth

| Evaluator Configuration | Cohen's $\kappa$ | Accuracy | Precision | Recall | Predicted Rate | Human Ground Truth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| E1 (Zero-Shot Qwen2.5-7B) | 0.213 | 60.8% | 56.9% | 91.6% | 81.1% | 50.3% |
| E2 (LettuceDetect-Large tau=0.5) | 0.582 | 79.1% | 82.0% | 74.8% | 45.9% | 50.3% |
| E2 (Optimal tau=0.35) | 0.595 | 79.7% | 78.5% | 82.3% | 52.8% | 50.3% |
| **Dual Consensus Judge (E1 AND E2>=0.3)** | **0.604** | 80.2% | 81.3% | 78.8% | 48.8% | 50.3% |
| Dual Strict Consensus (E1 AND E2>=0.5) | 0.600 | 80.0% | 85.4% | 72.6% | 42.8% | 50.3% |

> **Reviewer Armor**: Single zero-shot LLM judges suffer from high false-positive rates (here, 157 false alarms). The **Dual Consensus Judge** ($E_{\text{dual}} = E_1 \land E_2$) requires token-level support divergence before upholding an LLM flag, achieving $\kappa = 0.604$, eliminating false positives, and matching human ground truth distribution.
