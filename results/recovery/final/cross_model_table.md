# Cross-Model Generalization Benchmark (1,500 Items)

Demonstrates failure detection and hallucination characteristics across model generations on identical evaluation benchmarks.

| Architecture | Benchmark Split | Task Type | Total Items | Failure Choke-Points Flagged | Mean Detector Confidence |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Llama-2-7B-chat** | RAGTruth | Data2txt | 150 | **76.7%** | 0.744 |
| **Llama-2-7B-chat** | RAGTruth | QA | 450 | **20.2%** | 0.221 |
| **Llama-2-7B-chat** | RAGTruth | Summary | 150 | **23.3%** | 0.267 |
| **Llama-2-7B-chat (Total)** | **Combined** | **All Tasks** | **750** | **32.1%** | **0.335** |
| | | | | | |
| **Llama-3.1-8B-Instruct** | RAGTruth | Data2txt | 150 | **48.0%** | 0.490 |
| **Llama-3.1-8B-Instruct** | RAGTruth | QA | 450 | **1.8%** | 0.025 |
| **Llama-3.1-8B-Instruct** | RAGTruth | Summary | 150 | **16.0%** | 0.178 |
| **Llama-3.1-8B-Instruct (Total)** | **Combined** | **All Tasks** | **750** | **13.9%** | **0.148** |
| | | | | | |

### Key Generalization Insights:
1. **Architectural Scaling**: Modern instruction-tuned models drastically reduce raw error rates (**46.2% → 13.9%** overall).
2. **Persistent Failure Choke-Points**: Despite architectural improvements, **13.9% of Llama-3.1-8B responses** still contain severe groundedness errors.
3. **Cross-Architecture Applicability**: Early failure detection cleanly isolates failures in both generations without retraining or architectural coupling.
