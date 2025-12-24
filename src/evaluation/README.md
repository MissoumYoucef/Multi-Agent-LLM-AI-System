# 📊 Evaluation Framework

This module provides tools for **Quantitative and Qualitative Analysis** of the system's performance, ensuring the quality of agent responses and RAG retrieval.

## 🏗 Evaluation Flow

The framework supports both offline analysis (batch testing) and online monitoring (continuous evaluation) to track key performance indicators.

```mermaid
graph LR
    Response[Agent Response] --> Evaluator[Evaluation Engine]
    GroundTruth[Ground Truth] --> Evaluator
    
    subgraph "Metrics Suite"
        Evaluator --> Function[Functional Correctness]
        Evaluator --> Lexical[ROUGE / BLEU]
        Evaluator --> Semant[Semantic Similarity]
        Evaluator --> Judge[LLM-as-a-Judge]
    end
    
    Metrics_Suite --> Report[Quality Report]
    
    style Evaluator fill:#f9f,stroke:#333,stroke-width:2px
```

## 🧩 Components

| Component | File | Description |
|-----------|------|-------------|
| **Metrics Library** | `metrics.py` | Implementation of core evaluation metrics, including ROUGE scores, exact matching, and semantic similarity. |
| **Continuous Eval** | `continuous_eval.py` | System for running evaluations on a schedule or sampled from live traffic to detect drift or regression. |

## 🚀 Key Features

-   **Multi-Dimensional Scoring:** Evaluates answers not just for correctness, but for style, safety, and relevance.
-   **LLM Judge:** Uses a powerful LLM to grade the output of smaller/specialized agents.
-   **Regression Testing:** Can be used in CI/CD pipelines to ensure code changes don't degrade answer quality.
