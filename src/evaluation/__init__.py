"""
Evaluation module for quality monitoring.

Provides continuous evaluation pipeline for monitoring RAG system quality
with shadow evaluation, A/B testing, and quality tracking.
"""

from .continuous_eval import (
    ContinuousEvaluator,
    create_continuous_evaluator,
    EvalResult,
    QualityTrend
)
from .metrics import EvaluationMetrics

__all__ = [
    # Continuous Evaluation
    "ContinuousEvaluator",
    "create_continuous_evaluator",
    "EvalResult",
    "QualityTrend",
    # Metrics
    "EvaluationMetrics",
]
