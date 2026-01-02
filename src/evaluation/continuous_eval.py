"""
Continuous Evaluation module.

Provides continuous evaluation pipeline for monitoring RAG system quality
with shadow evaluation, A/B testing, and quality tracking.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    query: str
    response: str
    quality_score: float
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    context: str = ""
    session_id: str = "default"


@dataclass
class QualityTrend:
    """Quality trend over a time window."""
    window_days: int
    sample_count: int
    avg_quality: float
    min_quality: float
    max_quality: float
    trend: str  # "improving", "degrading", "stable"
    change_pct: float


class ContinuousEvaluator:
    """
    Continuous evaluation pipeline for production monitoring.

    Features:
    - Shadow evaluation on production traffic
    - Quality score tracking over time
    - A/B testing support
    - Threshold-based alerts
    """

    def __init__(
        self,
        quality_threshold: float = 0.7,
        window_size: int = 10000,
        enable_shadow_eval: bool = True
    ):
        """
        Initialize continuous evaluator.

        Args:
            quality_threshold: Minimum acceptable quality score.
            window_size: Number of evaluations to keep in history.
            enable_shadow_eval: Enable shadow evaluation mode.
        """
        self.quality_threshold = quality_threshold
        self.window_size = window_size
        self.enable_shadow_eval = enable_shadow_eval

        self._eval_history: deque = deque(maxlen=window_size)
        self._ab_results: Dict[str, List[EvalResult]] = {}

        # Lazy load evaluation metrics
        self._metrics = None

        logger.info(f"ContinuousEvaluator initialized: threshold={quality_threshold}")

    @property
    def metrics(self):
        """Lazy load evaluation metrics."""
        if self._metrics is None:
            from .metrics import EvaluationMetrics
            self._metrics = EvaluationMetrics()
        return self._metrics

    def evaluate_response(
        self,
        query: str,
        response: str,
        context: str = "",
        expected_keywords: Optional[List[str]] = None,
        reference: Optional[str] = None,
        session_id: str = "default"
    ) -> EvalResult:
        """
        Evaluate a response quality.

        Args:
            query: The user query.
            response: The generated response.
            context: The retrieval context used.
            expected_keywords: Optional keywords to check.
            reference: Optional reference answer.
            session_id: Session identifier.

        Returns:
            EvalResult with quality assessment.
        """
        metrics_dict = {}
        scores = []

        # Functional correctness if keywords provided
        if expected_keywords:
            fc_score = self.metrics.functional_correctness(response, expected_keywords)
            metrics_dict["functional_correctness"] = fc_score
            scores.append(fc_score)

        # Lexical exactness if reference provided
        if reference:
            lex_score = self.metrics.lexical_exactness(response, reference)
            rouge_score = self.metrics.rouge_l_score(response, reference)
            metrics_dict["lexical_exactness"] = lex_score
            metrics_dict["rouge_l"] = rouge_score
            scores.append(lex_score)
            scores.append(rouge_score)

        # Response quality heuristics
        quality_score = self._compute_quality_heuristics(response, context)
        metrics_dict["heuristic_quality"] = quality_score
        scores.append(quality_score)

        # Overall quality score
        overall_score = sum(scores) / len(scores) if scores else quality_score

        result = EvalResult(
            query=query,
            response=response,
            quality_score=overall_score,
            metrics=metrics_dict,
            context=context,
            session_id=session_id
        )

        self._eval_history.append(result)

        # Check threshold
        if overall_score < self.quality_threshold:
            logger.warning(f"Low quality response: {overall_score:.2f} < {self.quality_threshold}")

        return result

    def _compute_quality_heuristics(
        self,
        response: str,
        context: str
    ) -> float:
        """Compute quality score based on heuristics."""
        score = 1.0

        # Check for empty or very short responses
        if not response or len(response.strip()) < 10:
            return 0.1

        # Check response length relative to context
        if context and len(response) < len(context) * 0.1:
            score -= 0.2

        # Check for uncertainty markers
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "i cannot", "i can't",
            "unclear", "unknown", "not available"
        ]
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                score -= 0.1
                break

        # Check for context utilization
        if context:
            # Simple overlap check
            context_words = set(context.lower().split())
            response_words = set(response_lower.split())
            overlap = len(context_words & response_words)
            context_utilization = overlap / len(context_words) if context_words else 0
            if context_utilization < 0.05:
                score -= 0.2  # Response doesn't use context

        return max(0.0, min(1.0, score))

    def get_quality_trend(
        self,
        window_days: int = 30
    ) -> QualityTrend:
        """
        Get quality trend over a time window.

        Args:
            window_days: Days to analyze.

        Returns:
            QualityTrend with analysis.
        """
        cutoff = time.time() - (window_days * 86400)
        recent = [e for e in self._eval_history if e.timestamp >= cutoff]

        if not recent:
            return QualityTrend(
                window_days=window_days,
                sample_count=0,
                avg_quality=0.0,
                min_quality=0.0,
                max_quality=0.0,
                trend="unknown",
                change_pct=0.0
            )

        scores = [e.quality_score for e in recent]
        avg = sum(scores) / len(scores)

        # Calculate trend by comparing first and second half
        mid = len(scores) // 2
        if mid > 0:
            first_half_avg = sum(scores[:mid]) / mid
            second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
            change = second_half_avg - first_half_avg
            change_pct = (change / first_half_avg * 100) if first_half_avg > 0 else 0

            if change_pct > 5:
                trend = "improving"
            elif change_pct < -5:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            change_pct = 0
            trend = "insufficient_data"

        return QualityTrend(
            window_days=window_days,
            sample_count=len(recent),
            avg_quality=avg,
            min_quality=min(scores),
            max_quality=max(scores),
            trend=trend,
            change_pct=change_pct
        )

    def check_quality_threshold(
        self,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if current quality meets threshold.

        Args:
            threshold: Quality threshold (uses default if not specified).

        Returns:
            True if quality is above threshold.
        """
        threshold = threshold or self.quality_threshold
        trend = self.get_quality_trend(window_days=7)
        return trend.avg_quality >= threshold

    def start_ab_test(
        self,
        test_name: str
    ) -> None:
        """Start an A/B test."""
        self._ab_results[test_name] = []
        logger.info(f"Started A/B test: {test_name}")

    def record_ab_result(
        self,
        test_name: str,
        variant: str,
        result: EvalResult
    ) -> None:
        """Record a result for an A/B test."""
        result.metrics["ab_test"] = test_name
        result.metrics["variant"] = variant
        if test_name in self._ab_results:
            self._ab_results[test_name].append(result)

    def get_ab_results(
        self,
        test_name: str
    ) -> Dict[str, Any]:
        """Get A/B test results."""
        if test_name not in self._ab_results:
            return {"error": "Test not found"}

        results = self._ab_results[test_name]
        by_variant: Dict[str, List[float]] = {}

        for r in results:
            variant = r.metrics.get("variant", "unknown")
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(r.quality_score)

        analysis = {}
        for variant, scores in by_variant.items():
            analysis[variant] = {
                "count": len(scores),
                "avg_quality": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0
            }

        return {
            "test_name": test_name,
            "total_samples": len(results),
            "variants": analysis
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        trend = self.get_quality_trend(7)

        return {
            "total_evaluations": len(self._eval_history),
            "quality_threshold": self.quality_threshold,
            "current_avg_quality": trend.avg_quality,
            "quality_trend": trend.trend,
            "below_threshold_pct": sum(
                1 for e in self._eval_history
                if e.quality_score < self.quality_threshold
            ) / len(self._eval_history) * 100 if self._eval_history else 0,
            "active_ab_tests": list(self._ab_results.keys())
        }


# Factory function
def create_continuous_evaluator(
    quality_threshold: float = 0.7
) -> ContinuousEvaluator:
    """Create a continuous evaluator."""
    return ContinuousEvaluator(quality_threshold=quality_threshold)
