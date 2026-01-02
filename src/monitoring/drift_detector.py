"""
Drift Detector module.

Monitors semantic drift in embeddings, query patterns, and response quality
to detect when system retraining or re-indexing is needed.
"""
import logging
import time
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for a drift measurement."""
    timestamp: float
    embedding_drift: float  # 0.0 = no drift, 1.0 = complete drift
    query_drift: float
    quality_drift: float
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report of drift analysis."""
    overall_drift_score: float
    embedding_drift: float
    query_pattern_drift: float
    quality_drift: float
    is_significant: bool
    recommendations: List[str]
    analysis_window_days: int
    sample_count: int


class DriftDetector:
    """
    Detects semantic drift in RAG systems.

    Monitors:
    - Embedding distribution changes
    - Query pattern shifts
    - Response quality degradation
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Initialize drift detector.

        Args:
            drift_threshold: Threshold above which drift is significant.
            window_size: Number of samples to keep in rolling window.
            min_samples: Minimum samples needed for drift calculation.
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.min_samples = min_samples

        # Rolling windows for different metrics
        self._embeddings: deque = deque(maxlen=window_size)
        self._query_patterns: deque = deque(maxlen=window_size)
        self._quality_scores: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

        # Baseline statistics (set during calibration)
        self._baseline_embedding_mean: Optional[List[float]] = None
        self._baseline_embedding_var: Optional[List[float]] = None
        self._baseline_query_distribution: Dict[str, float] = {}
        self._baseline_quality_mean: float = 0.0

        self._is_calibrated = False
        self._drift_history: List[DriftMetrics] = []

        logger.info(f"DriftDetector initialized: threshold={drift_threshold}")

    def record_embedding(
        self,
        embedding: List[float],
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an embedding for drift monitoring.

        Args:
            embedding: The embedding vector.
            category: Query category for pattern tracking.
            metadata: Optional additional metadata.
        """
        self._embeddings.append({
            "vector": embedding,
            "category": category,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self._query_patterns.append(category)
        self._timestamps.append(time.time())

    def record_quality_score(
        self,
        score: float,
        query: Optional[str] = None
    ) -> None:
        """
        Record a quality score for drift monitoring.

        Args:
            score: Quality score (0.0 to 1.0).
            query: Optional associated query.
        """
        self._quality_scores.append({
            "score": score,
            "query": query,
            "timestamp": time.time()
        })

    def calibrate_baseline(self) -> bool:
        """
        Calibrate baseline statistics from current samples.

        Should be called after initial warm-up period.

        Returns:
            True if calibration successful, False otherwise.
        """
        if len(self._embeddings) < self.min_samples:
            logger.warning(f"Not enough samples for calibration: "
                          f"{len(self._embeddings)}/{self.min_samples}")
            return False

        # Calculate embedding baseline
        embeddings = [e["vector"] for e in self._embeddings]
        self._baseline_embedding_mean = self._calculate_mean(embeddings)
        self._baseline_embedding_var = self._calculate_variance(
            embeddings, self._baseline_embedding_mean
        )

        # Calculate query distribution baseline
        query_counts: Dict[str, int] = {}
        for category in self._query_patterns:
            query_counts[category] = query_counts.get(category, 0) + 1
        total = sum(query_counts.values())
        self._baseline_query_distribution = {
            k: v / total for k, v in query_counts.items()
        }

        # Calculate quality baseline
        if self._quality_scores:
            self._baseline_quality_mean = sum(
                s["score"] for s in self._quality_scores
            ) / len(self._quality_scores)

        self._is_calibrated = True
        logger.info("Drift detector calibrated")
        return True

    def check_drift(
        self,
        window_days: int = 7
    ) -> DriftReport:
        """
        Check for drift in recent data.

        Args:
            window_days: Days to analyze for drift.

        Returns:
            DriftReport with analysis results.
        """
        if not self._is_calibrated:
            # Auto-calibrate if enough samples
            if len(self._embeddings) >= self.min_samples:
                self.calibrate_baseline()
            else:
                return DriftReport(
                    overall_drift_score=0.0,
                    embedding_drift=0.0,
                    query_pattern_drift=0.0,
                    quality_drift=0.0,
                    is_significant=False,
                    recommendations=["Not enough samples for drift detection"],
                    analysis_window_days=window_days,
                    sample_count=len(self._embeddings)
                )

        cutoff = time.time() - (window_days * 86400)

        # Calculate embedding drift
        recent_embeddings = [
            e["vector"] for e in self._embeddings
            if e["timestamp"] >= cutoff
        ]
        embedding_drift = self._calculate_embedding_drift(recent_embeddings)

        # Calculate query pattern drift
        recent_queries = [
            q for i, q in enumerate(self._query_patterns)
            if self._timestamps[i] >= cutoff
        ]
        query_drift = self._calculate_query_drift(recent_queries)

        # Calculate quality drift
        recent_quality = [
            s["score"] for s in self._quality_scores
            if s["timestamp"] >= cutoff
        ]
        quality_drift = self._calculate_quality_drift(recent_quality)

        # Overall drift score (weighted average)
        overall = (
            0.4 * embedding_drift +
            0.3 * query_drift +
            0.3 * quality_drift
        )

        is_significant = overall >= self.drift_threshold

        recommendations = self._generate_recommendations(
            embedding_drift, query_drift, quality_drift
        )

        report = DriftReport(
            overall_drift_score=overall,
            embedding_drift=embedding_drift,
            query_pattern_drift=query_drift,
            quality_drift=quality_drift,
            is_significant=is_significant,
            recommendations=recommendations,
            analysis_window_days=window_days,
            sample_count=len(recent_embeddings)
        )

        # Record drift metrics
        self._drift_history.append(DriftMetrics(
            timestamp=time.time(),
            embedding_drift=embedding_drift,
            query_drift=query_drift,
            quality_drift=quality_drift,
            sample_size=len(recent_embeddings)
        ))

        if is_significant:
            logger.warning(f"Significant drift detected: {overall:.2f}")

        return report

    def get_drift_score(self) -> float:
        """Get the current overall drift score."""
        report = self.check_drift()
        return report.overall_drift_score

    def get_drift_history(
        self,
        window_days: int = 30
    ) -> List[DriftMetrics]:
        """Get historical drift metrics."""
        cutoff = time.time() - (window_days * 86400)
        return [m for m in self._drift_history if m.timestamp >= cutoff]

    def reset_baseline(self) -> None:
        """Reset and recalibrate baseline."""
        self._is_calibrated = False
        self.calibrate_baseline()

    def _calculate_embedding_drift(
        self,
        embeddings: List[List[float]]
    ) -> float:
        """Calculate drift in embedding distribution."""
        if not embeddings or not self._baseline_embedding_mean:
            return 0.0

        # Calculate current mean
        current_mean = self._calculate_mean(embeddings)

        # Calculate cosine distance from baseline
        drift = 1.0 - self._cosine_similarity(
            self._baseline_embedding_mean,
            current_mean
        )

        return min(1.0, drift * 2)  # Scale for sensitivity

    def _calculate_query_drift(
        self,
        queries: List[str]
    ) -> float:
        """Calculate drift in query patterns."""
        if not queries or not self._baseline_query_distribution:
            return 0.0

        # Calculate current distribution
        query_counts: Dict[str, int] = {}
        for q in queries:
            query_counts[q] = query_counts.get(q, 0) + 1
        total = sum(query_counts.values())
        current_dist = {k: v / total for k, v in query_counts.items()}

        # Calculate Jensen-Shannon divergence
        all_keys = set(self._baseline_query_distribution.keys()) | set(current_dist.keys())

        kl_sum = 0.0
        for key in all_keys:
            p = self._baseline_query_distribution.get(key, 0.001)
            q = current_dist.get(key, 0.001)
            m = (p + q) / 2
            kl_sum += p * math.log(p / m) + q * math.log(q / m)

        js_divergence = kl_sum / 2
        return min(1.0, js_divergence)

    def _calculate_quality_drift(
        self,
        scores: List[float]
    ) -> float:
        """Calculate drift in quality scores."""
        if not scores or self._baseline_quality_mean == 0:
            return 0.0

        current_mean = sum(scores) / len(scores)
        diff = abs(current_mean - self._baseline_quality_mean)

        # Normalize: 0.5 difference = 1.0 drift
        return min(1.0, diff * 2)

    def _calculate_mean(self, vectors: List[List[float]]) -> List[float]:
        """Calculate mean of embedding vectors."""
        if not vectors:
            return []
        dim = len(vectors[0])
        mean = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                mean[i] += v
        return [m / len(vectors) for m in mean]

    def _calculate_variance(
        self,
        vectors: List[List[float]],
        mean: List[float]
    ) -> List[float]:
        """Calculate variance of embedding vectors."""
        if not vectors:
            return []
        dim = len(vectors[0])
        var = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                var[i] += (v - mean[i]) ** 2
        return [v / len(vectors) for v in var]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _generate_recommendations(
        self,
        embedding_drift: float,
        query_drift: float,
        quality_drift: float
    ) -> List[str]:
        """Generate recommendations based on drift metrics."""
        recommendations = []

        if embedding_drift > self.drift_threshold:
            recommendations.append(
                "Consider re-indexing documents - embedding distribution has shifted"
            )

        if query_drift > self.drift_threshold:
            recommendations.append(
                "Query patterns have changed significantly - review retrieval strategy"
            )

        if quality_drift > self.drift_threshold:
            recommendations.append(
                "Response quality is degrading - consider model fine-tuning or prompt updates"
            )

        if not recommendations:
            recommendations.append("System is operating within normal parameters")

        return recommendations


# Factory function
def create_drift_detector(
    threshold: float = 0.3,
    window_size: int = 1000
) -> DriftDetector:
    """Create a drift detector with specified settings."""
    return DriftDetector(
        drift_threshold=threshold,
        window_size=window_size
    )
