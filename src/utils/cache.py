"""
Cache module.

Provides multi-tier caching with in-memory LRU cache and optional Redis backend
for response caching and semantic similarity matching.
"""
import logging
import time
import hashlib
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import lru_cache

try:
    from prometheus_client import Counter
    CACHE_HITS = Counter("cache_hits_total", "Total number of cache hits", ["type"])
    CACHE_MISSES = Counter("cache_misses_total", "Total number of cache misses", ["type"])

    # Initialize labels to ensure they show up in Prometheus
    for t in ["local", "redis", "semantic", "total"]:
        CACHE_HITS.labels(type=t)
        CACHE_MISSES.labels(type=t)

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry."""
    key: str
    value: str
    query: str
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update hit count."""
        self.hit_count += 1


class LRUCache:
    """
    In-memory LRU (Least Recently Used) cache.

    Evicts least recently used entries when capacity is reached.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get an entry from cache.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._hits += 1

        return entry

    def set(
        self,
        key: str,
        value: str,
        query: str,
        ttl: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set an entry in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            query: Original query.
            ttl: Time-to-live in seconds.
            metadata: Optional metadata.
        """
        # Remove if exists to update position
        if key in self._cache:
            del self._cache[key]

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            query=query,
            ttl=ttl,
            metadata=metadata or {}
        )

    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0
        }


class ResponseCache:
    """
    High-level response cache with semantic matching.

    Supports exact match caching with optional semantic similarity.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        enable_semantic: bool = False,
        similarity_threshold: float = 0.95,
        redis_url: Optional[str] = None
    ):
        """
        Initialize response cache.

        Args:
            max_size: Maximum cache entries.
            default_ttl: Default TTL in seconds.
            enable_semantic: Enable semantic similarity matching.
            similarity_threshold: Threshold for semantic match (0.0-1.0).
            redis_url: Optional Redis URL for distributed caching.
        """
        self.default_ttl = default_ttl
        self.enable_semantic = enable_semantic
        self.similarity_threshold = similarity_threshold

        self._local_cache = LRUCache(max_size)
        self._redis_client = None

        if redis_url:
            self._init_redis(redis_url)

        logger.info(f"ResponseCache initialized: max_size={max_size}, "
                   f"semantic={enable_semantic}")

    def _init_redis(self, redis_url: str) -> None:
        """Initialize Redis connection."""
        try:
            import redis
            self._redis_client = redis.from_url(redis_url)
            self._redis_client.ping()
            logger.info("Redis cache connected")
        except ImportError:
            logger.warning("Redis package not installed, using local cache only")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using local cache only")

    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Get cached response for a query.

        Args:
            query: The query to look up.
            threshold: Optional similarity threshold override.

        Returns:
            Cached response if found, None otherwise.
        """
        key = self._generate_key(query)

        # Try exact match in local cache
        entry = self._local_cache.get(key)
        if entry:
            logger.debug(f"Cache hit (local): {key[:8]}...")
            if METRICS_AVAILABLE:
                CACHE_HITS.labels(type="local").inc()
            return entry.value

        # Try Redis if available
        if self._redis_client:
            try:
                value = self._redis_client.get(f"cache:{key}")
                if value:
                    logger.debug(f"Cache hit (redis): {key[:8]}...")
                    if METRICS_AVAILABLE:
                        CACHE_HITS.labels(type="redis").inc()
                    return value.decode() if isinstance(value, bytes) else value
                elif METRICS_AVAILABLE:
                    CACHE_MISSES.labels(type="redis").inc()
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Semantic matching (if enabled and embeddings available)
        if self.enable_semantic:
            match = self._find_semantic_match(query, threshold)
            if match:
                logger.debug("Cache hit (semantic)")
                if METRICS_AVAILABLE:
                    CACHE_HITS.labels(type="semantic").inc()
                return match
            elif METRICS_AVAILABLE:
                CACHE_MISSES.labels(type="semantic").inc()

        # Record final miss
        if METRICS_AVAILABLE:
            CACHE_MISSES.labels(type="local").inc()
            CACHE_MISSES.labels(type="total").inc()

        return None

    def set(
        self,
        query: str,
        response: str,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache a response.

        Args:
            query: The original query.
            response: The response to cache.
            ttl: Optional TTL override.
            metadata: Optional metadata.
        """
        key = self._generate_key(query)
        ttl = ttl or self.default_ttl

        # Store in local cache
        self._local_cache.set(key, response, query, ttl, metadata)

        # Store in Redis if available
        if self._redis_client:
            try:
                self._redis_client.setex(
                    f"cache:{key}",
                    ttl,
                    response
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        logger.debug(f"Cached response: {key[:8]}...")

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            pattern: Optional pattern to match (invalidate all if None).

        Returns:
            Number of entries invalidated.
        """
        count = 0

        if pattern is None:
            count = len(self._local_cache._cache)
            self._local_cache.clear()
        else:
            # Pattern-based invalidation
            keys_to_delete = [
                key for key, entry in self._local_cache._cache.items()
                if pattern.lower() in entry.query.lower()
            ]
            for key in keys_to_delete:
                self._local_cache.delete(key)
                count += 1

        logger.info(f"Invalidated {count} cache entries")
        return count

    def _find_semantic_match(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Find semantically similar cached query.

        Production implementation should:
        1. Generate embeddings for the query using the same model as documents
        2. Compute cosine similarity against cached query embeddings
        3. Return cached response if similarity >= threshold
        
        Current implementation: Simple exact normalized string matching.
        """
        # Simplified: exact normalized match only
        # Full implementation would compute embeddings and cosine similarity
        threshold = threshold or self.similarity_threshold
        normalized_query = query.strip().lower()

        for entry in self._local_cache._cache.values():
            if entry.query.strip().lower() == normalized_query:
                return entry.value

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._local_cache.get_stats()
        stats["redis_enabled"] = self._redis_client is not None
        stats["semantic_enabled"] = self.enable_semantic
        return stats


# Decorator for caching function results
def cached(
    ttl: int = 3600,
    key_prefix: str = "func"
) -> callable:
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds.
        key_prefix: Prefix for cache keys.

    Returns:
        Decorated function.
    """
    cache = ResponseCache(max_size=100, default_ttl=ttl)

    def decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs):
            # Generate key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            result = cache.get(key)
            if result:
                return result

            result = func(*args, **kwargs)
            cache.set(key, str(result), ttl)
            return result

        return wrapper
    return decorator


# Factory function
def create_response_cache(
    redis_url: Optional[str] = None,
    enable_semantic: bool = False
) -> ResponseCache:
    """Create a response cache with sensible defaults."""
    return ResponseCache(
        max_size=1000,
        default_ttl=3600,
        enable_semantic=enable_semantic,
        redis_url=redis_url
    )
