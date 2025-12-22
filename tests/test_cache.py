"""
Unit tests for Cache module.

Tests CacheEntry, LRUCache, ResponseCache, and caching decorator.
"""
import pytest
import os
import time
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_creation(self):
        """Test CacheEntry creation."""
        from src.utils.cache import CacheEntry
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            query="test query"
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.hit_count == 0
    
    def test_is_expired_false(self):
        """Entry should not be expired immediately."""
        from src.utils.cache import CacheEntry
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            query="test query",
            ttl=3600
        )
        assert entry.is_expired is False
    
    def test_is_expired_true(self):
        """Entry should be expired after TTL."""
        from src.utils.cache import CacheEntry
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            query="test query",
            ttl=0,
            created_at=time.time() - 10
        )
        assert entry.is_expired is True
    
    def test_touch_increments_hit_count(self):
        """Touch should increment hit count."""
        from src.utils.cache import CacheEntry
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            query="test query"
        )
        entry.touch()
        entry.touch()
        assert entry.hit_count == 2


class TestLRUCache:
    """Tests for LRUCache class."""
    
    def test_initialization(self):
        """Test LRU cache initialization."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100)
        assert cache.max_size == 100
    
    def test_set_and_get(self):
        """Test setting and getting values."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache()
        cache.set("key1", "value1", "query1")
        
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"
    
    def test_get_nonexistent_key(self):
        """Should return None for nonexistent key."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache()
        entry = cache.get("nonexistent")
        assert entry is None
    
    def test_eviction_on_full(self):
        """Should evict oldest when full."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=2)
        cache.set("key1", "value1", "query1")
        cache.set("key2", "value2", "query2")
        cache.set("key3", "value3", "query3")
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key3") is not None
    
    def test_delete(self):
        """Test deleting entries."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache()
        cache.set("key1", "value1", "query1")
        
        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None
    
    def test_delete_nonexistent(self):
        """Delete should return False for nonexistent key."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache()
        result = cache.delete("nonexistent")
        assert result is False
    
    def test_clear(self):
        """Test clearing cache."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache()
        cache.set("key1", "value1", "query1")
        cache.set("key2", "value2", "query2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100)
        cache.set("key1", "value1", "query1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestResponseCache:
    """Tests for ResponseCache class."""
    
    def test_initialization(self):
        """Test response cache initialization."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache(max_size=100, default_ttl=1800)
        assert cache.default_ttl == 1800
    
    def test_set_and_get(self):
        """Test setting and getting responses."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("What is Python?", "Python is a programming language.")
        
        response = cache.get("What is Python?")
        assert response == "Python is a programming language."
    
    def test_case_insensitive_key(self):
        """Keys should be case-insensitive."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("What is Python?", "Answer here")
        
        response = cache.get("what is python?")
        assert response == "Answer here"
    
    def test_cache_miss(self):
        """Should return None on cache miss."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        response = cache.get("Unknown query")
        assert response is None
    
    def test_invalidate_all(self):
        """Test invalidating all entries."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("q1", "a1")
        cache.set("q2", "a2")
        
        count = cache.invalidate()
        assert count == 2
        assert cache.get("q1") is None
    
    def test_invalidate_pattern(self):
        """Test invalidating by pattern."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("python question", "answer1")
        cache.set("java question", "answer2")
        
        count = cache.invalidate("python")
        assert count == 1
        assert cache.get("python question") is None
        assert cache.get("java question") is not None
    
    def test_get_stats(self):
        """Test getting response cache stats."""
        from src.utils.cache import ResponseCache
        
        cache = ResponseCache()
        cache.set("q1", "a1")
        
        stats = cache.get_stats()
        assert "size" in stats
        assert "redis_enabled" in stats
        assert "semantic_enabled" in stats


class TestCachedDecorator:
    """Tests for the @cached decorator."""
    
    def test_caches_function_result(self):
        """Decorator should cache function results."""
        from src.utils.cache import cached
        
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        # Second call with same args - should use cache
        result2 = expensive_function(5)
        
        assert str(result1) == result2  # Cache retourne une string


class TestCreateResponseCache:
    """Tests for create_response_cache factory function."""
    
    def test_factory_function(self):
        """Should create ResponseCache with defaults."""
        from src.utils.cache import create_response_cache
        
        cache = create_response_cache()
        assert cache is not None
        assert cache.default_ttl == 3600
    
    def test_factory_with_semantic(self):
        """Should create cache with semantic matching."""
        from src.utils.cache import create_response_cache
        
        cache = create_response_cache(enable_semantic=True)
        assert cache.enable_semantic is True
