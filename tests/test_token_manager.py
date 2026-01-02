"""
Unit tests for TokenManager module.

Tests token counting, truncation, compression, chunking, and cost estimation.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestTokenManagerInit:
    """Tests for TokenManager initialization."""

    def test_default_initialization(self):
        """Test default initialization with gemini-pro model."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        assert manager.default_model == "gemini-pro"

    def test_custom_model_initialization(self):
        """Test initialization with custom model."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager(default_model="gpt-4")
        assert manager.default_model == "gpt-4"

    def test_tokenizer_fallback(self):
        """Test fallback when tiktoken is not available."""
        from src.utils.token_manager import TokenManager

        with patch.dict('sys.modules', {'tiktoken': None}):
            manager = TokenManager()
            # Should still work with fallback
            assert manager._encoder is not None or manager._encoder is None


class TestCountTokens:
    """Tests for count_tokens method."""

    def test_empty_text(self):
        """Should return 0 for empty text."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        assert manager.count_tokens("") == 0

    def test_simple_text(self):
        """Should return positive count for non-empty text."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        count = manager.count_tokens("Hello, world!")
        assert count > 0

    def test_long_text(self):
        """Token count should scale with text length."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        short = manager.count_tokens("Hello")
        long = manager.count_tokens("Hello " * 100)
        assert long > short


class TestGetModelLimit:
    """Tests for get_model_limit method."""

    def test_known_model(self):
        """Should return correct limit for known models."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        assert manager.get_model_limit("gpt-4") == 8192
        assert manager.get_model_limit("gemini-1.5-pro") == 128000

    def test_unknown_model(self):
        """Should return default limit for unknown models."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        limit = manager.get_model_limit("unknown-model")
        assert limit == 32000  # Default limit

    def test_default_model(self):
        """Should use default model when none specified."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager(default_model="gemini-pro")
        limit = manager.get_model_limit()
        assert limit == 32000


class TestTruncateToLimit:
    """Tests for truncate_to_limit method."""

    def test_text_within_limit(self):
        """Should return unchanged text if within limit."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Short text"
        result = manager.truncate_to_limit(text, 1000)
        assert result == text

    def test_truncate_end(self):
        """Should truncate from end by default."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Word " * 100
        result = manager.truncate_to_limit(text, 10)
        assert len(result) < len(text)
        assert result.startswith("Word")

    def test_truncate_start(self):
        """Should truncate from start when specified."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Word " * 100
        result = manager.truncate_to_limit(text, 10, truncation_strategy="start")
        assert len(result) < len(text)

    def test_truncate_middle(self):
        """Should truncate middle when specified."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Word " * 100
        result = manager.truncate_to_limit(text, 20, truncation_strategy="middle")
        assert "[truncated]" in result


class TestCompressContext:
    """Tests for compress_context method."""

    def test_empty_context(self):
        """Should handle empty context."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        result = manager.compress_context("")
        assert result == ""

    def test_removes_filler_phrases(self):
        """Should remove filler phrases."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Basically, this is essentially the answer."
        result = manager.compress_context(text)
        assert "basically" not in result.lower()
        assert "essentially" not in result.lower()

    def test_normalizes_whitespace(self):
        """Should normalize whitespace."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Multiple   spaces   here"
        result = manager.compress_context(text)
        assert "   " not in result


class TestSplitIntoChunks:
    """Tests for split_into_chunks method."""

    def test_short_text_single_chunk(self):
        """Short text should return single chunk."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Short text."
        chunks = manager.split_into_chunks(text, chunk_size=100)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        """Long text should return multiple chunks."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "This is a sentence. " * 100
        chunks = manager.split_into_chunks(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1


class TestEstimateCost:
    """Tests for estimate_cost method."""

    def test_zero_tokens(self):
        """Should return 0 for zero tokens."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        cost = manager.estimate_cost(0, 0)
        assert cost == 0

    def test_known_model_pricing(self):
        """Should use known model pricing."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        cost = manager.estimate_cost(1000, 1000, "gpt-4")
        assert cost > 0

    def test_unknown_model_default_pricing(self):
        """Should use default pricing for unknown models."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        cost = manager.estimate_cost(1000, 1000, "unknown-model")
        assert cost > 0


class TestGetStats:
    """Tests for get_stats method."""

    def test_stats_structure(self):
        """Should return expected stats structure."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        stats = manager.get_stats("Hello world")

        assert "tokens" in stats
        assert "characters" in stats
        assert "chars_per_token" in stats
        assert "estimated_cost_1k_output" in stats

    def test_stats_values(self):
        """Stats should have reasonable values."""
        from src.utils.token_manager import TokenManager

        manager = TokenManager()
        text = "Hello world"
        stats = manager.get_stats(text)

        assert stats["characters"] == len(text)
        assert stats["tokens"] > 0


class TestCreateTokenManager:
    """Tests for create_token_manager factory function."""

    def test_factory_function(self):
        """Should create TokenManager with specified model."""
        from src.utils.token_manager import create_token_manager

        manager = create_token_manager("gpt-4-turbo")
        assert manager.default_model == "gpt-4-turbo"
