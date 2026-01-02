"""
Unit tests for configuration module.

Tests environment variable loading and configuration defaults.
"""
import pytest
import os
from unittest.mock import patch


class TestGetRequiredEnv:
    """Tests for get_required_env function."""

    def test_existing_env_var(self):
        """Should return value of existing environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            # Need to reimport to get fresh function
            from src.utils.config import get_required_env

            # Test with existing var
            os.environ["MY_TEST_VAR"] = "my_value"
            from src.utils import config
            result = config.get_required_env("MY_TEST_VAR")
            assert result == "my_value"
            del os.environ["MY_TEST_VAR"]

    def test_missing_env_var_raises(self):
        """Should raise ValueError for missing environment variable."""
        # Ensure the var doesn't exist
        if "NONEXISTENT_VAR_12345" in os.environ:
            del os.environ["NONEXISTENT_VAR_12345"]

        from src.utils.config import get_required_env

        with pytest.raises(ValueError) as excinfo:
            get_required_env("NONEXISTENT_VAR_12345")

        assert "NONEXISTENT_VAR_12345" in str(excinfo.value)


class TestGetOptionalEnv:
    """Tests for get_optional_env function."""

    def test_existing_env_var(self):
        """Should return value of existing environment variable."""
        os.environ["OPTIONAL_TEST_VAR"] = "custom_value"

        from src.utils.config import get_optional_env
        result = get_optional_env("OPTIONAL_TEST_VAR", "default")

        assert result == "custom_value"
        del os.environ["OPTIONAL_TEST_VAR"]

    def test_missing_env_var_returns_default(self):
        """Should return default for missing environment variable."""
        if "MISSING_OPTIONAL_VAR" in os.environ:
            del os.environ["MISSING_OPTIONAL_VAR"]

        from src.utils.config import get_optional_env
        result = get_optional_env("MISSING_OPTIONAL_VAR", "my_default")

        assert result == "my_default"


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_chunk_size_is_int(self):
        """CHUNK_SIZE should be an integer."""
        from src.utils.config import CHUNK_SIZE
        assert isinstance(CHUNK_SIZE, int)
        assert CHUNK_SIZE > 0

    def test_chunk_overlap_is_int(self):
        """CHUNK_OVERLAP should be an integer."""
        from src.utils.config import CHUNK_OVERLAP
        assert isinstance(CHUNK_OVERLAP, int)
        assert CHUNK_OVERLAP >= 0

    def test_retriever_k_is_int(self):
        """RETRIEVER_K should be an integer."""
        from src.utils.config import RETRIEVER_K
        assert isinstance(RETRIEVER_K, int)
        assert RETRIEVER_K > 0

    def test_weights_are_floats(self):
        """BM25_WEIGHT and VECTOR_WEIGHT should be floats."""
        from src.utils.config import BM25_WEIGHT, VECTOR_WEIGHT
        assert isinstance(BM25_WEIGHT, float)
        assert isinstance(VECTOR_WEIGHT, float)
        assert 0 <= BM25_WEIGHT <= 1
        assert 0 <= VECTOR_WEIGHT <= 1

    def test_paths_are_strings(self):
        """Path configurations should be strings."""
        from src.utils.config import VECTOR_STORE_PATH, PDF_DATA_PATH
        assert isinstance(VECTOR_STORE_PATH, str)
        assert isinstance(PDF_DATA_PATH, str)
