"""
Unit tests for Error Handling module.

Tests RetryHandler, CircuitBreaker, FallbackChain, and decorators.
"""
import pytest
import os
import time
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from src.agents.error_handling import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        from src.agents.error_handling import RetryConfig

        config = RetryConfig(initial_delay=1.0, backoff_multiplier=2.0)

        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        from src.agents.error_handling import RetryConfig

        config = RetryConfig(initial_delay=1.0, backoff_multiplier=10.0, max_delay=5.0)

        delay = config.get_delay(5)
        assert delay == 5.0


class TestRetryHandler:
    """Tests for RetryHandler class."""

    def test_successful_execution(self):
        """Test successful function execution."""
        from src.agents.error_handling import RetryHandler

        handler = RetryHandler()

        def success_func():
            return "success"

        result = handler.execute(success_func)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1

    def test_retry_on_failure(self):
        """Test retry on failure."""
        from src.agents.error_handling import RetryHandler, RetryConfig

        config = RetryConfig(max_retries=2, initial_delay=0.01)
        handler = RetryHandler(config)

        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = handler.execute(failing_func)
        assert result.success is True
        assert result.attempts == 2

    def test_all_retries_exhausted(self):
        """Test when all retries are exhausted."""
        from src.agents.error_handling import RetryHandler, RetryConfig

        config = RetryConfig(max_retries=2, initial_delay=0.01)
        handler = RetryHandler(config)

        def always_fails():
            raise ValueError("Always fails")

        result = handler.execute(always_fails)
        assert result.success is False
        assert result.attempts == 3  # Initial + 2 retries
        assert len(result.errors) == 3


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Circuit should start closed."""
        from src.agents.error_handling import CircuitBreaker, CircuitState

        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call(self):
        """Successful calls should pass through."""
        from src.agents.error_handling import CircuitBreaker

        breaker = CircuitBreaker()

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"

    def test_opens_after_failures(self):
        """Circuit should open after reaching failure threshold."""
        from src.agents.error_handling import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3)

        def failing_func():
            raise ValueError("Failure")

        for _ in range(3):
            try:
                breaker.call(failing_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        """Open circuit should reject calls."""
        from src.agents.error_handling import CircuitBreaker, CircuitBreakerOpen

        breaker = CircuitBreaker(failure_threshold=1)

        # Force open
        try:
            breaker.call(lambda: 1/0)
        except ZeroDivisionError:
            pass

        with pytest.raises(CircuitBreakerOpen):
            breaker.call(lambda: "test")

    def test_reset(self):
        """Manual reset should close circuit."""
        from src.agents.error_handling import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=1)

        # Force open
        try:
            breaker.call(lambda: 1/0)
        except ZeroDivisionError:
            pass

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_transition(self):
        """Circuit should transition to half-open after timeout."""
        from src.agents.error_handling import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.01)

        # Force open
        try:
            breaker.call(lambda: 1/0)
        except ZeroDivisionError:
            pass

        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN


class TestFallbackModel:
    """Tests for FallbackModel dataclass."""

    def test_creation(self):
        """Test FallbackModel creation."""
        from src.agents.error_handling import FallbackModel

        model = FallbackModel(
            name="gemini-pro",
            cost_per_1k_tokens=0.001
        )
        assert model.name == "gemini-pro"
        assert model.priority == 0


class TestFallbackChain:
    """Tests for FallbackChain class."""

    def test_initialization(self):
        """Test FallbackChain initialization."""
        from src.agents.error_handling import FallbackChain

        chain = FallbackChain()
        assert len(chain.models) >= 1

    def test_invoke_success(self):
        """Test successful invoke."""
        from src.agents.error_handling import FallbackChain

        chain = FallbackChain()

        def mock_invoke(model_name, prompt):
            return f"Response from {model_name}"

        response, model_used = chain.invoke(mock_invoke, "test prompt")
        assert "Response from" in response

    def test_get_total_cost(self):
        """Test getting total cost."""
        from src.agents.error_handling import FallbackChain

        chain = FallbackChain()
        cost = chain.get_total_cost()
        assert cost >= 0

    def test_get_usage_stats(self):
        """Test getting usage stats."""
        from src.agents.error_handling import FallbackChain

        chain = FallbackChain()
        stats = chain.get_usage_stats()

        assert "usage" in stats
        assert "costs" in stats
        assert "total_cost" in stats


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""
        from src.agents.error_handling import with_retry

        @with_retry(max_retries=1, initial_delay=0.01)
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_decorator_retries(self):
        """Test decorator retries on failure."""
        from src.agents.error_handling import with_retry

        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_retry_handler(self):
        """Test create_retry_handler factory."""
        from src.agents.error_handling import create_retry_handler

        handler = create_retry_handler(max_retries=5, initial_delay=0.5)
        assert handler.config.max_retries == 5
        assert handler.config.initial_delay == 0.5

    def test_create_circuit_breaker(self):
        """Test create_circuit_breaker factory."""
        from src.agents.error_handling import create_circuit_breaker

        breaker = create_circuit_breaker(failure_threshold=10, reset_timeout=120.0)
        assert breaker.failure_threshold == 10
        assert breaker.reset_timeout == 120.0
