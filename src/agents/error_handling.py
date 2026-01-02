"""
Error Handling module.

Provides robust error handling with retry logic, circuit breaker,
and fallback chains for resilient agent operations.
"""
import logging
import time
import functools
from typing import Callable, Optional, Any, List, Dict, TypeVar
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    retry_exceptions: tuple = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for an attempt using exponential backoff."""
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    attempts: int = 0
    errors: List[Exception] = field(default_factory=list)
    total_time: float = 0.0


class RetryHandler:
    """
    Handles retrying failed operations with exponential backoff.

    Supports configurable retry limits, delays, and exception filtering.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Optional retry configuration.
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            RetryResult with outcome details.
        """
        errors = []
        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    errors=errors,
                    total_time=time.time() - start_time
                )
            except self.config.retry_exceptions as e:
                errors.append(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

        return RetryResult(
            success=False,
            attempts=self.config.max_retries + 1,
            errors=errors,
            total_time=time.time() - start_time
        )

    async def execute_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Execute an async function with retry logic.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            RetryResult with outcome details.
        """
        import asyncio

        errors = []
        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    errors=errors,
                    total_time=time.time() - start_time
                )
            except self.config.retry_exceptions as e:
                errors.append(e)
                logger.warning(f"Async attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt)
                    await asyncio.sleep(delay)

        return RetryResult(
            success=False,
            attempts=self.config.max_retries + 1,
            errors=errors,
            total_time=time.time() - start_time
        )


class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures.

    When failures exceed a threshold, the circuit opens and rejects
    calls for a timeout period before trying again.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
            reset_timeout: Seconds before trying half-open.
            half_open_max_calls: Successful calls to close circuit.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = None  # For thread safety if needed

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, handling timeout transitions."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker entering half-open state")
        return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpen: If circuit is open.
            Exception: Original exception if function fails.
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened from half-open")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit breaker manually reset")


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class FallbackModel:
    """Configuration for a fallback model."""
    name: str
    cost_per_1k_tokens: float
    max_tokens: int = 8192
    priority: int = 0  # Lower = higher priority


class FallbackChain:
    """
    Manages a chain of fallback models.

    Tries models in order until one succeeds.
    """

    def __init__(
        self,
        models: Optional[List[FallbackModel]] = None,
        retry_handler: Optional[RetryHandler] = None
    ):
        """
        Initialize fallback chain.

        Args:
            models: List of fallback models in priority order.
            retry_handler: Optional retry handler for each model.
        """
        self.models = models or [
            FallbackModel("gemini-pro", 0.001, priority=0),
            FallbackModel("gemini-flash", 0.0005, priority=1),
        ]
        self.models.sort(key=lambda m: m.priority)
        self.retry_handler = retry_handler or RetryHandler(
            RetryConfig(max_retries=1, initial_delay=0.5)
        )

        self._usage: Dict[str, int] = {m.name: 0 for m in self.models}
        self._costs: Dict[str, float] = {m.name: 0.0 for m in self.models}

    def invoke(
        self,
        invoke_func: Callable[[str], str],
        prompt: str,
        model_name: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Invoke with fallback chain.

        Args:
            invoke_func: Function that takes model name and returns invoke function.
            prompt: The prompt to send.
            model_name: Optional specific model to start with.

        Returns:
            Tuple of (response, model_used).

        Raises:
            Exception: If all models fail.
        """
        models_to_try = self.models if not model_name else [
            m for m in self.models if m.name == model_name
        ] + [m for m in self.models if m.name != model_name]

        last_error = None

        for model in models_to_try:
            try:
                result = self.retry_handler.execute(
                    invoke_func, model.name, prompt
                )

                if result.success:
                    self._update_usage(model.name, len(prompt))
                    return result.result, model.name

                last_error = result.errors[-1] if result.errors else None
                logger.warning(f"Model {model.name} failed, trying next")

            except Exception as e:
                last_error = e
                logger.warning(f"Model {model.name} failed: {e}")
                continue

        raise last_error or Exception("All fallback models failed")

    def _update_usage(self, model_name: str, token_count: int) -> None:
        """Update usage tracking for a model."""
        self._usage[model_name] = self._usage.get(model_name, 0) + token_count
        model = next((m for m in self.models if m.name == model_name), None)
        if model:
            cost = (token_count / 1000) * model.cost_per_1k_tokens
            self._costs[model_name] = self._costs.get(model_name, 0.0) + cost

    def get_total_cost(self) -> float:
        """Get total estimated cost across all models."""
        return sum(self._costs.values())

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "usage": self._usage,
            "costs": self._costs,
            "total_cost": self.get_total_cost()
        }


# Decorator for retry
def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0
) -> Callable:
    """
    Decorator to add retry logic to a function.

    Args:
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        backoff_multiplier: Multiplier for exponential backoff.

    Returns:
        Decorated function.
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_multiplier=backoff_multiplier
    )
    handler = RetryHandler(config)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = handler.execute(func, *args, **kwargs)
            if result.success:
                return result.result
            raise result.errors[-1] if result.errors else Exception("Retry failed")
        return wrapper
    return decorator


# Factory functions
def create_retry_handler(
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> RetryHandler:
    """Create a retry handler with common defaults."""
    return RetryHandler(RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay
    ))


def create_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
) -> CircuitBreaker:
    """Create a circuit breaker with common defaults."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        reset_timeout=reset_timeout
    )
