"""
Tracing module.

Provides request tracing with OpenTelemetry integration for
observability and debugging.
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A tracing span representing a unit of work."""
    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@dataclass
class TraceContext:
    """Context for a distributed trace."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get total trace duration."""
        return (time.time() - self.start_time) * 1000

    @property
    def root_span(self) -> Optional[Span]:
        """Get the root span."""
        return self.spans[0] if self.spans else None


class Tracer:
    """
    Request tracing for observability.

    Provides span-based tracing compatible with OpenTelemetry concepts.
    """

    def __init__(
        self,
        service_name: str = "llm-agents",
        enable_export: bool = False,
        export_endpoint: Optional[str] = None
    ):
        """
        Initialize tracer.

        Args:
            service_name: Name of the service for traces.
            enable_export: Whether to export traces.
            export_endpoint: Endpoint for trace export (e.g., Jaeger).
        """
        self.service_name = service_name
        self.enable_export = enable_export
        self.export_endpoint = export_endpoint

        self._active_traces: Dict[str, TraceContext] = {}
        self._completed_traces: List[TraceContext] = []
        self._max_completed = 1000

        self._otel_tracer = None
        if enable_export:
            self._init_otel()

        logger.info(f"Tracer initialized: service={service_name}")

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry tracer."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.service_name)

            logger.info("OpenTelemetry tracer initialized")
        except ImportError:
            logger.warning("OpenTelemetry not installed, using local tracing only")

    def start_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceContext:
        """
        Start a new trace.

        Args:
            name: Name of the trace.
            trace_id: Optional trace ID (auto-generated if not provided).
            attributes: Optional trace attributes.

        Returns:
            TraceContext for the new trace.
        """
        trace_id = trace_id or self._generate_id()

        root_span = Span(
            trace_id=trace_id,
            span_id=self._generate_id()[:16],
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        context = TraceContext(
            trace_id=trace_id,
            spans=[root_span],
            metadata={"name": name}
        )

        self._active_traces[trace_id] = context
        logger.debug(f"Started trace: {name} ({trace_id[:8]}...)")

        return context

    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span within a trace.

        Args:
            trace_id: ID of the parent trace.
            name: Name of the span.
            parent_span_id: Optional parent span ID.
            attributes: Optional span attributes.

        Returns:
            The new Span.
        """
        if trace_id not in self._active_traces:
            raise ValueError(f"Trace not found: {trace_id}")

        context = self._active_traces[trace_id]

        # Use root span as parent if not specified
        if parent_span_id is None and context.spans:
            parent_span_id = context.spans[-1].span_id

        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id()[:16],
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )

        context.spans.append(span)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End a span.

        Args:
            span: The span to end.
            status: Status code (OK or ERROR).
            attributes: Additional attributes to set.
        """
        span.end_time = time.time()
        span.status = status
        if attributes:
            span.attributes.update(attributes)

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to a span."""
        span.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def end_trace(self, trace_id: str) -> Optional[TraceContext]:
        """
        End a trace.

        Args:
            trace_id: ID of the trace to end.

        Returns:
            The completed TraceContext.
        """
        if trace_id not in self._active_traces:
            return None

        context = self._active_traces.pop(trace_id)

        # End any open spans
        for span in context.spans:
            if span.end_time is None:
                span.end_time = time.time()

        # Store completed trace
        self._completed_traces.append(context)
        if len(self._completed_traces) > self._max_completed:
            self._completed_traces = self._completed_traces[-self._max_completed:]

        # Export if enabled
        if self.enable_export:
            self._export_trace(context)

        logger.debug(f"Ended trace: {context.metadata.get('name')} "
                    f"({trace_id[:8]}..., {context.duration_ms:.2f}ms)")

        return context

    @contextmanager
    def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing a block of code.

        Args:
            name: Name of the trace.
            attributes: Optional trace attributes.

        Yields:
            TraceContext for the trace.
        """
        context = self.start_trace(name, attributes=attributes)
        try:
            yield context
        except Exception as e:
            if context.spans:
                context.spans[-1].status = "ERROR"
                self.add_event(context.spans[-1], "exception", {"message": str(e)})
            raise
        finally:
            self.end_trace(context.trace_id)

    @contextmanager
    def span(
        self,
        trace_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for a span.

        Args:
            trace_id: Parent trace ID.
            name: Span name.
            attributes: Optional span attributes.

        Yields:
            The Span object.
        """
        span = self.start_span(trace_id, name, attributes=attributes)
        try:
            yield span
        except Exception as e:
            span.status = "ERROR"
            self.add_event(span, "exception", {"message": str(e)})
            raise
        finally:
            self.end_span(span)

    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Get a trace by ID (active or completed)."""
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]
        return next(
            (t for t in self._completed_traces if t.trace_id == trace_id),
            None
        )

    def get_recent_traces(self, limit: int = 100) -> List[TraceContext]:
        """Get recent completed traces."""
        return self._completed_traces[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        durations = [t.duration_ms for t in self._completed_traces[-100:]]

        return {
            "active_traces": len(self._active_traces),
            "completed_traces": len(self._completed_traces),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "service_name": self.service_name,
            "export_enabled": self.enable_export
        }

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return uuid.uuid4().hex

    def _export_trace(self, context: TraceContext) -> None:
        """Export trace to external system."""
        # Placeholder for export logic
        # In production, would send to Jaeger, Zipkin, or other backends
        pass


def traced(name: Optional[str] = None):
    """
    Decorator to trace a function.

    Args:
        name: Optional trace name (defaults to function name).

    Returns:
        Decorated function.
    """
    def decorator(func):
        trace_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.trace(trace_name) as ctx:
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.trace(trace_name) as ctx:
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "llm-agents") -> Tracer:
    """Get or create the global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(service_name)
    return _global_tracer


def create_tracer(
    service_name: str = "llm-agents",
    enable_export: bool = False
) -> Tracer:
    """Create a new tracer instance."""
    return Tracer(
        service_name=service_name,
        enable_export=enable_export
    )





