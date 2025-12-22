"""
Unit tests for Tracing module.

Tests Span, TraceContext, Tracer, and tracing utilities.
"""
import pytest
import os
import time
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestSpan:
    """Tests for Span dataclass."""
    
    def test_creation(self):
        """Test Span creation."""
        from src.scaling.tracing import Span
        
        span = Span(
            trace_id="trace_123",
            span_id="span_456",
            name="test_operation",
            start_time=time.time()
        )
        
        assert span.trace_id == "trace_123"
        assert span.span_id == "span_456"
        assert span.status == "OK"
    
    def test_duration_ms_running(self):
        """Test duration calculation while running."""
        from src.scaling.tracing import Span
        
        span = Span(
            trace_id="trace_1",
            span_id="span_1",
            name="test",
            start_time=time.time() - 0.1  # Started 100ms ago
        )
        
        assert span.duration_ms >= 100
    
    def test_duration_ms_completed(self):
        """Test duration calculation when completed."""
        from src.scaling.tracing import Span
        
        start = time.time()
        span = Span(
            trace_id="trace_1",
            span_id="span_1",
            name="test",
            start_time=start,
            end_time=start + 0.5
        )
        
        assert abs(span.duration_ms - 500) < 1  # ~500ms


class TestTraceContext:
    """Tests for TraceContext dataclass."""
    
    def test_creation(self):
        """Test TraceContext creation."""
        from src.scaling.tracing import TraceContext
        
        ctx = TraceContext(trace_id="trace_123")
        
        assert ctx.trace_id == "trace_123"
        assert len(ctx.spans) == 0
    
    def test_root_span_empty(self):
        """Test root_span with no spans."""
        from src.scaling.tracing import TraceContext
        
        ctx = TraceContext(trace_id="trace_1")
        assert ctx.root_span is None
    
    def test_root_span_with_spans(self):
        """Test root_span with spans."""
        from src.scaling.tracing import TraceContext, Span
        
        span = Span(
            trace_id="trace_1",
            span_id="span_1",
            name="root",
            start_time=time.time()
        )
        
        ctx = TraceContext(trace_id="trace_1", spans=[span])
        assert ctx.root_span == span


class TestTracer:
    """Tests for Tracer class."""
    
    def test_initialization(self):
        """Test tracer initialization."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer(service_name="test-service")
        assert tracer.service_name == "test-service"
    
    def test_start_trace(self):
        """Test starting a trace."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("test_operation")
        
        assert ctx is not None
        assert ctx.trace_id is not None
        assert len(ctx.spans) == 1  # Root span created
    
    def test_start_trace_with_attributes(self):
        """Test starting trace with attributes."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace(
            "test_operation",
            attributes={"key": "value"}
        )
        
        assert ctx.spans[0].attributes["key"] == "value"
    
    def test_start_span(self):
        """Test starting a span within a trace."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("parent_trace")
        span = tracer.start_span(ctx.trace_id, "child_span")
        
        assert span is not None
        assert span.name == "child_span"
        assert len(ctx.spans) == 2
    
    def test_start_span_invalid_trace(self):
        """Test starting span with invalid trace ID."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        
        with pytest.raises(ValueError):
            tracer.start_span("invalid_trace_id", "span")
    
    def test_end_span(self):
        """Test ending a span."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("test")
        span = tracer.start_span(ctx.trace_id, "child")
        
        tracer.end_span(span, status="OK", attributes={"result": "success"})
        
        assert span.end_time is not None
        assert span.status == "OK"
        assert span.attributes["result"] == "success"
    
    def test_add_event(self):
        """Test adding event to span."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("test")
        span = ctx.spans[0]
        
        tracer.add_event(span, "event_name", {"key": "value"})
        
        assert len(span.events) == 1
        assert span.events[0]["name"] == "event_name"
    
    def test_end_trace(self):
        """Test ending a trace."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("test")
        
        result = tracer.end_trace(ctx.trace_id)
        
        assert result is not None
        assert ctx.trace_id not in tracer._active_traces
        assert len(tracer._completed_traces) == 1
    
    def test_end_trace_invalid(self):
        """Test ending invalid trace."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        result = tracer.end_trace("invalid_id")
        
        assert result is None
    
    def test_trace_context_manager(self):
        """Test trace context manager."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        
        with tracer.trace("test_operation") as ctx:
            assert ctx.trace_id is not None
        
        # Trace should be ended
        assert ctx.trace_id not in tracer._active_traces
    
    def test_trace_context_manager_exception(self):
        """Test trace context manager with exception."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        
        with pytest.raises(ValueError):
            with tracer.trace("test_operation") as ctx:
                raise ValueError("Test error")
        
        # Span should be marked as error
        completed = tracer._completed_traces[-1]
        assert completed.spans[-1].status == "ERROR"
    
    def test_span_context_manager(self):
        """Test span context manager."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("parent")
        
        with tracer.span(ctx.trace_id, "child_span") as span:
            assert span.name == "child_span"
        
        assert span.end_time is not None
    
    def test_get_trace(self):
        """Test getting a trace by ID."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        ctx = tracer.start_trace("test")
        
        found = tracer.get_trace(ctx.trace_id)
        assert found is not None
        assert found.trace_id == ctx.trace_id
    
    def test_get_recent_traces(self):
        """Test getting recent traces."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer()
        
        for i in range(5):
            ctx = tracer.start_trace(f"trace_{i}")
            tracer.end_trace(ctx.trace_id)
        
        recent = tracer.get_recent_traces(limit=3)
        assert len(recent) == 3
    
    def test_get_stats(self):
        """Test getting tracer statistics."""
        from src.scaling.tracing import Tracer
        
        tracer = Tracer(service_name="test-service")
        stats = tracer.get_stats()
        
        assert stats["service_name"] == "test-service"
        assert "active_traces" in stats
        assert "completed_traces" in stats


class TestTracedDecorator:
    """Tests for @traced decorator."""
    
    def test_decorator_traces_function(self):
        """Test that decorator traces function."""
        from src.scaling.tracing import traced, get_tracer
        
        @traced("test_func")
        def my_function():
            return "result"
        
        result = my_function()
        assert result == "result"
        
        # Check trace was created
        tracer = get_tracer()
        assert len(tracer._completed_traces) >= 1


class TestGlobalTracer:
    """Tests for global tracer functions."""
    
    def test_get_tracer(self):
        """Test getting global tracer."""
        from src.scaling.tracing import get_tracer
        
        tracer = get_tracer()
        assert tracer is not None
    
    def test_create_tracer(self):
        """Test creating new tracer."""
        from src.scaling.tracing import create_tracer
        
        tracer = create_tracer(service_name="custom-service")
        assert tracer.service_name == "custom-service"
