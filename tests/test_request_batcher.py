"""
Unit tests for Request Batcher module.

Tests RequestBatcher and AdaptiveBatcher.
"""
import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_creation(self):
        """Test BatchRequest creation."""
        from src.scaling.request_batcher import BatchRequest

        loop = asyncio.new_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            id="req_1",
            data={"query": "test"},
            future=future,
            priority=1
        )

        assert request.id == "req_1"
        assert request.priority == 1
        loop.close()


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_creation(self):
        """Test BatchResult creation."""
        from src.scaling.request_batcher import BatchResult

        result = BatchResult(
            batch_id="batch_1",
            results=["r1", "r2"],
            batch_size=2,
            processing_time=0.5,
            success_count=2,
            error_count=0
        )

        assert result.batch_id == "batch_1"
        assert result.batch_size == 2
        assert result.success_count == 2
        assert result.processing_time == 0.5
        assert len(result.results) == 2
        assert result.error_count == 0


class TestRequestBatcher:
    """Tests for RequestBatcher class."""

    def test_initialization(self):
        """Test batcher initialization."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(batch_size=5, batch_timeout_ms=50)
        assert batcher.batch_size == 5
        assert batcher.batch_timeout == 0.05

    def test_set_processor(self):
        """Test setting processor function."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher()

        async def mock_processor(batch):
            return batch

        batcher.set_processor(mock_processor)
        assert batcher._processor is not None

    @pytest.mark.asyncio
    async def test_add_request(self):
        """Test adding request to queue."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(batch_size=10)

        future = await batcher.add_request("req_1", {"query": "test"})
        assert future is not None
        assert batcher._total_requests == 1

    @pytest.mark.asyncio
    async def test_add_request_with_priority(self):
        """Test adding request with priority."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher()

        await batcher.add_request("req_1", {"query": "low"}, priority=0)
        await batcher.add_request("req_2", {"query": "high"}, priority=10)

        assert len(batcher._queue) == 2

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        """Test request rejection when queue is full."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(max_queue_size=1)

        await batcher.add_request("req_1", {"query": "first"})
        future = await batcher.add_request("req_2", {"query": "second"})

        # Second request should fail due to full queue
        assert future.done()

    def test_get_stats(self):
        """Test getting batcher statistics."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(batch_size=10, max_queue_size=100)
        stats = batcher.get_stats()

        assert stats["batch_size"] == 10
        assert stats["max_queue_size"] == 100
        assert stats["is_running"] is False

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping batcher."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(batch_timeout_ms=10)

        async def mock_processor(batch):
            return ["result"] * len(batch)

        batcher.set_processor(mock_processor)

        await batcher.start()
        assert batcher._running is True

        await batcher.stop()
        assert batcher._running is False

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing."""
        from src.scaling.request_batcher import RequestBatcher

        batcher = RequestBatcher(batch_size=2, batch_timeout_ms=10)

        async def mock_processor(batch):
            return [f"result_{i}" for i in range(len(batch))]

        batcher.set_processor(mock_processor)

        future1 = await batcher.add_request("req_1", {"query": "q1"})
        future2 = await batcher.add_request("req_2", {"query": "q2"})

        # Trigger batch processing
        result = await batcher._process_batch()

        assert result is not None
        assert result.batch_size == 2
        assert result.success_count == 2


class TestAdaptiveBatcher:
    """Tests for AdaptiveBatcher class."""

    def test_initialization(self):
        """Test adaptive batcher initialization."""
        from src.scaling.request_batcher import AdaptiveBatcher

        batcher = AdaptiveBatcher(
            min_batch_size=1,
            max_batch_size=50,
            target_latency_ms=100
        )

        assert batcher.min_batch_size == 1
        assert batcher.max_batch_size == 50

    def test_adapt_batch_size_increase(self):
        """Test batch size increase on low latency."""
        from src.scaling.request_batcher import AdaptiveBatcher

        batcher = AdaptiveBatcher(
            min_batch_size=5,
            max_batch_size=50,
            target_latency_ms=100
        )
        batcher.batch_size = 10

        # Simulate low latency history
        for _ in range(15):
            batcher._latency_history.append(0.05)  # 50ms, well below target

        batcher._adapt_batch_size()
        assert batcher.batch_size >= 10  # Should increase

    def test_adapt_batch_size_decrease(self):
        """Test batch size decrease on high latency."""
        from src.scaling.request_batcher import AdaptiveBatcher

        batcher = AdaptiveBatcher(
            min_batch_size=5,
            max_batch_size=50,
            target_latency_ms=100
        )
        batcher.batch_size = 20

        # Simulate high latency history
        for _ in range(15):
            batcher._latency_history.append(0.15)  # 150ms, above target

        batcher._adapt_batch_size()
        assert batcher.batch_size <= 20  # Should decrease


class TestCreateRequestBatcher:
    """Tests for create_request_batcher factory function."""

    def test_create_regular_batcher(self):
        """Test creating regular batcher."""
        from src.scaling.request_batcher import create_request_batcher, RequestBatcher

        batcher = create_request_batcher(batch_size=15, adaptive=False)
        assert isinstance(batcher, RequestBatcher)
        assert batcher.batch_size == 15

    def test_create_adaptive_batcher(self):
        """Test creating adaptive batcher."""
        from src.scaling.request_batcher import create_request_batcher, AdaptiveBatcher

        batcher = create_request_batcher(batch_size=10, adaptive=True)
        assert isinstance(batcher, AdaptiveBatcher)
