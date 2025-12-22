"""
Request Batcher module.
Provides request batching for efficient API usage and throughput optimization.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A request in the batch queue."""
    id: str
    data: Dict[str, Any]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    results: List[Any]
    batch_size: int
    processing_time: float
    success_count: int
    error_count: int


class RequestBatcher:
    """
    Batches incoming requests for efficient processing.
    Collects requests within a time window or until batch size
    is reached, then processes them together.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout_ms: int = 100,
        max_queue_size: int = 1000
    ):
        """
        Initialize request batcher.
        
        Args:
            batch_size: Maximum requests per batch.
            batch_timeout_ms: Max wait time before processing batch.
            max_queue_size: Maximum pending requests in queue.
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000  # Convert to seconds
        self.max_queue_size = max_queue_size
        
        self._queue: deque[BatchRequest] = deque(maxlen=max_queue_size)
        self._processor: Optional[Callable] = None
        self._running = False
        self._batch_count = 0
        self._total_requests = 0
        self._lock = asyncio.Lock()
        
        logger.info(f"RequestBatcher initialized: size={batch_size}, timeout={batch_timeout_ms}ms")
    
    def set_processor(
        self,
        processor: Callable[[List[Dict[str, Any]]], Awaitable[List[Any]]]
    ) -> None:
        """
        Set the batch processor function.
        
        Args:
            processor: Async function that processes a batch of requests.
        """
        self._processor = processor
    
    async def add_request(
        self,
        request_id: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> asyncio.Future:
        """
        Add a request to the batch queue.
        
        Args:
            request_id: Unique request identifier.
            data: Request data.
            priority: Priority (higher = processed first).
            
        Returns:
            Future that will contain the result.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        request = BatchRequest(
            id=request_id,
            data=data,
            future=future,
            priority=priority
        )
        
        async with self._lock:
            if len(self._queue) >= self.max_queue_size:
                future.set_exception(Exception("Queue full"))
                return future
            
            self._queue.append(request)
            self._total_requests += 1
            
            # Check if we should process immediately
            if len(self._queue) >= self.batch_size:
                asyncio.create_task(self._process_batch())
        
        return future
    
    async def start(self) -> None:
        """Start the background batch processing loop."""
        if self._running:
            return
        
        self._running = True
        asyncio.create_task(self._processing_loop())
        logger.info("Batch processor started")
    
    async def stop(self) -> None:
        """Stop the batch processing loop."""
        self._running = False
        # Process remaining requests
        while self._queue:
            await self._process_batch()
        logger.info("Batch processor stopped")
    
    async def _processing_loop(self) -> None:
        """Background loop that processes batches on timeout."""
        while self._running:
            await asyncio.sleep(self.batch_timeout)
            if self._queue:
                await self._process_batch()
    
    async def _process_batch(self) -> Optional[BatchResult]:
        """Process current batch of requests."""
        if not self._processor:
            logger.error("No processor set")
            return None
        
        async with self._lock:
            if not self._queue:
                return None
            
            # Get batch of requests (respecting priority)
            batch_requests = []
            for _ in range(min(self.batch_size, len(self._queue))):
                if self._queue:
                    batch_requests.append(self._queue.popleft())
            
            # Sort by priority
            batch_requests.sort(key=lambda r: -r.priority)
        
        if not batch_requests:
            return None
        
        self._batch_count += 1
        batch_id = f"batch_{self._batch_count}"
        start_time = time.time()
        
        try:
            # Extract data for processing
            batch_data = [r.data for r in batch_requests]
            
            # Process batch
            results = await self._processor(batch_data)
            
            # Set results on futures
            success_count = 0
            error_count = 0
            
            for i, request in enumerate(batch_requests):
                try:
                    if i < len(results):
                        request.future.set_result(results[i])
                        success_count += 1
                    else:
                        request.future.set_exception(
                            Exception("No result for request")
                        )
                        error_count += 1
                except Exception as e:
                    if not request.future.done():
                        request.future.set_exception(e)
                    error_count += 1
            
            processing_time = time.time() - start_time
            
            result = BatchResult(
                batch_id=batch_id,
                results=results,
                batch_size=len(batch_requests),
                processing_time=processing_time,
                success_count=success_count,
                error_count=error_count
            )
            
            logger.debug(f"Batch {batch_id} processed: "
                        f"{success_count} success, {error_count} errors, "
                        f"{processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Set error on all pending futures
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "queue_size": len(self._queue),
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "batch_count": self._batch_count,
            "total_requests": self._total_requests,
            "is_running": self._running
        }


class AdaptiveBatcher(RequestBatcher):
    """
    Request batcher with adaptive batch sizing.
    
    Adjusts batch size based on latency and throughput metrics.
    """
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 50,
        target_latency_ms: float = 100,
        **kwargs
    ):
        """
        Initialize adaptive batcher.
        
        Args:
            min_batch_size: Minimum batch size.
            max_batch_size: Maximum batch size.
            target_latency_ms: Target latency in milliseconds.
            **kwargs: Additional arguments for RequestBatcher.
        """
        super().__init__(batch_size=min_batch_size, **kwargs)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency_ms / 1000
        
        self._latency_history: deque = deque(maxlen=100)
    
    async def _process_batch(self) -> Optional[BatchResult]:
        """Process batch and adapt size based on latency."""
        result = await super()._process_batch()
        
        if result:
            self._latency_history.append(result.processing_time)
            self._adapt_batch_size()
        
        return result
    
    def _adapt_batch_size(self) -> None:
        """Adjust batch size based on recent latency."""
        if len(self._latency_history) < 10:
            return
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        if avg_latency > self.target_latency * 1.2:
            # Latency too high, reduce batch size
            self.batch_size = max(
                self.min_batch_size,
                int(self.batch_size * 0.8)
            )
        elif avg_latency < self.target_latency * 0.8:
            # Latency low, can increase batch size
            self.batch_size = min(
                self.max_batch_size,
                int(self.batch_size * 1.2)
            )


# Factory function
def create_request_batcher(
    batch_size: int = 10,
    batch_timeout_ms: int = 100,
    adaptive: bool = False
) -> RequestBatcher:
    """Create a request batcher."""
    if adaptive:
        return AdaptiveBatcher(
            min_batch_size=1,
            max_batch_size=batch_size * 5,
            batch_timeout_ms=batch_timeout_ms
        )
    return RequestBatcher(
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms
    )
