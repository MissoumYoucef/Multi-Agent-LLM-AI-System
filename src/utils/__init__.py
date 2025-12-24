# utils package
"""
Utility modules for LLM-Agents-RAG.

Modules:
    - cache: Multi-tier response caching with LRU and optional Redis
    - cost_controller: Budget management and cost tracking
    - token_manager: Token counting and context truncation
    - config: Configuration constants and environment variables
"""

from .cache import ResponseCache, LRUCache, CacheEntry, cached
from .cost_controller import (
    CostController,
    create_cost_controller,
    BudgetPeriod,
    UsageRecord,
    BudgetConfig
)
from .token_manager import TokenManager, create_token_manager
from .config import (
    GOOGLE_API_KEY,
    USE_LOCAL,
    LOCAL_LLM_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    VECTOR_STORE_PATH,
    PDF_DATA_PATH,
    MEMORY_BUFFER_SIZE,
    MEMORY_SUMMARY_THRESHOLD,
    MEMORY_SESSION_TTL,
    CACHE_ENABLED,
    REDIS_URL,
    CACHE_TTL_SECONDS,
    DAILY_BUDGET_USD,
    COST_ALERT_THRESHOLD,
    FALLBACK_MODELS,
    STALENESS_THRESHOLD_DAYS
)

__all__ = [
    # Cache
    "ResponseCache",
    "LRUCache",
    "CacheEntry",
    "cached",
    # Cost Controller
    "CostController",
    "create_cost_controller",
    "BudgetPeriod",
    "UsageRecord",
    "BudgetConfig",
    # Token Manager
    "TokenManager",
    "create_token_manager",
    # Config values
    "GOOGLE_API_KEY",
    "USE_LOCAL",
    "LOCAL_LLM_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVER_K",
    "VECTOR_STORE_PATH",
    "PDF_DATA_PATH",
    "MEMORY_BUFFER_SIZE",
    "MEMORY_SUMMARY_THRESHOLD",
    "MEMORY_SESSION_TTL",
    "CACHE_ENABLED",
    "REDIS_URL",
    "CACHE_TTL_SECONDS",
    "DAILY_BUDGET_USD",
    "COST_ALERT_THRESHOLD",
    "FALLBACK_MODELS",
    "STALENESS_THRESHOLD_DAYS",
]
