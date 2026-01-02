"""
Configuration module for LLM-Agents-RAG.
Loads environment variables and defines configuration constants.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    """
    Get a required environment variable or raise ValueError.
    
    Args:
        name: Name of the environment variable.
        
    Returns:
        The value of the environment variable.
        
    Raises:
        ValueError: If the environment variable is not set.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable '{name}' is not set. "
                        f"Please set it in your .env file or environment.")
    return value


def get_optional_env(name: str, default: str) -> str:
    """
    Get an optional environment variable with a default value.
    
    Args:
        name: Name of the environment variable.
        default: Default value if not set.
        
    Returns:
        The value of the environment variable or the default.
    """
    return os.getenv(name, default)


# API Configuration
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

# Local Model Configuration
# Use local if key is missing/empty OR if USE_LOCAL is explicitly true
USE_LOCAL: bool = (not GOOGLE_API_KEY or not GOOGLE_API_KEY.strip()) or os.getenv("USE_LOCAL", "false").lower() == "true"
LOCAL_LLM_MODEL: str = get_optional_env("LOCAL_LLM_MODEL", "llama3.2:1b")
LOCAL_EMBEDDING_MODEL: str = get_optional_env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_BASE_URL: str = get_optional_env("OLLAMA_BASE_URL", "http://localhost:11434")


# Model Configuration
LLM_MODEL: str = get_optional_env("LLM_MODEL", "gemini-pro")
EMBEDDING_MODEL: str = get_optional_env("EMBEDDING_MODEL", "models/embedding-001")

# Logging Configuration
LOG_LEVEL: str = get_optional_env("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

# RAG Configuration
CHUNK_SIZE: int = int(get_optional_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(get_optional_env("CHUNK_OVERLAP", "200"))
RETRIEVER_K: int = int(get_optional_env("RETRIEVER_K", "3"))
BM25_WEIGHT: float = float(get_optional_env("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT: float = float(get_optional_env("VECTOR_WEIGHT", "0.5"))

# Path Configuration
VECTOR_STORE_PATH: str = get_optional_env("VECTOR_STORE_PATH", "./chroma_db")
PDF_DATA_PATH: str = get_optional_env("PDF_DATA_PATH", "./data/pdfs")

# Memory Configuration
MEMORY_BUFFER_SIZE: int = int(get_optional_env("MEMORY_BUFFER_SIZE", "10"))
MEMORY_SUMMARY_THRESHOLD: int = int(get_optional_env("MEMORY_SUMMARY_THRESHOLD", "20"))
MEMORY_SESSION_TTL: int = int(get_optional_env("MEMORY_SESSION_TTL", "3600"))

# Caching Configuration
CACHE_ENABLED: bool = get_optional_env("CACHE_ENABLED", "true").lower() == "true"
REDIS_URL: str = get_optional_env("REDIS_URL", "redis://redis:6379")
CACHE_TTL_SECONDS: int = int(get_optional_env("CACHE_TTL_SECONDS", "3600"))

# Cost Control
DAILY_BUDGET_USD: float = float(get_optional_env("DAILY_BUDGET_USD", "10.0"))
COST_ALERT_THRESHOLD: float = float(get_optional_env("COST_ALERT_THRESHOLD", "0.8"))

# Fallback Models
FALLBACK_MODELS: list = get_optional_env("FALLBACK_MODELS", "gemini-pro,gemini-flash").split(",")

# Drift Detection
DRIFT_ALERT_THRESHOLD: float = float(get_optional_env("DRIFT_ALERT_THRESHOLD", "0.3"))
STALENESS_THRESHOLD_DAYS: int = int(get_optional_env("STALENESS_THRESHOLD_DAYS", "30"))

# Tracing Configuration
TRACING_ENABLED: bool = get_optional_env("TRACING_ENABLED", "false").lower() == "true"
TRACING_SERVICE_NAME: str = get_optional_env("TRACING_SERVICE_NAME", "llm-agents")
