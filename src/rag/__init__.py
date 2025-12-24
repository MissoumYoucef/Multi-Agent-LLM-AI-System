# rag package
"""
RAG (Retrieval-Augmented Generation) module.

Provides document loading, vectorstore management, hybrid retrieval,
and document freshness tracking.
"""

from . import vectorstore
from .freshness_tracker import (
    FreshnessTracker,
    create_freshness_tracker,
    DocumentMetadata
)

__all__ = [
    "vectorstore",
    "FreshnessTracker",
    "create_freshness_tracker",
    "DocumentMetadata",
]
