"""
Freshness Tracker module.

Monitors document freshness and triggers alerts for stale content.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a tracked document."""
    path: str
    content_hash: str
    last_modified: float
    last_indexed: float
    size_bytes: int
    chunk_count: int = 0
    is_stale: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class FreshnessTracker:
    """
    Tracks document freshness and identifies stale content.

    Monitors source documents and their indexed versions
    to detect when re-indexing is needed.
    """

    def __init__(
        self,
        staleness_threshold_days: int = 30,
        check_content_hash: bool = True
    ):
        """
        Initialize freshness tracker.

        Args:
            staleness_threshold_days: Days before content is considered stale.
            check_content_hash: Whether to check content hashes for changes.
        """
        self.staleness_threshold = staleness_threshold_days * 86400  # Convert to seconds
        self.check_content_hash = check_content_hash

        self._documents: Dict[str, DocumentMetadata] = {}
        self._last_full_scan: float = 0

        logger.info(f"FreshnessTracker initialized: staleness={staleness_threshold_days} days")

    def register_document(
        self,
        path: str,
        content: Optional[str] = None,
        chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """
        Register a document for tracking.

        Args:
            path: Path to the document.
            content: Optional content for hash calculation.
            chunk_count: Number of chunks from this document.
            metadata: Optional additional metadata.

        Returns:
            DocumentMetadata for the registered document.
        """
        path_obj = Path(path)

        # Get file stats if path exists
        if path_obj.exists():
            stat = path_obj.stat()
            last_modified = stat.st_mtime
            size_bytes = stat.st_size

            if content is None and self.check_content_hash:
                try:
                    content = path_obj.read_text()
                except Exception:
                    content = ""
        else:
            last_modified = time.time()
            size_bytes = len(content) if content else 0

        content_hash = self._compute_hash(content) if content else ""

        doc_meta = DocumentMetadata(
            path=str(path),
            content_hash=content_hash,
            last_modified=last_modified,
            last_indexed=time.time(),
            size_bytes=size_bytes,
            chunk_count=chunk_count,
            metadata=metadata or {}
        )

        self._documents[str(path)] = doc_meta
        logger.debug(f"Registered document: {path}")

        return doc_meta

    def update_indexed_time(self, path: str) -> None:
        """Update the last indexed time for a document."""
        if path in self._documents:
            self._documents[path].last_indexed = time.time()
            self._documents[path].is_stale = False

    def check_freshness(self, path: str) -> Dict[str, Any]:
        """
        Check freshness status of a document.

        Args:
            path: Path to the document.

        Returns:
            Freshness status dictionary.
        """
        if path not in self._documents:
            return {"error": "Document not tracked", "path": path}

        doc = self._documents[path]
        path_obj = Path(path)

        result = {
            "path": path,
            "last_indexed": doc.last_indexed,
            "last_modified": doc.last_modified,
            "is_stale": False,
            "reasons": []
        }

        # Check if file was modified since indexing
        if path_obj.exists():
            current_mtime = path_obj.stat().st_mtime
            if current_mtime > doc.last_indexed:
                result["is_stale"] = True
                result["reasons"].append("File modified since last index")

        # Check content hash if enabled
        if self.check_content_hash and path_obj.exists():
            try:
                current_content = path_obj.read_text()
                current_hash = self._compute_hash(current_content)
                if current_hash != doc.content_hash:
                    result["is_stale"] = True
                    result["reasons"].append("Content hash changed")
            except Exception as e:
                logger.warning(f"Could not read content for hash check: {e}")

        # Check age-based staleness
        age = time.time() - doc.last_indexed
        if age > self.staleness_threshold:
            result["is_stale"] = True
            result["reasons"].append(f"Age exceeds threshold ({age/86400:.1f} days)")

        # Update document state
        doc.is_stale = result["is_stale"]

        return result

    def get_stale_documents(self) -> List[DocumentMetadata]:
        """Get list of documents that need re-indexing."""
        stale = []
        for path, doc in self._documents.items():
            status = self.check_freshness(path)
            if status.get("is_stale", False):
                stale.append(doc)
        return stale

    def get_all_documents(self) -> List[DocumentMetadata]:
        """Get all tracked documents."""
        return list(self._documents.values())

    def remove_document(self, path: str) -> bool:
        """Remove a document from tracking."""
        if path in self._documents:
            del self._documents[path]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        total = len(self._documents)
        stale_count = sum(1 for d in self._documents.values() if d.is_stale)

        return {
            "total_documents": total,
            "stale_documents": stale_count,
            "fresh_documents": total - stale_count,
            "staleness_threshold_days": self.staleness_threshold / 86400,
            "total_chunks": sum(d.chunk_count for d in self._documents.values())
        }

    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Factory function
def create_freshness_tracker(
    staleness_days: int = 30
) -> FreshnessTracker:
    """Create a freshness tracker with specified settings."""
    return FreshnessTracker(staleness_threshold_days=staleness_days)
