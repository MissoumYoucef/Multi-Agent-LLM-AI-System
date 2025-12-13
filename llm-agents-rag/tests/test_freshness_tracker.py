"""
Unit tests for Freshness Tracker module.

Tests document freshness monitoring and stale content detection.
"""
import pytest
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""
    
    def test_creation(self):
        """Test DocumentMetadata creation."""
        from src.rag.freshness_tracker import DocumentMetadata
        
        meta = DocumentMetadata(
            path="/path/to/doc.pdf",
            content_hash="abc123",
            last_modified=time.time(),
            last_indexed=time.time(),
            size_bytes=1024
        )
        
        assert meta.path == "/path/to/doc.pdf"
        assert meta.content_hash == "abc123"
        assert meta.is_stale is False


class TestFreshnessTrackerInit:
    """Tests for FreshnessTracker initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        assert tracker.staleness_threshold == 30 * 86400  # 30 days in seconds
    
    def test_custom_threshold(self):
        """Test initialization with custom threshold."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker(staleness_threshold_days=7)
        assert tracker.staleness_threshold == 7 * 86400


class TestRegisterDocument:
    """Tests for register_document method."""
    
    def test_register_with_content(self):
        """Test registering document with content."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        meta = tracker.register_document(
            path="/virtual/doc.txt",
            content="Test content here",
            chunk_count=5
        )
        
        assert meta.path == "/virtual/doc.txt"
        assert meta.chunk_count == 5
        assert meta.content_hash != ""
    
    def test_register_real_file(self):
        """Test registering real file from filesystem."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            tracker = FreshnessTracker()
            meta = tracker.register_document(path=temp_path)
            
            assert meta.path == temp_path
            assert meta.size_bytes > 0
        finally:
            os.unlink(temp_path)
    
    def test_register_with_metadata(self):
        """Test registering with custom metadata."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        meta = tracker.register_document(
            path="/virtual/doc.txt",
            content="Content",
            metadata={"source": "api", "version": 1}
        )
        
        assert meta.metadata["source"] == "api"
        assert meta.metadata["version"] == 1


class TestUpdateIndexedTime:
    """Tests for update_indexed_time method."""
    
    def test_update_resets_stale(self):
        """Test that updating indexed time resets stale flag."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        tracker.register_document("/virtual/doc.txt", "Content")
        
        # Manually mark as stale
        tracker._documents["/virtual/doc.txt"].is_stale = True
        
        tracker.update_indexed_time("/virtual/doc.txt")
        
        assert tracker._documents["/virtual/doc.txt"].is_stale is False


class TestCheckFreshness:
    """Tests for check_freshness method."""
    
    def test_untracked_document(self):
        """Test checking freshness of untracked document."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        result = tracker.check_freshness("/unknown/doc.txt")
        
        assert "error" in result
    
    def test_fresh_document(self):
        """Test checking freshness of fresh document."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker(staleness_threshold_days=30)
        tracker.register_document("/virtual/doc.txt", "Content")
        
        result = tracker.check_freshness("/virtual/doc.txt")
        
        assert result["is_stale"] is False
    
    def test_stale_by_age(self):
        """Test document is stale due to age."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker(staleness_threshold_days=1)
        tracker.register_document("/virtual/doc.txt", "Content")
        
        # Manually age the document
        tracker._documents["/virtual/doc.txt"].last_indexed = time.time() - (2 * 86400)
        
        result = tracker.check_freshness("/virtual/doc.txt")
        
        assert result["is_stale"] is True
        assert any("age" in reason.lower() for reason in result["reasons"])
    
    def test_stale_by_modification(self):
        """Test document is stale due to file modification."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Original content")
            temp_path = f.name
        
        try:
            tracker = FreshnessTracker()
            tracker.register_document(temp_path)
            
            # Backdate the indexed time
            tracker._documents[temp_path].last_indexed = time.time() - 100
            
            # Modify the file
            time.sleep(0.1)
            with open(temp_path, 'w') as f:
                f.write("Modified content")
            os.utime(temp_path, None)  # Touch to update mtime
            
            result = tracker.check_freshness(temp_path)
            
            # File was modified after indexing
            assert result["is_stale"] is True
        finally:
            os.unlink(temp_path)


class TestGetStaleDocuments:
    """Tests for get_stale_documents method."""
    
    def test_returns_stale_only(self):
        """Test that only stale documents are returned."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker(staleness_threshold_days=1)
        
        # Register fresh document
        tracker.register_document("/virtual/fresh.txt", "Content 1")
        
        # Register stale document
        tracker.register_document("/virtual/stale.txt", "Content 2")
        tracker._documents["/virtual/stale.txt"].last_indexed = time.time() - (2 * 86400)
        
        stale = tracker.get_stale_documents()
        
        assert len(stale) == 1
        assert stale[0].path == "/virtual/stale.txt"


class TestGetAllDocuments:
    """Tests for get_all_documents method."""
    
    def test_returns_all(self):
        """Test that all documents are returned."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        tracker.register_document("/virtual/doc1.txt", "Content 1")
        tracker.register_document("/virtual/doc2.txt", "Content 2")
        
        all_docs = tracker.get_all_documents()
        
        assert len(all_docs) == 2


class TestRemoveDocument:
    """Tests for remove_document method."""
    
    def test_remove_existing(self):
        """Test removing existing document."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        tracker.register_document("/virtual/doc.txt", "Content")
        
        result = tracker.remove_document("/virtual/doc.txt")
        
        assert result is True
        assert len(tracker.get_all_documents()) == 0
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent document."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        result = tracker.remove_document("/unknown/doc.txt")
        
        assert result is False


class TestGetStats:
    """Tests for get_stats method."""
    
    def test_stats_structure(self):
        """Test stats structure."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        stats = tracker.get_stats()
        
        assert "total_documents" in stats
        assert "stale_documents" in stats
        assert "fresh_documents" in stats
        assert "staleness_threshold_days" in stats
        assert "total_chunks" in stats
    
    def test_stats_values(self):
        """Test stats values are correct."""
        from src.rag.freshness_tracker import FreshnessTracker
        
        tracker = FreshnessTracker()
        tracker.register_document("/virtual/doc1.txt", "Content 1", chunk_count=5)
        tracker.register_document("/virtual/doc2.txt", "Content 2", chunk_count=3)
        
        stats = tracker.get_stats()
        
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 8


class TestCreateFreshnessTracker:
    """Tests for create_freshness_tracker factory function."""
    
    def test_factory_function(self):
        """Test factory function."""
        from src.rag.freshness_tracker import create_freshness_tracker
        
        tracker = create_freshness_tracker(staleness_days=14)
        assert tracker.staleness_threshold == 14 * 86400
