"""
Unit tests for PDFLoader class.

Tests document loading and cleaning functionality.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Set env var before importing
os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestPDFLoaderInit:
    """Tests for PDFLoader initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path")
        assert loader.directory_path == "/some/path"
        assert loader.load_all is False

    def test_init_load_all(self):
        """Test initialization with load_all=True."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path", load_all=True)
        assert loader.load_all is True


class TestLoadDocuments:
    """Tests for load_documents method."""

    def test_nonexistent_directory(self):
        """Should return empty list for nonexistent directory."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/nonexistent/path")
        docs = loader.load_documents()
        assert docs == []

    def test_empty_directory(self, temp_pdf_dir):
        """Should return empty list for directory with no PDFs."""
        from src.rag.loader import PDFLoader

        # load_all=False expects text.pdf specifically
        loader = PDFLoader(str(temp_pdf_dir), load_all=False)
        docs = loader.load_documents()
        assert docs == []


class TestCleanDocuments:
    """Tests for clean_documents method."""

    def test_removes_empty_content(self):
        """Should remove documents with empty content."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path")

        docs = [
            Document(page_content="Valid content with sufficient length to pass filter.", metadata={}),
            Document(page_content="", metadata={}),
            Document(page_content="   ", metadata={}),
        ]

        cleaned = loader.clean_documents(docs)
        assert len(cleaned) == 1
        assert cleaned[0].page_content == "Valid content with sufficient length to pass filter."

    def test_removes_short_content(self):
        """Should remove documents with very short content."""
        from src.rag.loader import PDFLoader, MIN_CONTENT_LENGTH

        loader = PDFLoader("/some/path")

        short_content = "x" * (MIN_CONTENT_LENGTH - 1)
        long_content = "x" * (MIN_CONTENT_LENGTH + 10)

        docs = [
            Document(page_content=short_content, metadata={}),
            Document(page_content=long_content, metadata={}),
        ]

        cleaned = loader.clean_documents(docs)
        assert len(cleaned) == 1
        assert len(cleaned[0].page_content) >= MIN_CONTENT_LENGTH

    def test_removes_duplicates(self):
        """Should remove duplicate documents."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path")

        content = "This is valid content that appears multiple times in the document set."
        docs = [
            Document(page_content=content, metadata={"page": 0}),
            Document(page_content=content, metadata={"page": 1}),
            Document(page_content=content, metadata={"page": 2}),
        ]

        cleaned = loader.clean_documents(docs)
        assert len(cleaned) == 1

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path")

        docs = [
            Document(page_content="  Content with whitespace around it!  " * 5, metadata={}),
        ]

        cleaned = loader.clean_documents(docs)
        assert cleaned[0].page_content.startswith("Content")
        assert cleaned[0].page_content.endswith("!")

    def test_preserves_order(self):
        """Should preserve document order."""
        from src.rag.loader import PDFLoader

        loader = PDFLoader("/some/path")

        docs = [
            Document(page_content="First document with valid content length for testing.", metadata={}),
            Document(page_content="Second document also has enough content length to pass.", metadata={}),
            Document(page_content="Third document completes the test with sufficient length.", metadata={}),
        ]

        cleaned = loader.clean_documents(docs)
        assert len(cleaned) == 3
        assert "First" in cleaned[0].page_content
        assert "Second" in cleaned[1].page_content
        assert "Third" in cleaned[2].page_content
