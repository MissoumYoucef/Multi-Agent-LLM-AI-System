"""
Pytest configuration and fixtures for llm-agents-rag tests.

This module provides mock objects and fixtures to test the RAG system
without making actual API calls.
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document


# Mock the GOOGLE_API_KEY for testing
os.environ["GOOGLE_API_KEY"] = "test_api_key_for_testing"


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test.pdf", "page": 0}
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Natural language processing enables computers to understand text.",
            metadata={"source": "test.pdf", "page": 2}
        ),
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke.return_value = "This is a mocked LLM response."
    return mock


@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store."""
    mock = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Relevant document 1", metadata={}),
        Document(page_content="Relevant document 2", metadata={}),
    ]
    mock.as_retriever.return_value = mock_retriever
    return mock


@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Create a temporary directory for PDF testing."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return pdf_dir


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock
