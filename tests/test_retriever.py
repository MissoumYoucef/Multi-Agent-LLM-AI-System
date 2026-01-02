"""
Unit tests for HybridRetriever class.

Tests retriever initialization and document retrieval.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# Set env var before importing
os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization."""

    def test_default_parameters(self, sample_documents, mock_vectorstore):
        """Test initialization with default parameters."""
        from src.rag.retriever import HybridRetriever
        from src.utils.config import RETRIEVER_K, BM25_WEIGHT, VECTOR_WEIGHT

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(mock_vectorstore, sample_documents)

            assert retriever.k == RETRIEVER_K
            assert retriever.bm25_weight == BM25_WEIGHT
            assert retriever.vector_weight == VECTOR_WEIGHT

    def test_custom_parameters(self, sample_documents, mock_vectorstore):
        """Test initialization with custom parameters."""
        from src.rag.retriever import HybridRetriever

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(
                mock_vectorstore,
                sample_documents,
                k=5,
                bm25_weight=0.3,
                vector_weight=0.7
            )

            assert retriever.k == 5
            assert retriever.bm25_weight == 0.3
            assert retriever.vector_weight == 0.7

    def test_creates_ensemble_retriever(self, sample_documents, mock_vectorstore):
        """Test that ensemble retriever is created."""
        from src.rag.retriever import HybridRetriever

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(mock_vectorstore, sample_documents)

            assert retriever.retriever is not None


class TestRetrieve:
    """Tests for retrieve method."""

    def test_returns_documents(self, sample_documents, mock_vectorstore):
        """Test that retrieve returns documents."""
        from src.rag.retriever import HybridRetriever

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(mock_vectorstore, sample_documents)

            # Mock the ensemble retriever
            mock_ensemble = MagicMock()
            mock_ensemble.invoke.return_value = sample_documents[:2]
            retriever.retriever = mock_ensemble

            results = retriever.retrieve("machine learning")

            assert len(results) == 2
            mock_ensemble.invoke.assert_called_once_with("machine learning")

    def test_empty_query_returns_empty(self, sample_documents, mock_vectorstore):
        """Test that empty query returns empty list."""
        from src.rag.retriever import HybridRetriever

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(mock_vectorstore, sample_documents)

            results = retriever.retrieve("")
            assert results == []

            results = retriever.retrieve("   ")
            assert results == []

    def test_none_query_returns_empty(self, sample_documents, mock_vectorstore):
        """Test that None query returns empty list."""
        from src.rag.retriever import HybridRetriever

        with patch('src.rag.retriever.BM25Retriever.from_documents') as mock_bm25, \
             patch('src.rag.retriever.EnsembleRetriever'):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            retriever = HybridRetriever(mock_vectorstore, sample_documents)

            results = retriever.retrieve(None)
            assert results == []
