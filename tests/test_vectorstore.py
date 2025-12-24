"""
Unit tests for VectorStore module.

Tests VectorStoreManager for document embedding and retrieval.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

os.environ["GOOGLE_API_KEY"] = "test_api_key"


class TestVectorStoreManagerInit:
    """Tests for VectorStoreManager initialization."""
    
    def test_initialization_with_api_key(self):
        """Test initialization with API key set."""
        with patch('src.rag.vectorstore.GoogleGenerativeAIEmbeddings') as mock_google_embed, \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings') as mock_hf_embed:
            from src.rag.vectorstore import VectorStoreManager
            
            manager = VectorStoreManager()
            assert manager.embeddings is not None
            # Depending on USE_LOCAL, one of them should be called
            assert mock_google_embed.called or mock_hf_embed.called
    
    def test_initialization_no_api_key(self):
        """Test initialization fails without API key and local mode off."""
        # Temporarily remove key and force USE_LOCAL to False
        original_key = os.environ.get("GOOGLE_API_KEY")
        
        with patch('src.utils.config.GOOGLE_API_KEY', None), \
             patch('src.utils.config.USE_LOCAL', False), \
             patch('src.rag.vectorstore.GOOGLE_API_KEY', None), \
             patch('src.rag.vectorstore.USE_LOCAL', False), \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings'):
            from importlib import reload
            import src.rag.vectorstore as vs
            
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                reload(vs)
                vs.VectorStoreManager()
        
        # Restore key
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key


class TestCreateVectorStore:
    """Tests for create_vector_store method."""
    
    def test_creates_chunks(self):
        """Test that documents are split into chunks."""
        with patch('src.rag.vectorstore.GoogleGenerativeAIEmbeddings'), \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings'):
            with patch('src.rag.vectorstore.Chroma') as mock_chroma:
                mock_chroma.from_documents.return_value = MagicMock()
                
                from src.rag.vectorstore import VectorStoreManager
                
                manager = VectorStoreManager()
                documents = [
                    Document(page_content="Test content " * 100, metadata={}),
                    Document(page_content="More test content " * 50, metadata={})
                ]
                
                vectorstore, splits = manager.create_vector_store(documents)
                
                assert len(splits) > 0
                mock_chroma.from_documents.assert_called_once()
    
    def test_empty_documents(self):
        """Test handling of empty documents list."""
        with patch('src.rag.vectorstore.GoogleGenerativeAIEmbeddings'), \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings'):
            with patch('src.rag.vectorstore.Chroma') as mock_chroma:
                mock_chroma.from_documents.return_value = MagicMock()
                
                from src.rag.vectorstore import VectorStoreManager
                
                manager = VectorStoreManager()
                vectorstore, splits = manager.create_vector_store([])
                
                assert len(splits) == 0


class TestLoadVectorStore:
    """Tests for load_vector_store method."""
    
    def test_load_nonexistent_store(self):
        """Test loading nonexistent vector store."""
        with patch('src.rag.vectorstore.GoogleGenerativeAIEmbeddings'), \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings'):
            with patch('src.rag.vectorstore.os.path.exists', return_value=False):
                from src.rag.vectorstore import VectorStoreManager
                
                manager = VectorStoreManager()
                result = manager.load_vector_store()
                
                assert result is None
    
    def test_load_existing_store(self):
        """Test loading existing vector store."""
        with patch('src.rag.vectorstore.GoogleGenerativeAIEmbeddings'), \
             patch('src.rag.vectorstore.HuggingFaceEmbeddings'):
            with patch('src.rag.vectorstore.os.path.exists', return_value=True):
                with patch('src.rag.vectorstore.Chroma') as mock_chroma:
                    mock_vectorstore = MagicMock()
                    mock_chroma.return_value = mock_vectorstore
                    
                    from src.rag.vectorstore import VectorStoreManager
                    
                    manager = VectorStoreManager()
                    result = manager.load_vector_store()
                    
                    assert result is not None
                    mock_chroma.assert_called_once()
