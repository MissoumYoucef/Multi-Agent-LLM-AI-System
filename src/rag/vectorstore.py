import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ..utils.config import (
    GOOGLE_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    VECTOR_STORE_PATH, USE_LOCAL, LOCAL_EMBEDDING_MODEL
)
from .freshness_tracker import FreshnessTracker

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the vector store for RAG.

    Handles embedding initialization, document splitting, and vector store creation.
    Integrates with FreshnessTracker to track document updates.
    """

    def __init__(self, freshness_tracker: Optional[FreshnessTracker] = None):
        """
        Initialize the vector store manager.

        Args:
            freshness_tracker: Optional tracker for document freshness.
        """
        self.freshness_tracker = freshness_tracker

        if USE_LOCAL:
            logger.info(f"Using local embeddings: {LOCAL_EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL
            )
        else:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set and USE_LOCAL is false")

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
        self.vector_store_path = VECTOR_STORE_PATH

    def create_vector_store(self, documents: List[Document]):
        """
        Create a vector store from documents.

        Args:
            documents: List of documents to index.

        Returns:
            Tuple of (vectorstore, splits).
        """
        # Register documents with freshness tracker if active
        if self.freshness_tracker:
            processed_sources = set()
            for doc in documents:
                source = doc.metadata.get("source")
                if source and source not in processed_sources:
                    self.freshness_tracker.register_document(path=source)
                    processed_sources.add(source)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} chunks from {len(documents)} documents.")

        if os.path.exists(self.vector_store_path):
             # For this implementation, we allow overwriting or appending.
             # Chroma will handle persistence.
             pass

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        logger.info(f"Vector store updated at {self.vector_store_path}")

        # Update indexed time for tracked documents
        if self.freshness_tracker:
            for source in processed_sources:
                self.freshness_tracker.update_indexed_time(source)

        return vectorstore, splits

    def load_vector_store(self):
        """
        Load an existing vector store.

        Returns:
            VectorStore if exists, None otherwise.
        """
        if not os.path.exists(self.vector_store_path):
            return None

        vectorstore = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )
        return vectorstore
