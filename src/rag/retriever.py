"""
Hybrid Retriever module.

Provides a hybrid retrieval system combining BM25 and vector search.
"""
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStore
from ..utils.config import RETRIEVER_K, BM25_WEIGHT, VECTOR_WEIGHT

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    A hybrid retriever that combines BM25 keyword search with vector similarity search.
    
    Uses an ensemble approach with configurable weights.
    """
    
    def __init__(
        self, 
        vectorstore: VectorStore, 
        documents: List[Document],
        k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore: The vector store for semantic search.
            documents: Documents for BM25 indexing.
            k: Number of documents to retrieve. Defaults to RETRIEVER_K from config.
            bm25_weight: Weight for BM25 results. Defaults to BM25_WEIGHT from config.
            vector_weight: Weight for vector results. Defaults to VECTOR_WEIGHT from config.
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k or RETRIEVER_K
        self.bm25_weight = bm25_weight or BM25_WEIGHT
        self.vector_weight = vector_weight or VECTOR_WEIGHT
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self) -> EnsembleRetriever:
        """
        Initialize the ensemble retriever with BM25 and vector search.
        
        Returns:
            An EnsembleRetriever combining both search methods.
        """
        # 1. Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )
        logger.info(f"Vector retriever initialized with k={self.k}")

        # 2. Keyword Retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.k
        logger.info(f"BM25 retriever initialized with k={self.k}")

        # 3. Ensemble Retriever (Hybrid)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.bm25_weight, self.vector_weight]
        )
        logger.info(f"Ensemble retriever initialized with weights: "
                   f"BM25={self.bm25_weight}, Vector={self.vector_weight}")
        
        return ensemble_retriever

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query.
            
        Returns:
            A list of relevant documents.
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []
        
        logger.debug(f"Retrieving for query: {query[:50]}...")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents for query")
        return docs

