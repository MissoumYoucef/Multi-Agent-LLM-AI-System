"""
RAG Service - FastAPI application for document retrieval.
This service manages the vector store and provides a REST API for retrieval.
"""
import os
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.utils.config import PDF_DATA_PATH
from src.rag.freshness_tracker import create_freshness_tracker
from src.utils.tracing import setup_tracing

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    METRICS_SUPPORT = True
except ImportError:
    METRICS_SUPPORT = False

# Global state
retriever: Optional[HybridRetriever] = None
freshness_tracker: Optional[Any] = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class DocumentChunk(BaseModel):
    content: str
    metadata: dict


class RetrieveResponse(BaseModel):
    query: str
    documents: List[DocumentChunk]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline on startup."""
    global retriever, freshness_tracker
    print("--- RAG Service Starting ---")
    
    # Initialize Freshness Tracker
    freshness_tracker = create_freshness_tracker(staleness_days=30)
    
    # Initialize Vector Store Manager
    vs_manager = VectorStoreManager(freshness_tracker=freshness_tracker)
    
    # We always need the documents for the BM25 part of HybridRetriever
    print("Loading source documents...")
    loader = PDFLoader(PDF_DATA_PATH, load_all=True)
    documents = loader.load_documents()
    
    if not documents:
        print("WARNING: No documents found. Service will return empty results.")
        retriever = None
    else:
        # Check if vector store already exists
        vectorstore = vs_manager.load_vector_store()
        
        if vectorstore:
            print("Loading existing vector store...")
            # We still need splits for the hybrid retriever
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            retriever = HybridRetriever(vectorstore, splits)
        else:
            print("Vector store not found. Building from source documents...")
            # Create vector store
            vectorstore, splits = vs_manager.create_vector_store(documents)
            retriever = HybridRetriever(vectorstore, splits)
            
        print(f"RAG pipeline ready.")
    
    yield  # Application runs
    
    print("--- RAG Service Shutting Down ---")


app = FastAPI(
    title="RAG Service",
    description="Document retrieval service using hybrid search (BM25 + Vector)",
    version="1.0.0",
    lifespan=lifespan
)

if METRICS_SUPPORT:
    Instrumentator().instrument(app).expose(app)

# Setup tracing
setup_tracing(app, service_name="rag-service")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "retriever_loaded": retriever is not None}


@app.get("/stats")
async def get_stats():
    """Get document freshness statistics."""
    if freshness_tracker is None:
         return {"error": "Freshness tracker not initialized"}
    return freshness_tracker.get_stats()


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Retrieve relevant document chunks for a given query.
    Uses hybrid search combining BM25 (keyword) and vector similarity.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized. Check if documents are loaded.")
    
    docs = retriever.retrieve(request.query)
    
    # Convert to response format
    doc_chunks = [
        DocumentChunk(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]
    
    return RetrieveResponse(query=request.query, documents=doc_chunks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
