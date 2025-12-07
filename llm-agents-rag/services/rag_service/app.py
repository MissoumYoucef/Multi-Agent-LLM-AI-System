"""
RAG Service - FastAPI application for document retrieval.
This service manages the vector store and provides a REST API for retrieval.
"""
import os
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.utils.config import PDF_DATA_PATH

# Global state for the retriever (loaded once at startup)
retriever: Optional[HybridRetriever] = None


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
    global retriever
    print("--- RAG Service Starting ---")
    
    # Load documents
    loader = PDFLoader(PDF_DATA_PATH)
    documents = loader.load_documents()
    
    if not documents:
        print("WARNING: No documents found. Service will return empty results.")
        retriever = None
    else:
        # Create vector store
        vs_manager = VectorStoreManager()
        vectorstore, splits = vs_manager.create_vector_store(documents)
        retriever = HybridRetriever(vectorstore, splits)
        print(f"RAG pipeline ready with {len(splits)} chunks.")
    
    yield  # Application runs
    
    print("--- RAG Service Shutting Down ---")


app = FastAPI(
    title="RAG Service",
    description="Document retrieval service using hybrid search (BM25 + Vector)",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "retriever_loaded": retriever is not None}


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
