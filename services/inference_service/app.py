"""
Inference Service - FastAPI application for LLM agents.
This service orchestrates the chatbot, solver, and analyzer agents.
It calls the RAG service to retrieve context.
"""
import os
import httpx
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.orchestrator import Orchestrator
from src.utils.config import CACHE_ENABLED, DAILY_BUDGET_USD
from src.utils.tracing import setup_tracing

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    METRICS_SUPPORT = True
except ImportError:
    METRICS_SUPPORT = False

# Configuration
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")

# Global state
orchestrator: Optional[Orchestrator] = None


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default-session"


class ChatResponse(BaseModel):
    query: str
    solution: str
    analysis: str
    context_length: int
    quality_score: float
    cache_hit: bool
    reflection_history: Optional[list] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Orchestrator on startup."""
    global orchestrator
    print("--- Inference Service Starting ---")
    
    # In a microservices architecture, the orchestrator needs a retriever but retrieval
    # is handled by the separate RAG service. We create a RemoteRetriever that makes
    # HTTP calls to the RAG service's /retrieve endpoint, converting the response back
    # into LangChain Document objects for the orchestrator to use.
    
    class RemoteRetriever:
        """Proxy retriever that calls the RAG microservice for document retrieval."""
        def __init__(self, url):
            self.url = url
            
        def retrieve(self, query):
            """Retrieve documents from RAG service via HTTP."""
            import requests
            response = requests.post(f"{self.url}/retrieve", json={"query": query, "top_k": 5})
            response.raise_for_status()
            data = response.json()
            from langchain_core.documents import Document
            return [Document(page_content=d["content"], metadata=d["metadata"]) for d in data["documents"]]

    retriever = RemoteRetriever(RAG_SERVICE_URL)
    
    orchestrator = Orchestrator(
        retriever=retriever,
        enable_guardrails=True,
        enable_reflection=True,
        enable_react=False,
        enable_caching=CACHE_ENABLED,
        enable_cost_control=True,
        enable_memory=True,
        enable_evaluation=True,
        daily_budget=DAILY_BUDGET_USD
    )
    
    print("Orchestrator initialized.")
    yield
    
    print("--- Inference Service Shutting Down ---")


app = FastAPI(
    title="Inference Service",
    description="LLM Agent orchestration service",
    version="1.0.0",
    lifespan=lifespan
)

if METRICS_SUPPORT:
    Instrumentator().instrument(app).expose(app)

# Setup tracing
setup_tracing(app, service_name="inference-service")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "orchestrator_loaded": orchestrator is not None}


@app.get("/stats")
async def get_stats():
    """Get system usage statistics."""
    if orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    return orchestrator.get_stats()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query through the Orchestrator pipeline.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")
    
    try:
        # Run the orchestrator
        result = orchestrator.run(request.query, session_id=request.session_id)
        
        return ChatResponse(
            query=request.query,
            solution=result.get("solution", ""),
            analysis=result.get("analysis", ""),
            context_length=len(result.get("context", "")),
            quality_score=result.get("quality_score", 0.0),
            cache_hit=result.get("cache_hit", False),
            reflection_history=result.get("reflection_history")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.inference_service.app:app", host="0.0.0.0", port=8000, workers=4)
