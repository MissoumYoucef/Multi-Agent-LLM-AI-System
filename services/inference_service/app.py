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

from src.agents.chatbot import ChatbotAgent
from src.agents.solver import SolverAgent
from src.agents.analyzer import AnalyzerAgent

# Configuration
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")

# Global agents
chatbot: Optional[ChatbotAgent] = None
solver: Optional[SolverAgent] = None
analyzer: Optional[AnalyzerAgent] = None


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    context_length: int
    solution: str
    analysis: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup."""
    global chatbot, solver, analyzer
    print("--- Inference Service Starting ---")
    
    chatbot = ChatbotAgent()
    solver = SolverAgent()
    analyzer = AnalyzerAgent()
    
    print("Agents initialized.")
    yield
    
    print("--- Inference Service Shutting Down ---")


app = FastAPI(
    title="Inference Service",
    description="LLM Agent orchestration service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agents_loaded": all([chatbot, solver, analyzer])}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query through the agent pipeline.
    1. Calls RAG service to retrieve context.
    2. Solver agent generates a solution.
    3. Analyzer agent provides analysis.
    """
    if not all([chatbot, solver, analyzer]):
        raise HTTPException(status_code=503, detail="Agents not initialized.")
    
    # Step 1: Retrieve context from RAG service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"query": request.query, "top_k": 5}
            )
            response.raise_for_status()
            rag_response = response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"RAG service error: {str(e)}")
    
    # Combine retrieved documents into context
    context = "\n\n".join([doc["content"] for doc in rag_response.get("documents", [])])
    
    if not context:
        context = "No relevant context found in the knowledge base."
    
    # Step 2: Solve
    print(f"--- Solving query: {request.query} ---")
    solution = solver.invoke(request.query, context)
    
    # Step 3: Analyze
    print("--- Analyzing ---")
    analysis = analyzer.invoke(request.query, solution, context)
    
    return ChatResponse(
        query=request.query,
        context_length=len(context),
        solution=solution,
        analysis=analysis
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
