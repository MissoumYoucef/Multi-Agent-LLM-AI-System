# =============================================================================
# Inference Service Dockerfile
# Multi-stage build copying ONLY necessary files for LLM agents.
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies (minimal for this service)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (for caching)
COPY services/inference_service/requirements.txt .

# Install dependencies to a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.9-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the necessary source files for Inference service
# 1. The service entrypoint
COPY services/inference_service/app.py ./services/inference_service/

# 2. The Agents module (chatbot, solver, analyzer)
COPY src/agents/chatbot.py ./src/agents/
COPY src/agents/solver.py ./src/agents/
COPY src/agents/analyzer.py ./src/agents/
COPY src/agents/__init__.py ./src/agents/

# 3. Utils (config)
COPY src/utils/ ./src/utils/

# 4. Make src a package
COPY src/__init__.py ./src/

# NOTE: We do NOT copy src/rag/ - that's the RAG service's responsibility.
# NOTE: We do NOT copy any data/ - inference doesn't need documents.

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD ["uvicorn", "services.inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
