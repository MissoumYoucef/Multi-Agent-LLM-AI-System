# =============================================================================
# RAG Service Dockerfile
# Multi-stage build copying ONLY necessary files for document retrieval.
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.9-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* # Packages Catalog

# Copy only the requirements file first (for caching)
COPY services/rag_service/requirements.txt .

# Install dependencies to a virtual env for easy copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    # --no-cache-dir = Meta data, archived files , .whl
# --- Stage 2: Runtime ---
FROM python:3.9-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the necessary source files for RAG service
# 1. The service entrypoint
COPY services/rag_service/app.py ./services/rag_service/

# 2. The RAG module (loader, vectorstore, retriever)
COPY src/rag/ ./src/rag/

# 3. Utils (config)
COPY src/utils/ ./src/utils/

# 4. Make src a package
COPY src/__init__.py ./src/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Data will be mounted as volume
# COPY data/ ./data/  # Don't copy, mount instead

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health').raise_for_status()" || exit 1

CMD ["uvicorn", "services.rag_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
