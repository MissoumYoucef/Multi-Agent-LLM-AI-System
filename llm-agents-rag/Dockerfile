# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies to a temporary location
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Define the command to run your app
# Using python main.py as default entrypoint, but allow overriding
ENTRYPOINT ["python", "main.py"]
