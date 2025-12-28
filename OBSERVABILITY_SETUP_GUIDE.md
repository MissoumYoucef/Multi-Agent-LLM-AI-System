# 📊 Full Observability Integration Guide (Local Setup)

This guide explains how to integrate **Prometheus**, **Grafana**, **Redis**, and **Jaeger** into a Python project for local development without using Docker.

---

## 🏗️ Architecture Overview

1.  **Redis**: High-speed caching layer used to store responses and reduce LLM costs.
2.  **Prometheus**: Scraper that collects numerical metrics (hits, misses, latency) from your apps.
3.  **Jaeger**: Distributed tracing system that tracks the lifecycle of a request across microservices.
4.  **Grafana**: The visual dashboard that connects to Prometheus and Jaeger to display charts.

---

## 🛠️ Step 1: Python Dependencies

Install the necessary client libraries in your project:

```bash
# Metrics
pip install prometheus-client prometheus-fastapi-instrumentator

# Redis
pip install redis

# Tracing (OpenTelemetry)
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc \
            opentelemetry-instrumentation-fastapi
```

---

## 📉 Step 2: Prometheus Integration

### A. Instrument your FastAPI App
In your main application file (`app.py`), add the instrumentator:

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# This exposes a /metrics endpoint automatically
Instrumentator().instrument(app).expose(app)
```

### B. Configure Prometheus Scraper
Create a `prometheus.yml` file to tell Prometheus where to look for data:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "my-service"
    static_configs:
      - targets: ["localhost:8000"] # Project API port
    metrics_path: /metrics
```

---

## 🔦 Step 3: Jaeger Tracing Integration

### A. Create a Tracing Utility
Create a `tracing.py` to handle the OpenTelemetry logic:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_tracing(app, service_name):
    provider = TracerProvider()
    # 4317 is the default OTLP gRPC port for Jaeger
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
```

### B. Initialize in App
```python
from .tracing import setup_tracing

app = FastAPI()
setup_tracing(app, "my-service-name")
```

---

## 📦 Step 4: Redis Caching Integration

In your logic layer, use the Redis client to save results:

```python
import redis

# Connection
r = redis.from_url("redis://localhost:6379")

def get_cached_response(query):
    return r.get(f"cache:{query}")

def set_cache(query, response):
    r.setex(f"cache:{query}", 3600, response) # Store for 1 hour
```

---

## 📈 Step 5: Grafana Configuration

### A. Provision Datasources
To avoid manual setup, create `provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus # Must match dashboard JSON
    url: http://localhost:9091 # Prometheus port
    isDefault: true
```

---

## 🚀 Step 6: Local Automation Script

Create a `run_local.sh` to start everything in the background:

```bash
#!/bin/bash

# 1. Start Background Binaries
./prometheus --config.file=prometheus.yml --web.listen-address=:9091 &
./grafana-server --config=grafana.ini &
./jaeger-all-in-one &
redis-server --daemonize yes

# 2. Export Tracing Envs
export TRACING_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# 3. Start Python App
python -m my_app.main
```

---

## ✅ Verification Checklist

| Service | Local URL | Verify Action |
| :--- | :--- | :--- |
| **Project API** | `localhost:8000` | Run a request |
| **Metrics** | `localhost:8000/metrics` | See text data |
| **Prometheus** | `localhost:9091` | Check "Status -> Targets" |
| **Jaeger** | `localhost:16686` | Search for traces |
| **Grafana** | `localhost:3001` | View your dashboard |
| **Redis** | `redis-cli ping` | Should return `PONG` |
