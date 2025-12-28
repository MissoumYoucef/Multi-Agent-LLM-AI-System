#!/bin/bash

# run_local.sh - Run the Multi-Agent LLM system without Docker

# Ensure we are in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting Multi-Agent LLM System (Local Mode)"
echo "📍 Working Directory: $SCRIPT_DIR"

# Path configuration
PROMETHEUS_PATH="./monitoring/prometheus-3.0.1.linux-amd64/prometheus"
GRAFANA_BIN="./monitoring/grafana-v11.4.0/bin/grafana"
GRAFANA_HOME="./monitoring/grafana-v11.4.0"
JAEGER_BIN="./monitoring/jaeger-1.64.0-linux-amd64/jaeger-all-in-one"

# Create logs directory
mkdir -p logs

# Export common environment variables
export CACHE_ENABLED=true
export REDIS_URL=redis://localhost:6379
export TRACING_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# 1. Start Redis
if command -v redis-server >/dev/null; then
    echo "📦 Starting Redis..."
    redis-server --daemonize yes
else
    echo "❌ Warning: redis-server not found. Caching will use local memory only."
fi

# 2. Start Jaeger (Tracing)
if [ -f "$JAEGER_BIN" ]; then
    echo "🔦 Starting Jaeger (Port 16686)..."
    "$JAEGER_BIN" > logs/jaeger.log 2>&1 &
else
    echo "❌ Error: jaeger binary not found at $JAEGER_BIN"
fi

# 3. Start Prometheus (Port 9091 to avoid conflicts)
if [ -f "$PROMETHEUS_PATH" ]; then
    echo "📊 Starting Prometheus (Port 9091)..."
    "$PROMETHEUS_PATH" --config.file=prometheus.yml --web.listen-address=:9091 > logs/prometheus.log 2>&1 &
else
    echo "❌ Error: prometheus binary not found at $PROMETHEUS_PATH"
fi

# 3. Start Grafana (Port 3001 to avoid conflicts)
if [ -d "$GRAFANA_HOME" ]; then
    echo "📈 Starting Grafana (Port 3001)..."
    
    # Create a local config to ensure absolute paths for provisioning
    cat << EOF > grafana_local.ini
[server]
http_port = 3001

[paths]
provisioning = $SCRIPT_DIR/grafana_provisioning
data = $SCRIPT_DIR/monitoring/grafana-v11.4.0/data
logs = $SCRIPT_DIR/logs
plugins = $SCRIPT_DIR/monitoring/grafana-v11.4.0/plugins
EOF

    "$GRAFANA_BIN" server --config ./grafana_local.ini --homepath "$GRAFANA_HOME" > logs/grafana.log 2>&1 &
else
    echo "❌ Error: grafana directory not found at $GRAFANA_HOME"
fi

# 4. Start RAG Service
if [ -f ".venv/bin/activate" ]; then
    echo "🏗️ Starting RAG Service (port 8001)..."
    source .venv/bin/activate
    python -m services.rag_service.app > logs/rag.log 2>&1 &
else
    echo "❌ Error: .venv/bin/activate not found."
fi

# 5. Start Inference Service
if [ -f ".venv/bin/activate" ]; then
    echo "🧠 Starting Inference Service (port 8000)..."
    export RAG_SERVICE_URL=http://localhost:8001
    python -m services.inference_service.app > logs/inference.log 2>&1 &
fi

echo ""
echo "✅ All services initiated!"
echo "   - RAG Service: http://localhost:8001"
echo "   - Inference Service: http://localhost:8000"
echo "   - Prometheus: http://localhost:9091"
echo "   - Jaeger (Tracing): http://localhost:16686"
echo "   - Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "Use 'tail -f logs/*.log' to watch the logs."
echo "Use 'pkill -9 -f python', 'pkill -9 -f prometheus', 'pkill -9 -f grafana', and 'pkill -9 -f jaeger' to stop services."
