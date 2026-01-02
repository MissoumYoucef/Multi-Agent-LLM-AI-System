import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .config import TRACING_ENABLED, TRACING_SERVICE_NAME

logger = logging.getLogger(__name__)

def setup_tracing(app, service_name=None):
    """
    Setup OpenTelemetry tracing for a FastAPI app.

    Args:
        app: The FastAPI application instance.
        service_name: Optional service name override.
    """
    if not TRACING_ENABLED:
        logger.info("Tracing is disabled.")
        return

    name = service_name or TRACING_SERVICE_NAME
    logger.info(f"Setting up tracing for service: {name}")

    resource = Resource(attributes={
        SERVICE_NAME: name
    })

    provider = TracerProvider(resource=resource)

    # Use OTLP exporter (default for Jaeger)
    # Jaeger OTLP gRPC port is 4317 by default
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")

    try:
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        logger.info(f"Tracing setup complete (endpoint: {otlp_endpoint}).")
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
