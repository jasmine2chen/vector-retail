# ── Build stage ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ───────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="jasmine2chen"
LABEL description="Vector Retail — Production Finance AI Agent v2.0"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/false appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Set Python path so vector_retail is importable
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default environment variables
ENV DEPLOYMENT_SLOT="blue"
ENV LOG_LEVEL="INFO"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# Expose HTTP port
EXPOSE 8080

# Entry point — production Uvicorn server
CMD ["uvicorn", "vector_retail.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
