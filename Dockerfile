# Dockerfile for ML Trading Bot - Production Multi-stage Build
FROM python:3.11-slim as builder

# Set build arguments for metadata
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG GIT_COMMIT=unknown

# Add container metadata
LABEL maintainer="ML Trading Bot Team"
LABEL org.opencontainers.image.title="ML Trading Bot"
LABEL org.opencontainers.image.description="Production-grade ML cryptocurrency trading bot"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${GIT_COMMIT}

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all requirements files
COPY requirements.txt requirements-api.txt requirements-dashboard.txt ./

# Install Python dependencies with optimizations
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user -r requirements.txt && \
    pip install --user -r requirements-api.txt && \
    pip install --user -r requirements-dashboard.txt

# Production stage - Minimal runtime image
FROM python:3.11-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    redis-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r tradingbot && \
    useradd -r -g tradingbot -d /app -s /bin/bash tradingbot

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/tradingbot/.local/bin:$PATH" \
    PYTHONPATH=/app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/tradingbot/.local

# Set working directory and ownership
WORKDIR /app
RUN chown tradingbot:tradingbot /app

# Copy application code with correct ownership
COPY --chown=tradingbot:tradingbot . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/{logs,data,models,config} && \
    chown -R tradingbot:tradingbot /app/{logs,data,models,config}

# Create volumes for persistent data
VOLUME ["/app/data", "/app/models", "/app/logs", "/app/config"]

# Switch to non-root user
USER tradingbot

# Expose ports for API and Dashboard
EXPOSE 8000 8501

# Health check for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - Run API server
CMD ["python", "start_api.py", "--host", "0.0.0.0", "--port", "8000"]