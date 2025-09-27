# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md .

# Create necessary directories with cloud optimization
RUN mkdir -p logs data config /tmp/speed_demon_data && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/speed_demon_data

# Switch to non-root user
USER appuser

# Add local user bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Set cloud data path for Speed Demon
ENV CLOUD_DATA_PATH=/tmp/speed_demon_data

# Health check (updated for speed demon compatibility)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from src.main import *" || exit 1

# Expose port
EXPOSE 8080

# Speed Demon startup script - downloads data then starts main application
CMD ["sh", "-c", "python scripts/speed_demon_deploy.py --years 2 --testnet && python -m src.main"]
