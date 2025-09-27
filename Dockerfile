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

# Copy application code
COPY src/ ./src/
COPY README.md .

# Create necessary directories
RUN mkdir -p logs data config && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add local user bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from src.main import *" || exit 1

# Expose port
EXPOSE 8080

# Run application - change the command
CMD ["python", "-m", "src.main"]
