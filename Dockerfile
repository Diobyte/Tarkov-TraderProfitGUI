# Dockerfile for Tarkov Trader Profit GUI
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Base image with Python dependencies
# ==============================================================================
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Application image
# ==============================================================================
FROM python:3.12-slim as app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Docker-specific: Store data in /data volume
    TARKOV_DATA_DIR=/data \
    # Streamlit configuration
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install curl, bash, and gosu for health checks and entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from base stage
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Use Docker-specific Streamlit config
RUN cp .streamlit/config.docker.toml .streamlit/config.toml

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create data directory
RUN mkdir -p /data/exports /data/logs && chown -R appuser:appuser /data

# Note: We don't switch to appuser here because entrypoint needs to run as root
# to spawn multiple processes. The entrypoint handles this properly.

# Expose ports
# 8501 = Streamlit Dashboard
# 4000 = GraphQL API
EXPOSE 8501 4000

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: Run all services (collector + API + dashboard)
# Override with docker-compose for separate containers
CMD ["./docker-entrypoint.sh"]
