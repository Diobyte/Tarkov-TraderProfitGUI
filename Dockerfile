# Dockerfile for Tarkov Trader Profit GUI
# Multi-stage build for optimized image size
# Best practices: non-root user, minimal layers, health checks, proper signals

# ==============================================================================
# Stage 1: Builder - Install dependencies with build tools
# ==============================================================================
FROM python:3.12-slim AS builder

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy and install Python dependencies
# This layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# Stage 2: Production image - Minimal runtime
# ==============================================================================
FROM python:3.12-slim AS production

# Build arguments for version labeling (set by CI/CD)
ARG VERSION=dev
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# OCI Image Labels - Standard metadata for container registries
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.title="Tarkov Trader Profit GUI" \
    org.opencontainers.image.description="Flea market arbitrage finder for Escape from Tarkov" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.source="https://github.com/Diobyte/Tarkov-TraderProfitGUI" \
    org.opencontainers.image.url="https://github.com/Diobyte/Tarkov-TraderProfitGUI" \
    org.opencontainers.image.documentation="https://github.com/Diobyte/Tarkov-TraderProfitGUI#readme" \
    org.opencontainers.image.vendor="Diobyte" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.base.name="docker.io/library/python:3.12-slim"

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Use UTF-8 encoding
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Docker-specific: Store data in /data volume
    TARKOV_DATA_DIR=/data \
    # Streamlit configuration
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Python path for installed packages
    PYTHONPATH=/usr/local/lib/python3.12/site-packages

# Install only runtime dependencies (no build tools)
# Combine RUN commands to reduce layers
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    bash \
    gosu \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Create non-root user with specific UID/GID for consistency
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser \
    # Create data directories
    && mkdir -p /data/exports /data/logs \
    && chown -R appuser:appuser /data

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code with proper ownership
# Using .dockerignore to exclude unnecessary files
COPY --chown=appuser:appuser . .

# Store version info in the image for runtime checking
RUN echo "${VERSION}" > /app/.version \
    && echo "${BUILD_DATE}" > /app/.build_date \
    && echo "${VCS_REF}" > /app/.git_commit \
    # Use Docker-specific Streamlit config
    && cp .streamlit/config.docker.toml .streamlit/config.toml \
    # Make entrypoint script executable
    && chmod +x docker-entrypoint.sh

# Expose ports (documentation only - actual binding in docker-compose)
# 8501 = Streamlit Dashboard
# 4000 = GraphQL API
EXPOSE 8501 4000

# Health check for Streamlit dashboard
# --start-period gives the app time to start before checking
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# Use tini as init system for proper signal handling and zombie reaping
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: Run all services
CMD ["./docker-entrypoint.sh"]
