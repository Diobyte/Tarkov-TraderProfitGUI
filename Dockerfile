# Dockerfile for Tarkov Trader Profit GUI
# Multi-stage build for optimized image size
# Inspired by LinuxServer.io patterns: PUID/PGID, TZ, custom scripts
# Best practices: non-root user, minimal layers, health checks, proper signals

# ==============================================================================
# Stage 1: Builder - Install dependencies with build tools
# ==============================================================================
FROM python:3.14-slim AS builder

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
FROM python:3.14-slim AS production

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

# ==============================================================================
# Environment Variables (LinuxServer.io style)
# ==============================================================================
# User/Group IDs - Set these to match your host user for volume permissions
# Example: docker run -e PUID=1000 -e PGID=1000 ...
ENV PUID=1000 \
    PGID=1000 \
    # Timezone - Set to your local timezone
    # Example: docker run -e TZ=America/New_York ...
    TZ=Etc/UTC \
    # Python configuration
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Application data directory (mount a volume here)
    TARKOV_DATA_DIR=/data \
    # API collection interval in minutes
    TARKOV_COLLECTION_INTERVAL_MINUTES=5 \
    # Data retention period in days
    TARKOV_DATA_RETENTION_DAYS=7 \
    # Log level: DEBUG, INFO, WARNING, ERROR
    TARKOV_LOG_LEVEL=INFO \
    # Streamlit configuration
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # GraphQL API configuration
    API_HOST=0.0.0.0 \
    API_PORT=4000 \
    # Python path for installed packages
    PYTHONPATH=/usr/local/lib/python3.12/site-packages

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    bash \
    gosu \
    tini \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    # Create appuser with default UID/GID (will be modified at runtime)
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser \
    # Create directories
    && mkdir -p /data/exports /data/logs /custom-cont-init.d /custom-services.d \
    && chown -R 1000:1000 /data

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=1000:1000 . .

# Store version info and prepare entrypoint
RUN echo "${VERSION}" > /app/.version \
    && echo "${BUILD_DATE}" > /app/.build_date \
    && echo "${VCS_REF}" > /app/.git_commit \
    && cp .streamlit/config.docker.toml .streamlit/config.toml \
    && chmod +x docker-entrypoint.sh

# Expose ports
# 8501 = Streamlit Dashboard
# 4000 = GraphQL API
EXPOSE 8501 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["./docker-entrypoint.sh"]
