#!/bin/bash
# Docker entrypoint script for Tarkov Trader Profit GUI
# Inspired by LinuxServer.io patterns: PUID/PGID mapping, TZ, custom scripts
# Runs all services with proper signal handling via tini

set -e

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ==============================================================================
# Graceful Shutdown Handler
# ==============================================================================

cleanup() {
    echo ""
    log "Stopping services..."
    
    if [ -n "$COLLECTOR_PID" ] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
        kill -TERM "$COLLECTOR_PID" 2>/dev/null || true
        wait "$COLLECTOR_PID" 2>/dev/null || true
        log "  Collector stopped"
    fi
    
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill -TERM "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
        log "  API stopped"
    fi
    
    log "Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# ==============================================================================
# Initial Setup (runs as root)
# ==============================================================================

VERSION=$(cat /app/.version 2>/dev/null || echo "unknown")

echo "=========================================="
echo "  Tarkov Trader Profit GUI v${VERSION}"
echo "=========================================="
echo ""

# Only run setup if we're root
if [ "$(id -u)" = "0" ]; then
    
    # -------------------------------------------------------------------------
    # Set Timezone (LinuxServer.io pattern)
    # -------------------------------------------------------------------------
    if [ -n "$TZ" ] && [ -f "/usr/share/zoneinfo/$TZ" ]; then
        log "Setting timezone to $TZ"
        ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
        echo "$TZ" > /etc/timezone
    fi
    
    # -------------------------------------------------------------------------
    # Set User/Group IDs (LinuxServer.io pattern)
    # -------------------------------------------------------------------------
    PUID=${PUID:-1000}
    PGID=${PGID:-1000}
    
    log "Setting user appuser to UID=$PUID, GID=$PGID"
    
    # Modify group if PGID differs
    if [ "$(id -g appuser)" != "$PGID" ]; then
        groupmod -o -g "$PGID" appuser 2>/dev/null || true
    fi
    
    # Modify user if PUID differs
    if [ "$(id -u appuser)" != "$PUID" ]; then
        usermod -o -u "$PUID" appuser 2>/dev/null || true
    fi
    
    # -------------------------------------------------------------------------
    # Create and fix permissions on data directories
    # -------------------------------------------------------------------------
    log "Setting up data directories..."
    mkdir -p /data/exports /data/logs
    chown -R appuser:appuser /data
    chown -R appuser:appuser /app
    
    # -------------------------------------------------------------------------
    # Run custom init scripts (LinuxServer.io pattern)
    # Mount your scripts to /custom-cont-init.d
    # -------------------------------------------------------------------------
    if [ -d "/custom-cont-init.d" ]; then
        for script in /custom-cont-init.d/*; do
            if [ -f "$script" ] && [ -x "$script" ]; then
                log "Running custom init script: $(basename "$script")"
                "$script" || log "Warning: Custom script $(basename "$script") failed"
            fi
        done
    fi
    
    # -------------------------------------------------------------------------
    # Display configuration
    # -------------------------------------------------------------------------
    echo ""
    echo "Configuration:"
    echo "  User UID:    $(id -u appuser)"
    echo "  User GID:    $(id -g appuser)"
    echo "  Timezone:    ${TZ:-UTC}"
    echo "  Data Dir:    ${TARKOV_DATA_DIR:-/data}"
    echo "  Log Level:   ${TARKOV_LOG_LEVEL:-INFO}"
    echo "  Interval:    ${TARKOV_COLLECTION_INTERVAL_MINUTES:-5} minutes"
    echo "  Retention:   ${TARKOV_DATA_RETENTION_DAYS:-7} days"
    echo ""
    
    # Re-run this script as appuser
    exec gosu appuser "$0" "$@"
fi

# ==============================================================================
# Main Application Startup (runs as appuser)
# ==============================================================================

log "[1/3] Initializing database..."
python -c "import database; database.init_db()"

log "[2/3] Starting data collector..."
python -u collector.py --standalone &
COLLECTOR_PID=$!
log "  Collector started (PID: $COLLECTOR_PID)"

# Wait for initial data collection
sleep 5

log "[3/3] Starting GraphQL API..."
python -u api/server.py &
API_PID=$!
log "  API started (PID: $API_PID)"

sleep 2

echo ""
echo "=========================================="
echo "  All services started!"
echo "  Dashboard: http://localhost:8501"
echo "  GraphQL:   http://localhost:4000/graphql"
echo "=========================================="
echo ""

# Start Streamlit dashboard in the foreground
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
