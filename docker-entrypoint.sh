#!/bin/bash
# Docker entrypoint script for Tarkov Trader Profit GUI
# Runs all services in a single container

set -e

echo "=========================================="
echo "  Tarkov Trader Profit GUI - Starting"
echo "=========================================="

# Create data directories if they don't exist
mkdir -p /data/exports /data/logs

# Fix permissions if running as root
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser /data
    # Re-run this script as appuser
    exec gosu appuser "$0" "$@"
fi

# Initialize database
echo "[1/3] Initializing database..."
python -c "import database; database.init_db()"

# Start the data collector in the background
echo "[2/3] Starting data collector..."
python -u collector.py --standalone &
COLLECTOR_PID=$!
echo "  Collector started (PID: $COLLECTOR_PID)"

# Wait a moment for initial data collection
sleep 5

# Start the GraphQL API in the background
echo "[3/3] Starting GraphQL API..."
python -u api/server.py &
API_PID=$!
echo "  API started (PID: $API_PID)"

# Give API time to start
sleep 2

echo ""
echo "=========================================="
echo "  All services started!"
echo "  Dashboard: http://localhost:8501"
echo "  GraphQL:   http://localhost:4000/graphql"
echo "=========================================="
echo ""

# Start Streamlit dashboard in the foreground (keeps container alive)
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
