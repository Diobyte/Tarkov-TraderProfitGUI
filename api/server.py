"""
GraphQL API Server for Tarkov Trader Profit.

Runs a FastAPI server with Strawberry GraphQL integration.
Provides both GraphQL endpoint and a GraphQL Playground UI.
"""

import os
import sys
import logging
import uvicorn
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from strawberry.fastapi import GraphQLRouter

# Import schema and database
from api.schema import schema
import database
import config

# Configure logging
log_file = os.path.join(config.LOGS_DIR, "api.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Tarkov Trader Profit GraphQL API...")
    database.init_db()
    logger.info("Database initialized.")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Tarkov Trader Profit API",
    description="GraphQL API for Tarkov market data and trading analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for local network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local network
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create GraphQL router with Strawberry
graphql_app = GraphQLRouter(schema)

# Mount GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tarkov Trader Profit API",
        "version": "1.0.0",
        "endpoints": {
            "graphql": "/graphql",
            "graphql_playground": "/graphql (GET in browser)",
            "health": "/health",
        },
        "documentation": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health = database.get_database_health()
        return JSONResponse(
            status_code=200 if health['status'] != 'error' else 503,
            content={
                "status": health['status'],
                "database": {
                    "total_records": health.get('total_records', 0),
                    "unique_items": health.get('unique_items', 0),
                    "data_age_hours": health.get('data_age_hours', 0),
                },
                "api": "healthy",
            }
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.get("/stats")
async def stats():
    """Quick stats endpoint."""
    try:
        health = database.get_database_health()
        return {
            "items_tracked": health.get('unique_items', 0),
            "total_records": health.get('total_records', 0),
            "last_update": health.get('newest_record'),
            "data_age_hours": round(health.get('data_age_hours', 0), 2),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "4000"))
    
    logger.info("Starting API server on %s:%d", host, port)
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
