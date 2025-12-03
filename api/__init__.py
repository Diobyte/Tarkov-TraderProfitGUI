# API package for Tarkov Trader Profit GraphQL Service
"""
GraphQL API for Tarkov Trader Profit data.

This package provides a GraphQL API for querying market data,
profitable items, trends, and ML insights.

Endpoints:
    - /graphql - GraphQL endpoint (GET for playground, POST for queries)
    - /health - Health check endpoint

Example Query:
    query {
        profitableItems(minProfit: 5000, limit: 10) {
            name
            profit
            roi
            fleaPrice
            traderName
        }
    }

Note:
    The API requires FastAPI, Strawberry, and uvicorn packages.
    These are optional dependencies not included in base requirements.
    Install with: pip install fastapi strawberry-graphql uvicorn
"""

# Lazy imports to avoid ImportError when optional dependencies are missing
__all__ = ['schema', 'app']


def __getattr__(name: str):
    """Lazy load API components to handle missing optional dependencies."""
    if name == 'schema':
        try:
            from api.schema import schema
            return schema
        except ImportError as e:
            raise ImportError(
                f"API schema requires strawberry-graphql. Install with: pip install strawberry-graphql"
            ) from e
    elif name == 'app':
        try:
            from api.server import app
            return app
        except ImportError as e:
            raise ImportError(
                f"API server requires fastapi and uvicorn. Install with: pip install fastapi uvicorn"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
