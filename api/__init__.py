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
"""

__all__ = ['create_app', 'schema']
