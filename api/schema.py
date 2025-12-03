"""
GraphQL Schema for Tarkov Trader Profit API.

Uses Strawberry for modern, type-safe GraphQL in Python.
"""

import strawberry
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import config
from utils import calculate_metrics
import pandas as pd


# =============================================================================
# GraphQL Types
# =============================================================================

@strawberry.type
class Item:
    """Represents a Tarkov item with market data."""
    item_id: str
    name: str
    short_name: Optional[str] = None
    category: Optional[str] = None
    
    # Prices
    flea_price: int
    trader_price: int
    trader_name: Optional[str] = None
    profit: int
    
    # Calculated metrics
    roi: Optional[float] = None
    profit_per_slot: Optional[float] = None
    
    # Market data
    avg_24h_price: Optional[int] = None
    low_24h_price: Optional[int] = None
    high_24h_price: Optional[int] = None
    change_last_48h: Optional[float] = None
    last_offer_count: Optional[int] = None
    
    # Item properties
    width: Optional[int] = None
    height: Optional[int] = None
    weight: Optional[float] = None
    base_price: Optional[int] = None
    
    # Requirements
    trader_level_required: Optional[int] = None
    flea_level_required: Optional[int] = None
    
    # Item type flags (Patch 1.0+)
    no_flea: Optional[bool] = None
    marked_only: Optional[bool] = None
    
    # Links
    icon_link: Optional[str] = None
    wiki_link: Optional[str] = None
    
    # Timestamp
    timestamp: Optional[str] = None


@strawberry.type
class CategoryStats:
    """Statistics for a category."""
    category: str
    item_count: int
    avg_profit: float
    total_profit: float
    profitable_count: int


@strawberry.type
class TraderStats:
    """Statistics for a trader."""
    trader_name: str
    item_count: int
    avg_profit: float
    total_profit: float


@strawberry.type
class MarketTrend:
    """Market trend data for an item."""
    item_id: str
    avg_profit: float
    min_profit: float
    max_profit: float
    data_points: int
    volatility: Optional[float] = None


@strawberry.type
class DatabaseHealth:
    """Database health information."""
    status: str
    total_records: int
    unique_items: int
    oldest_record: Optional[str] = None
    newest_record: Optional[str] = None
    data_age_hours: float
    file_size_mb: float


@strawberry.type
class ApiInfo:
    """API information."""
    version: str
    name: str
    description: str
    collection_interval_minutes: int
    data_retention_days: int


# =============================================================================
# Helper Functions
# =============================================================================

def _get_latest_data() -> pd.DataFrame:
    """Get latest price data from database."""
    try:
        data = database.get_latest_prices()
        if not data:
            return pd.DataFrame()
        
        columns = [
            'item_id', 'name', 'flea_price', 'trader_price', 'trader_name', 'profit',
            'timestamp', 'icon_link', 'width', 'height', 'avg_24h_price', 'low_24h_price',
            'change_last_48h', 'weight', 'category', 'base_price', 'high_24h_price',
            'last_offer_count', 'short_name', 'wiki_link', 'trader_level_required',
            'trader_task_unlock', 'flea_level_required', 'price_velocity', 'liquidity_score',
            'no_flea', 'marked_only'
        ]
        
        # Handle different data formats
        if data and len(data[0]) < len(columns):
            columns = columns[:len(data[0])]
        
        df = pd.DataFrame(data, columns=columns)
        df = calculate_metrics(df)
        return df
    except Exception:
        return pd.DataFrame()


def _row_to_item(row: pd.Series) -> Item:
    """Convert a DataFrame row to an Item object."""
    return Item(
        item_id=str(row.get('item_id', '')),
        name=str(row.get('name', '')),
        short_name=row.get('short_name') if pd.notna(row.get('short_name')) else None,
        category=row.get('category') if pd.notna(row.get('category')) else None,
        flea_price=int(row.get('flea_price', 0)),
        trader_price=int(row.get('trader_price', 0)),
        trader_name=row.get('trader_name') if pd.notna(row.get('trader_name')) else None,
        profit=int(row.get('profit', 0)),
        roi=float(row.get('roi', 0)) if pd.notna(row.get('roi')) else None,
        profit_per_slot=float(row.get('profit_per_slot', 0)) if pd.notna(row.get('profit_per_slot')) else None,
        avg_24h_price=int(row.get('avg_24h_price', 0)) if pd.notna(row.get('avg_24h_price')) else None,
        low_24h_price=int(row.get('low_24h_price', 0)) if pd.notna(row.get('low_24h_price')) else None,
        high_24h_price=int(row.get('high_24h_price', 0)) if pd.notna(row.get('high_24h_price')) else None,
        change_last_48h=float(row.get('change_last_48h', 0)) if pd.notna(row.get('change_last_48h')) else None,
        last_offer_count=int(row.get('last_offer_count', 0)) if pd.notna(row.get('last_offer_count')) else None,
        width=int(row.get('width', 1)) if pd.notna(row.get('width')) else None,
        height=int(row.get('height', 1)) if pd.notna(row.get('height')) else None,
        weight=float(row.get('weight', 0)) if pd.notna(row.get('weight')) else None,
        base_price=int(row.get('base_price', 0)) if pd.notna(row.get('base_price')) else None,
        trader_level_required=int(row.get('trader_level_required', 1)) if pd.notna(row.get('trader_level_required')) else None,
        flea_level_required=int(row.get('flea_level_required', 15)) if pd.notna(row.get('flea_level_required')) else None,
        no_flea=bool(row.get('no_flea', 0)) if pd.notna(row.get('no_flea')) else None,
        marked_only=bool(row.get('marked_only', 0)) if pd.notna(row.get('marked_only')) else None,
        icon_link=row.get('icon_link') if pd.notna(row.get('icon_link')) else None,
        wiki_link=row.get('wiki_link') if pd.notna(row.get('wiki_link')) else None,
        timestamp=str(row.get('timestamp', '')) if pd.notna(row.get('timestamp')) else None,
    )


# =============================================================================
# GraphQL Query
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query root."""
    
    @strawberry.field
    def info(self) -> ApiInfo:
        """Get API information."""
        return ApiInfo(
            version="1.0.0",
            name="Tarkov Trader Profit API",
            description="GraphQL API for Tarkov market data and trading analysis",
            collection_interval_minutes=config.COLLECTION_INTERVAL_MINUTES,
            data_retention_days=config.DATA_RETENTION_DAYS,
        )
    
    @strawberry.field
    def health(self) -> DatabaseHealth:
        """Get database health status."""
        health = database.get_database_health()
        return DatabaseHealth(
            status=health.get('status', 'unknown'),
            total_records=health.get('total_records', 0),
            unique_items=health.get('unique_items', 0),
            oldest_record=health.get('oldest_record'),
            newest_record=health.get('newest_record'),
            data_age_hours=health.get('data_age_hours') or 0,
            file_size_mb=(health.get('file_size') or 0) / (1024 * 1024),
        )
    
    @strawberry.field
    def items(
        self,
        limit: int = 100,
        offset: int = 0,
        category: Optional[str] = None,
        trader: Optional[str] = None,
        min_profit: Optional[int] = None,
        max_profit: Optional[int] = None,
        min_roi: Optional[float] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = "profit",
        sort_desc: bool = True,
    ) -> List[Item]:
        """
        Get items with optional filtering and sorting.
        
        Args:
            limit: Maximum number of items to return (default: 100, max: 1000)
            offset: Number of items to skip
            category: Filter by category name
            trader: Filter by trader name
            min_profit: Minimum profit filter
            max_profit: Maximum profit filter
            min_roi: Minimum ROI percentage filter
            search: Search in item name (case-insensitive)
            sort_by: Field to sort by (profit, roi, flea_price, trader_price, name)
            sort_desc: Sort descending if True
        """
        df = _get_latest_data()
        if df.empty:
            return []
        
        # Apply filters
        if category:
            df = df[df['category'] == category]
        if trader:
            df = df[df['trader_name'] == trader]
        if min_profit is not None:
            df = df[df['profit'] >= min_profit]
        if max_profit is not None:
            df = df[df['profit'] <= max_profit]
        if min_roi is not None and 'roi' in df.columns:
            df = df[df['roi'] >= min_roi]
        if search:
            df = df[df['name'].str.contains(search, case=False, na=False)]
        
        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=not sort_desc)
        
        # Paginate
        limit = min(limit, 1000)
        df = df.iloc[offset:offset + limit]
        
        return [_row_to_item(row) for _, row in df.iterrows()]
    
    @strawberry.field
    def item(self, item_id: str) -> Optional[Item]:
        """Get a specific item by ID."""
        df = _get_latest_data()
        if df.empty:
            return None
        
        matches = df[df['item_id'] == item_id]
        if matches.empty:
            return None
        
        return _row_to_item(matches.iloc[0])
    
    @strawberry.field
    def item_by_name(self, name: str) -> Optional[Item]:
        """Get a specific item by name (exact match, case-insensitive)."""
        df = _get_latest_data()
        if df.empty:
            return None
        
        matches = df[df['name'].str.lower() == name.lower()]
        if matches.empty:
            return None
        
        return _row_to_item(matches.iloc[0])
    
    @strawberry.field
    def profitable_items(
        self,
        min_profit: int = 0,
        min_roi: float = 0.0,
        min_offers: int = 5,
        limit: int = 50,
        player_level: Optional[int] = None,
    ) -> List[Item]:
        """
        Get profitable items sorted by profit.
        
        Args:
            min_profit: Minimum profit threshold
            min_roi: Minimum ROI percentage
            min_offers: Minimum offer count for reliability
            limit: Maximum items to return
            player_level: Filter by flea market level requirement
        """
        df = _get_latest_data()
        if df.empty:
            return []
        
        # Filter
        df = df[df['profit'] >= min_profit]
        if 'roi' in df.columns:
            df = df[df['roi'] >= min_roi]
        if 'last_offer_count' in df.columns:
            df = df[df['last_offer_count'] >= min_offers]
        if player_level is not None and 'flea_level_required' in df.columns:
            df = df[df['flea_level_required'] <= player_level]
        
        # Sort and limit
        df = df.sort_values('profit', ascending=False).head(limit)
        
        return [_row_to_item(row) for _, row in df.iterrows()]
    
    @strawberry.field
    def categories(self) -> List[CategoryStats]:
        """Get statistics for all categories."""
        df = _get_latest_data()
        if df.empty or 'category' not in df.columns:
            return []
        
        stats = df.groupby('category').agg({
            'profit': ['count', 'mean', 'sum'],
        }).reset_index()
        
        stats.columns = ['category', 'item_count', 'avg_profit', 'total_profit']
        
        # Count profitable items per category
        profitable = df[df['profit'] > 0].groupby('category').size().reset_index(name='profitable_count')
        stats = stats.merge(profitable, on='category', how='left')
        stats['profitable_count'] = stats['profitable_count'].fillna(0).astype(int)
        
        return [
            CategoryStats(
                category=row['category'],
                item_count=int(row['item_count']),
                avg_profit=float(row['avg_profit']),
                total_profit=float(row['total_profit']),
                profitable_count=int(row['profitable_count']),
            )
            for _, row in stats.iterrows()
        ]
    
    @strawberry.field
    def traders(self) -> List[TraderStats]:
        """Get statistics for all traders."""
        df = _get_latest_data()
        if df.empty or 'trader_name' not in df.columns:
            return []
        
        stats = df.groupby('trader_name').agg({
            'profit': ['count', 'mean', 'sum'],
        }).reset_index()
        
        stats.columns = ['trader_name', 'item_count', 'avg_profit', 'total_profit']
        
        return [
            TraderStats(
                trader_name=str(row['trader_name']),
                item_count=int(row['item_count']),
                avg_profit=float(row['avg_profit']),
                total_profit=float(row['total_profit']),
            )
            for _, row in stats.iterrows()
        ]
    
    @strawberry.field
    def trends(self, hours: int = 24, limit: int = 50) -> List[MarketTrend]:
        """
        Get market trends over the specified time period.
        
        Args:
            hours: Number of hours to analyze
            limit: Maximum number of items to return
        """
        try:
            trends = database.get_market_trends(hours=hours)
            if not trends:
                return []
            
            results = []
            for row in trends[:limit]:
                item_id, avg_profit, min_profit, max_profit, data_points = row
                volatility = (max_profit - min_profit) / abs(avg_profit) if avg_profit != 0 else 0
                
                results.append(MarketTrend(
                    item_id=str(item_id),
                    avg_profit=float(avg_profit or 0),
                    min_profit=float(min_profit or 0),
                    max_profit=float(max_profit or 0),
                    data_points=int(data_points or 0),
                    volatility=float(volatility),
                ))
            
            return results
        except Exception:
            return []
    
    @strawberry.field
    def search(self, query: str, limit: int = 20) -> List[Item]:
        """
        Search for items by name.
        
        Args:
            query: Search string (case-insensitive, partial match)
            limit: Maximum results to return
        """
        df = _get_latest_data()
        if df.empty:
            return []
        
        matches = df[df['name'].str.contains(query, case=False, na=False)]
        matches = matches.sort_values('profit', ascending=False).head(limit)
        
        return [_row_to_item(row) for _, row in matches.iterrows()]


# =============================================================================
# Create Schema
# =============================================================================

schema = strawberry.Schema(query=Query)
