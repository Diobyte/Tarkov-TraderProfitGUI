"""Database operations for Tarkov Trader Profit application.

Handles SQLite connections, schema management, and data operations
with WAL mode for concurrent read/write access.
"""

import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any, Callable, Dict
from functools import wraps

import config

__all__ = [
    'init_db', 'save_prices_batch', 'get_latest_prices', 'get_item_history',
    'get_market_trends', 'get_all_prices', 'cleanup_old_data', 'get_latest_timestamp',
    'clear_all_data', 'parse_timestamp', 'retry_db_op', 'DB_NAME',
    'get_item_trend_data', 'get_profit_statistics', 'DatabaseConnection',
    'get_database_health'
]

# Database path from centralized config (stored in user's Documents folder)
DB_NAME = config.DB_PATH


class DatabaseConnection:
    """
    Context manager for database connections.
    
    Ensures proper connection handling and cleanup, with automatic
    commit on success and rollback on error. Uses WAL mode for
    better concurrent read/write performance.
    
    Usage:
        with DatabaseConnection() as (conn, cursor):
            cursor.execute('SELECT * FROM prices')
            rows = cursor.fetchall()
            
        # Read-only mode (opens connection in read-only mode):
        with DatabaseConnection(readonly=True) as (conn, cursor):
            cursor.execute('SELECT COUNT(*) FROM prices')
            
    Args:
        timeout: Connection timeout in seconds. Defaults to config value.
        readonly: If True, opens connection in read-only mode using URI.
        
    Note:
        Automatically commits on successful exit, rolls back on error.
        Connection is always closed on exit.
    """
    
    def __init__(self, timeout: Optional[int] = None, readonly: bool = False) -> None:
        """Initialize the connection manager."""
        self.timeout = timeout if timeout is not None else config.DATABASE_CONNECTION_TIMEOUT
        self.readonly = readonly
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def __enter__(self) -> Tuple['sqlite3.Connection', 'sqlite3.Cursor']:
        """Open database connection."""
        try:
            if self.readonly:
                # Check if database file exists before opening in read-only mode
                if not os.path.exists(DB_NAME):
                    raise sqlite3.OperationalError(f"Database file does not exist: {DB_NAME}")
                # Read-only connection via URI
                uri = f"file:{DB_NAME}?mode=ro"
                self.conn = sqlite3.connect(uri, timeout=self.timeout, uri=True)
            else:
                self.conn = sqlite3.connect(DB_NAME, timeout=self.timeout)
            self.cursor = self.conn.cursor()
            return self.conn, self.cursor
        except sqlite3.Error as e:
            logging.error("Failed to connect to database: %s", e)
            raise
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Close connection, commit if no errors, rollback otherwise."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
        return False  # Don't suppress exceptions


def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a timestamp string into a datetime object.
    Handles multiple formats for backward compatibility.
    
    Args:
        ts_str: Timestamp string in ISO or legacy format.
        
    Returns:
        datetime object or None if parsing fails.
    """
    if not ts_str:
        return None
    
    # Try ISO format first (most common)
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass
    
    # Try legacy formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    return None

def retry_db_op(
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to retry database operations when locked.

    SQLite can raise OperationalError with 'database is locked' when multiple
    processes/threads access the database simultaneously. This decorator
    automatically retries the operation with a fixed delay between attempts.
    
    Args:
        max_retries: Maximum number of retry attempts. Defaults to
                    config.DATABASE_RETRY_ATTEMPTS (5).
        delay: Delay in seconds between retries. Defaults to
              config.DATABASE_RETRY_DELAY (1.0).
              
    Returns:
        Decorated function that retries on database lock errors.
        
    Raises:
        sqlite3.OperationalError: If all retries are exhausted.
        Exception: Any other non-lock related exceptions are re-raised.
        
    Example:
        @retry_db_op(max_retries=3, delay=0.5)
        def get_item_count():
            with DatabaseConnection() as (conn, cursor):
                cursor.execute('SELECT COUNT(*) FROM prices')
                return cursor.fetchone()[0]
                
        # Can also use with default values:
        @retry_db_op()
        def save_item(item_data):
            # database operations here
            pass
    """

    effective_retries = max_retries or config.DATABASE_RETRY_ATTEMPTS
    effective_delay = delay or config.DATABASE_RETRY_DELAY

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(effective_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < effective_retries - 1:
                        time.sleep(effective_delay)
                        continue
                    logging.error("Database error in %s: %s", func.__name__, e)
                    raise
                except Exception as e:  # pragma: no cover - unexpected path
                    logging.error("Unexpected error in %s: %s", func.__name__, e)
                    raise
            # Should be unreachable because we re-raise above
            return None  # pragma: no cover

        return wrapper

    return decorator

@retry_db_op()
def init_db() -> None:
    """Initialize the database with required tables, indexes, and migrations.
    
    Creates the prices table if it doesn't exist and applies any necessary
    schema migrations for backward compatibility. Enables WAL mode for
    better concurrent read/write performance.
    
    This function is idempotent - safe to call multiple times.
    
    Raises:
        sqlite3.Error: If database initialization fails (after retries).
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL;')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                name TEXT,
                timestamp DATETIME,
                flea_price INTEGER,
                trader_price INTEGER,
                trader_name TEXT,
                profit INTEGER,
                icon_link TEXT,
                width INTEGER,
                height INTEGER,
                avg_24h_price INTEGER,
                low_24h_price INTEGER,
                change_last_48h REAL,
                weight REAL,
                category TEXT,
                base_price INTEGER,
                high_24h_price INTEGER,
                last_offer_count INTEGER,
                short_name TEXT,
                wiki_link TEXT,
                trader_level_required INTEGER,
                trader_task_unlock TEXT,
                flea_level_required INTEGER DEFAULT 15,
                price_velocity REAL,
                liquidity_score REAL,
                api_updated TEXT
            )
        ''')
    
        # Check if icon_link column exists (migration for existing DB)
        try:
            c.execute('SELECT icon_link FROM prices LIMIT 1')
        except sqlite3.OperationalError:
            c.execute('ALTER TABLE prices ADD COLUMN icon_link TEXT')

        # Check if width/height columns exist (migration for existing DB)
        try:
            c.execute('SELECT width FROM prices LIMIT 1')
        except sqlite3.OperationalError:
            c.execute('ALTER TABLE prices ADD COLUMN width INTEGER DEFAULT 1')
            c.execute('ALTER TABLE prices ADD COLUMN height INTEGER DEFAULT 1')

        # Check if new advanced columns exist
        try:
            c.execute('SELECT avg_24h_price FROM prices LIMIT 1')
        except sqlite3.OperationalError:
            c.execute('ALTER TABLE prices ADD COLUMN avg_24h_price INTEGER DEFAULT 0')
            c.execute('ALTER TABLE prices ADD COLUMN low_24h_price INTEGER DEFAULT 0')
            c.execute('ALTER TABLE prices ADD COLUMN change_last_48h REAL DEFAULT 0.0')
            c.execute('ALTER TABLE prices ADD COLUMN weight REAL DEFAULT 0.0')
            c.execute('ALTER TABLE prices ADD COLUMN category TEXT DEFAULT "Unknown"')

        # Check if enhanced analysis columns exist (v2 migration)
        try:
            c.execute('SELECT base_price FROM prices LIMIT 1')
        except sqlite3.OperationalError:
            c.execute('ALTER TABLE prices ADD COLUMN base_price INTEGER DEFAULT 0')
            c.execute('ALTER TABLE prices ADD COLUMN high_24h_price INTEGER DEFAULT 0')
            c.execute('ALTER TABLE prices ADD COLUMN last_offer_count INTEGER DEFAULT 0')
            c.execute('ALTER TABLE prices ADD COLUMN short_name TEXT')
            c.execute('ALTER TABLE prices ADD COLUMN wiki_link TEXT')
            c.execute('ALTER TABLE prices ADD COLUMN trader_level_required INTEGER DEFAULT 1')
            c.execute('ALTER TABLE prices ADD COLUMN trader_task_unlock TEXT')
            c.execute('ALTER TABLE prices ADD COLUMN price_velocity REAL DEFAULT 0.0')
            c.execute('ALTER TABLE prices ADD COLUMN liquidity_score REAL DEFAULT 0.0')
            c.execute('ALTER TABLE prices ADD COLUMN api_updated TEXT')

        # Check if flea_level_required column exists (v3 migration for Patch 1.0)
        try:
            c.execute('SELECT flea_level_required FROM prices LIMIT 1')
        except sqlite3.OperationalError:
            c.execute('ALTER TABLE prices ADD COLUMN flea_level_required INTEGER DEFAULT 15')

        # Create indexes for performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices (timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_item_id ON prices (item_id)')
        # Composite index for market trends query optimization
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp_item ON prices (timestamp, item_id)')
        # Index for optimizing "latest per item" queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_item_timestamp ON prices (item_id, timestamp)')
        # Index for liquidity filtering
        c.execute('CREATE INDEX IF NOT EXISTS idx_prices_liquidity ON prices (liquidity_score)')

        # Optimize database
        c.execute('PRAGMA optimize;')

        conn.commit()
    finally:
        if conn:
            conn.close()

@retry_db_op()
def save_prices_batch(items: List[Tuple[Any, ...]]) -> None:
    """
    Save a list of items to the database in a single transaction.
    
    Args:
        items: List of tuples matching the database schema (15, 25, or 26 columns).
        
    Raises:
        ValueError: If items list contains tuples of inconsistent lengths.
    """
    if not items:
        return
    
    # Validate that items is a list of tuples
    if not isinstance(items, list) or not all(isinstance(item, tuple) for item in items):
        raise ValueError("items must be a list of tuples")
    
    # Validate consistent tuple lengths
    first_len = len(items[0])
    if not all(len(item) == first_len for item in items):
        raise ValueError("All items must have the same number of columns")
    
    # Validate tuple length is expected
    if first_len not in (15, 25, 26):
        logging.warning(
            "Unexpected item tuple length: %d. Expected 15 (legacy), 25 (v2), or 26 (v3 with flea level). "
            "This may indicate schema mismatch.",
            first_len
        )
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # Prepare the data for executemany
        # Expected tuple (26 columns): 
        # (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, 
        #  avg_24h_price, low_24h_price, change_last_48h, weight, category,
        #  base_price, high_24h_price, last_offer_count, short_name, wiki_link,
        #  trader_level_required, trader_task_unlock, flea_level_required, price_velocity, liquidity_score, api_updated)
        
        # Convert datetime objects to string to ensure consistency
        processed_items = []
        for item in items:
            item_list = list(item)
            # timestamp is at index 2
            if isinstance(item_list[2], datetime):
                item_list[2] = item_list[2].isoformat()
            processed_items.append(tuple(item_list))

        # Handle legacy format (15 columns), v2 format (25 columns), and v3 format (26 columns)
        if len(processed_items[0]) == 15:
            # Old format - backward compatibility
            c.executemany('''
                INSERT INTO prices (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', processed_items)
        elif len(processed_items[0]) == 25:
            # v2 enhanced format (without flea_level_required)
            c.executemany('''
                INSERT INTO prices (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, 
                    avg_24h_price, low_24h_price, change_last_48h, weight, category,
                    base_price, high_24h_price, last_offer_count, short_name, wiki_link,
                    trader_level_required, trader_task_unlock, price_velocity, liquidity_score, api_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', processed_items)
        else:
            # v3 format with flea_level_required
            c.executemany('''
                INSERT INTO prices (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, 
                    avg_24h_price, low_24h_price, change_last_48h, weight, category,
                    base_price, high_24h_price, last_offer_count, short_name, wiki_link,
                    trader_level_required, trader_task_unlock, flea_level_required, price_velocity, liquidity_score, api_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', processed_items)
        
        conn.commit()
    finally:
        if conn:
            conn.close()

@retry_db_op()
def get_latest_prices() -> List[Tuple[Any, ...]]:
    """Get the latest prices for all items.
    
    Uses a lookback window to capture items from recent collection cycles,
    handling cases where the API returns partial item lists.
    
    Returns:
        List of tuples containing price data for each item, sorted by profit.
        Empty list if no data available.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # 1. Get the absolute latest timestamp to establish a "current" anchor point
        c.execute('SELECT MAX(timestamp) FROM prices')
        result = c.fetchone()
        latest_ts_str = result[0] if result else None
        
        if not latest_ts_str:
            return []

        # 2. Calculate a lookback window relative to the latest data
        # This ensures we capture items from recent previous batches if the latest batch was partial
        # (e.g. API returned 2300 items instead of 4000), but doesn't show ancient data.
        latest_dt = parse_timestamp(latest_ts_str)
        
        if latest_dt is None:
            # Fallback: if parsing fails, use exact timestamp match (legacy behavior)
            c.execute('''
                SELECT item_id, name, flea_price, trader_price, trader_name, profit, timestamp, icon_link, width, height, 
                    avg_24h_price, low_24h_price, change_last_48h, weight, category,
                    base_price, high_24h_price, last_offer_count, short_name, wiki_link,
                    trader_level_required, trader_task_unlock, flea_level_required, price_velocity, liquidity_score
                FROM prices
                WHERE timestamp = ?
                ORDER BY profit DESC
            ''', (latest_ts_str,))
            rows = c.fetchall()
            return rows

        # Window allows for missed/partial collection cycles
        # This bridges the gap when the API returns partial lists.
        cutoff_dt = latest_dt - timedelta(minutes=config.DB_LOOKBACK_WINDOW_MINUTES)
        cutoff_ts_str = cutoff_dt.isoformat()

        c.execute('''
            SELECT 
                p.item_id, p.name, p.flea_price, p.trader_price, p.trader_name, p.profit, 
                p.timestamp, p.icon_link, p.width, p.height, p.avg_24h_price, 
                p.low_24h_price, p.change_last_48h, p.weight, p.category,
                p.base_price, p.high_24h_price, p.last_offer_count, p.short_name, p.wiki_link,
                p.trader_level_required, p.trader_task_unlock, p.flea_level_required, p.price_velocity, p.liquidity_score
            FROM prices p
            INNER JOIN (
                SELECT item_id, MAX(timestamp) as max_ts
                FROM prices
                WHERE timestamp >= ?
                GROUP BY item_id
            ) latest ON p.item_id = latest.item_id AND p.timestamp = latest.max_ts
            ORDER BY p.profit DESC
        ''', (cutoff_ts_str,))
        
        rows = c.fetchall()
        return rows
    finally:
        if conn:
            conn.close()

@retry_db_op()
def get_item_history(item_id: str) -> List[Tuple[Any, ...]]:
    """Get the price history for a specific item.
    
    Retrieves all historical records for an item, useful for trend analysis
    and price charting. Results are sorted chronologically (oldest first).
    
    Args:
        item_id: The unique identifier for the item (from API).
        
    Returns:
        List of tuples with (timestamp, flea_price, trader_price, profit).
        Empty list if no history found, item doesn't exist, or item_id is invalid.
        
    Example:
        >>> history = get_item_history('5c0a840b86f7742ffa4f2482')
        >>> for ts, flea, trader, profit in history:
        ...     print(f"{ts}: profit={profit}")
    """
    # Validate item_id - must be non-empty string
    if item_id is None or not isinstance(item_id, str):
        return []
    item_id = item_id.strip()
    if not item_id or item_id.isspace():
        return []
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, flea_price, trader_price, profit
            FROM prices
            WHERE item_id = ?
            ORDER BY timestamp ASC
        ''', (item_id,))
        rows = c.fetchall()
        return rows
    finally:
        if conn:
            conn.close()

@retry_db_op()
def get_market_trends(hours: int = 6) -> List[Tuple[Any, ...]]:
    """
    Calculate volatility and average profit over the last X hours.
    
    Args:
        hours: Number of hours to look back for trend data. Must be positive.
        
    Returns:
        List of tuples with (item_id, avg_profit, min_profit, max_profit, data_points).
        Empty list if no data found or hours is invalid.
        
    Note:
        Uses the latest timestamp in the database as the anchor point, which
        ensures calculations work correctly even with stale or gapped data.
    """
    # Validate hours parameter
    if not isinstance(hours, int) or hours <= 0:
        logging.warning("get_market_trends called with invalid hours: %s, using default 6", hours)
        hours = 6
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # Use the latest timestamp in the DB as the anchor, not current time.
        # This ensures calculations work even if the data is old (e.g. reviewing a dataset)
        # or if there are gaps in collection.
        c.execute('SELECT MAX(timestamp) FROM prices')
        result = c.fetchone()
        
        anchor_time = datetime.now()
        if result and result[0]:
            parsed = parse_timestamp(result[0])
            if parsed:
                anchor_time = parsed

        time_threshold = anchor_time - timedelta(hours=hours)
        
        c.execute('''
            SELECT 
                item_id, 
                AVG(profit) as avg_profit, 
                MIN(profit) as min_profit, 
                MAX(profit) as max_profit,
                COUNT(*) as data_points
            FROM prices
            WHERE timestamp > ?
            GROUP BY item_id
        ''', (time_threshold.isoformat(),))
        
        rows = c.fetchall()
        return rows
    finally:
        if conn:
            conn.close()

@retry_db_op()
def get_all_prices() -> List[Tuple[Any, ...]]:
    """Get all prices from the database.
    
    Returns:
        List of all price records as tuples with (item_id, name, flea_price,
        trader_price, trader_name, profit, timestamp).
        Empty list if no data found.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        c.execute('''
            SELECT item_id, name, flea_price, trader_price, trader_name, profit, timestamp
            FROM prices
        ''')
        rows = c.fetchall()
        return rows
    finally:
        if conn:
            conn.close()

@retry_db_op()
def cleanup_old_data(days: int = 7, vacuum: bool = False) -> int:
    """
    Delete records older than the specified number of days.
    
    Args:
        days: Number of days to retain data for (default: 7). Must be positive.
        vacuum: Whether to run VACUUM after deletion. Can lock database.
        
    Returns:
        Number of records deleted. Returns 0 if days is invalid.
    """
    # Validate days parameter
    if days <= 0:
        logging.warning("cleanup_old_data called with invalid days: %d, skipping", days)
        return 0
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        c.execute('DELETE FROM prices WHERE timestamp < ?', (cutoff_date.isoformat(),))
        # rowcount can be -1 if no count is available, so ensure we return non-negative int
        deleted_count = max(0, c.rowcount) if c.rowcount is not None else 0
        conn.commit()
        
        # Only vacuum if explicitly requested and we deleted a significant number of records.
        # VACUUM can be slow and locks the database, so only run when beneficial.
        if vacuum and deleted_count >= 1000:
            try:
                logging.info("Starting database VACUUM after deleting %d records...", deleted_count)
                c.execute('VACUUM')
                logging.info("Database VACUUM completed.")
            except sqlite3.OperationalError as e:
                logging.warning("Could not VACUUM database (locked?): %s, skipping.", e)
        elif vacuum and deleted_count > 0:
            logging.debug("Skipping VACUUM - only %d records deleted (threshold: 1000)", deleted_count)
                
        return deleted_count
    finally:
        if conn:
            conn.close()

@retry_db_op()
def get_latest_timestamp() -> Optional[datetime]:
    """Get the most recent timestamp from the prices table.
    
    This is useful for determining data freshness and rate limiting.
    
    Returns:
        datetime object of the most recent record, or None if no data exists
        or database doesn't exist yet.
        
    Raises:
        sqlite3.Error: If database connection or query fails (after retries).
    """
    # Return None if database doesn't exist yet
    if not os.path.exists(DB_NAME):
        return None
    
    with DatabaseConnection(readonly=True) as (conn, cursor):
        cursor.execute('SELECT MAX(timestamp) FROM prices')
        result = cursor.fetchone()
        
        if result and result[0]:
            return parse_timestamp(result[0])
        return None

@retry_db_op()
def clear_all_data() -> int:
    """
    Delete ALL data from the prices table.
    
    Warning:
        This operation cannot be undone. Use with caution.
        Consider backing up data before calling this function.
        
    Returns:
        Number of records deleted.
        
    Raises:
        sqlite3.Error: If database operation fails (after retries).
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        c.execute('DELETE FROM prices')
        deleted = max(0, c.rowcount) if c.rowcount is not None else 0
        conn.commit()
        
        try:
            c.execute('VACUUM')
        except sqlite3.OperationalError:
            logging.warning("Could not VACUUM database (locked?), skipping.")
        
        logging.info("Cleared all data: %d records deleted", deleted)
        return deleted
    finally:
        if conn:
            conn.close()


@retry_db_op()
def get_item_trend_data(item_ids: Optional[List[str]] = None, hours: int = 24) -> List[Tuple[Any, ...]]:
    """
    Get historical trend data for specified items or all items.
    
    Returns aggregated statistics per item over the time window including:
    - Count of data points
    - Average, min, max of profit
    - First and last observed timestamps
    - Average offer count
    
    Args:
        item_ids: Optional list of item IDs to filter. If None, returns all items.
                  Empty strings and None values in the list are filtered out.
        hours: Number of hours to look back for trend data. Must be positive.
        
    Returns:
        List of tuples with (item_id, data_points, avg_profit, min_profit, max_profit,
        avg_flea_price, avg_trader_price, avg_offers, first_seen, last_seen).
        Empty list if no data found.
    """
    # Validate hours parameter
    if not isinstance(hours, int) or hours <= 0:
        logging.warning("get_item_trend_data called with invalid hours: %s, using default 24", hours)
        hours = 24
    
    # Validate and sanitize item_ids
    if item_ids is not None:
        item_ids = [str(iid).strip() for iid in item_ids if iid and str(iid).strip()]
        if not item_ids:
            item_ids = None
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # Get anchor time from latest data
        c.execute('SELECT MAX(timestamp) FROM prices')
        result = c.fetchone()
        
        anchor_time = datetime.now()
        if result and result[0]:
            parsed = parse_timestamp(result[0])
            if parsed:
                anchor_time = parsed

        time_threshold = anchor_time - timedelta(hours=hours)
        
        if item_ids:
            placeholders = ','.join('?' * len(item_ids))
            query = f'''
                SELECT 
                    item_id,
                    COUNT(*) as data_points,
                    AVG(profit) as avg_profit,
                    MIN(profit) as min_profit,
                    MAX(profit) as max_profit,
                    AVG(flea_price) as avg_flea_price,
                    AVG(trader_price) as avg_trader_price,
                    AVG(last_offer_count) as avg_offers,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM prices
                WHERE timestamp > ? AND item_id IN ({placeholders})
                GROUP BY item_id
                HAVING COUNT(*) >= 2
            '''
            c.execute(query, [time_threshold.isoformat()] + item_ids)
        else:
            c.execute('''
                SELECT 
                    item_id,
                    COUNT(*) as data_points,
                    AVG(profit) as avg_profit,
                    MIN(profit) as min_profit,
                    MAX(profit) as max_profit,
                    AVG(flea_price) as avg_flea_price,
                    AVG(trader_price) as avg_trader_price,
                    AVG(last_offer_count) as avg_offers,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM prices
                WHERE timestamp > ?
                GROUP BY item_id
                HAVING COUNT(*) >= 2
            ''', (time_threshold.isoformat(),))
        
        rows = c.fetchall()
        return rows
    finally:
        if conn:
            conn.close()


@retry_db_op()
def get_profit_statistics(hours: int = 24) -> Dict[str, Any]:
    """
    Get overall profit statistics across all items for a time period.
    
    This helps calibrate the ML engine by understanding the distribution
    of profits in the dataset.
    
    Args:
        hours: Number of hours to analyze.
        
    Returns:
        Dict with overall statistics including total items, profit distribution, etc.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # Get anchor time
        c.execute('SELECT MAX(timestamp) FROM prices')
        result = c.fetchone()
        
        anchor_time = datetime.now()
        if result and result[0]:
            parsed = parse_timestamp(result[0])
            if parsed:
                anchor_time = parsed

        time_threshold = anchor_time - timedelta(hours=hours)
        
        c.execute('''
            SELECT 
                COUNT(DISTINCT item_id) as unique_items,
                COUNT(*) as total_records,
                AVG(profit) as avg_profit,
                MIN(profit) as min_profit,
                MAX(profit) as max_profit,
                AVG(flea_price) as avg_flea_price,
                AVG(last_offer_count) as avg_offers
            FROM prices
            WHERE timestamp > ?
        ''', (time_threshold.isoformat(),))
        
        row = c.fetchone()
        
        if row:
            return {
                'unique_items': row[0] or 0,
                'total_records': row[1] or 0,
                'avg_profit': row[2] or 0,
                'min_profit': row[3] or 0,
                'max_profit': row[4] or 0,
                'avg_flea_price': row[5] or 0,
                'avg_offers': row[6] or 0,
                'data_hours': hours
            }
        return {
            'unique_items': 0,
            'total_records': 0,
            'avg_profit': 0,
            'min_profit': 0,
            'max_profit': 0,
            'avg_flea_price': 0,
            'avg_offers': 0,
            'data_hours': hours
        }
    finally:
        if conn:
            conn.close()


@retry_db_op()
def get_database_health() -> Dict[str, Any]:
    """
    Get database health metrics for monitoring.
    
    Returns:
        Dict with database health information including:
        - file_size: Size of database file in bytes
        - total_records: Total number of records
        - oldest_record: Timestamp of oldest record
        - newest_record: Timestamp of newest record
        - unique_items: Number of unique items tracked
        - wal_size: Size of WAL file if exists
        - status: 'healthy', 'warning', or 'error'
    """
    health: Dict[str, Any] = {
        'status': 'healthy',
        'file_size': 0,
        'wal_size': 0,
        'total_records': 0,
        'unique_items': 0,
        'oldest_record': None,
        'newest_record': None,
        'data_age_hours': 0,
        'errors': [],
    }
    
    # Check file sizes
    try:
        if os.path.exists(DB_NAME):
            health['file_size'] = os.path.getsize(DB_NAME)
        
        wal_file = DB_NAME + '-wal'
        if os.path.exists(wal_file):
            health['wal_size'] = os.path.getsize(wal_file)
    except OSError as e:
        health['errors'].append(f"File access error: {e}")
    
    # Check database contents
    try:
        conn = sqlite3.connect(DB_NAME, timeout=config.DATABASE_CONNECTION_TIMEOUT)
        c = conn.cursor()
        
        # Get record counts
        c.execute('SELECT COUNT(*), COUNT(DISTINCT item_id) FROM prices')
        row = c.fetchone()
        if row:
            health['total_records'] = row[0] or 0
            health['unique_items'] = row[1] or 0
        
        # Get time range
        c.execute('SELECT MIN(timestamp), MAX(timestamp) FROM prices')
        row = c.fetchone()
        if row and row[0] and row[1]:
            health['oldest_record'] = row[0]
            health['newest_record'] = row[1]
            
            newest = parse_timestamp(row[1])
            if newest:
                age = datetime.now() - newest
                health['data_age_hours'] = age.total_seconds() / 3600
        
        conn.close()
        
    except sqlite3.Error as e:
        health['status'] = 'error'
        health['errors'].append(f"Database error: {e}")
    except Exception as e:
        health['status'] = 'error'
        health['errors'].append(f"Unexpected error: {e}")
    
    # Determine health status
    if health['errors']:
        health['status'] = 'error'
    elif health['total_records'] == 0:
        health['status'] = 'warning'
        health['errors'].append("No data in database")
    elif health['data_age_hours'] > 2:
        health['status'] = 'warning'
        health['errors'].append("Data may be stale (>2 hours old)")
    
    return health
