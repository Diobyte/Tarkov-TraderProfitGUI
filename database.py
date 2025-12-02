import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any, Callable
from functools import wraps

# Ensure DB is always created in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, 'tarkov_data.db')

def retry_db_op(max_retries: int = 5, delay: float = 1.0):
    """
    Decorator to retry database operations when locked.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower():
                        if attempt < max_retries - 1:
                            time.sleep(delay)
                            continue
                    logging.error(f"Database error in {func.__name__}: {e}")
                    raise e
                except Exception as e:
                    logging.error(f"Unexpected error in {func.__name__}: {e}")
                    raise e
            return None # Should not be reached if raise e is working
        return wrapper
    return decorator

@retry_db_op()
def init_db() -> None:
    conn = sqlite3.connect(DB_NAME, timeout=30)
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
            category TEXT
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

    # Create indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices (timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_prices_item_id ON prices (item_id)')
    # Composite index for market trends query optimization
    c.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp_item ON prices (timestamp, item_id)')
    # Index for optimizing "latest per item" queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_prices_item_timestamp ON prices (item_id, timestamp)')

    # Optimize database
    c.execute('PRAGMA optimize;')

    conn.commit()
    conn.close()

@retry_db_op()
def save_prices_batch(items: List[Tuple]) -> None:
    """
    Saves a list of items in a single transaction.
    items: list of tuples/dicts matching the schema
    """
    if not items:
        return
        
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    
    # Prepare the data for executemany
    # Expected tuple: (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
    
    # Convert datetime objects to string to ensure consistency
    processed_items = []
    for item in items:
        item_list = list(item)
        # timestamp is at index 2
        if isinstance(item_list[2], datetime):
            item_list[2] = item_list[2].isoformat()
        processed_items.append(tuple(item_list))

    c.executemany('''
        INSERT INTO prices (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', processed_items)
    
    conn.commit()
    conn.close()

@retry_db_op()
def get_latest_prices() -> List[Tuple]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    
    # 1. Get the absolute latest timestamp to establish a "current" anchor point
    c.execute('SELECT MAX(timestamp) FROM prices')
    result = c.fetchone()
    latest_ts_str = result[0] if result else None
    
    if not latest_ts_str:
        conn.close()
        return []

    # 2. Calculate a lookback window relative to the latest data
    # This ensures we capture items from recent previous batches if the latest batch was partial
    # (e.g. API returned 2300 items instead of 4000), but doesn't show ancient data.
    try:
        # Handle ISO format (most common)
        latest_dt = datetime.fromisoformat(latest_ts_str)
    except ValueError:
        try:
            # Fallback for legacy formats
            latest_dt = datetime.strptime(latest_ts_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # If parsing fails, fallback to exact match logic (old behavior)
            c.execute('''
                SELECT item_id, name, flea_price, trader_price, trader_name, profit, timestamp, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category
                FROM prices
                WHERE timestamp = ?
                ORDER BY profit DESC
            ''', (latest_ts_str,))
            rows = c.fetchall()
            conn.close()
            return rows

    # Window of 45 minutes allows for ~9 missed/partial collection cycles (assuming 5m interval)
    # This bridges the gap when the API returns partial lists.
    cutoff_dt = latest_dt - timedelta(minutes=45)
    cutoff_ts_str = cutoff_dt.isoformat()

    c.execute('''
        SELECT 
            p.item_id, p.name, p.flea_price, p.trader_price, p.trader_name, p.profit, 
            p.timestamp, p.icon_link, p.width, p.height, p.avg_24h_price, 
            p.low_24h_price, p.change_last_48h, p.weight, p.category
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
    conn.close()
    return rows

@retry_db_op()
def get_item_history(item_id: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    c.execute('''
        SELECT timestamp, flea_price, trader_price, profit
        FROM prices
        WHERE item_id = ?
        ORDER BY timestamp ASC
    ''', (item_id,))
    rows = c.fetchall()
    conn.close()
    return rows

@retry_db_op()
def get_market_trends(hours: int = 6) -> List[Tuple]:
    """
    Calculates volatility and average profit over the last X hours.
    Returns a dictionary keyed by item_id.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    
    # Use the latest timestamp in the DB as the anchor, not current time.
    # This ensures calculations work even if the data is old (e.g. reviewing a dataset)
    # or if there are gaps in collection.
    c.execute('SELECT MAX(timestamp) FROM prices')
    result = c.fetchone()
    
    anchor_time = datetime.now()
    if result and result[0]:
        try:
            # Try ISO format first
            anchor_time = datetime.fromisoformat(result[0])
        except ValueError:
            try:
                # Try legacy format
                anchor_time = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                pass

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
    ''', (time_threshold,))
    
    rows = c.fetchall()
    conn.close()
    return rows

@retry_db_op()
def get_all_prices() -> List[Tuple]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    c.execute('''
        SELECT item_id, name, flea_price, trader_price, trader_name, profit, timestamp
        FROM prices
    ''')
    rows = c.fetchall()
    conn.close()
    return rows

@retry_db_op()
def cleanup_old_data(days: int = 7, vacuum: bool = False) -> int:
    """
    Deletes records older than the specified number of days to keep the DB size manageable.
    Vacuuming is optional as it can lock the database for a long time.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    cutoff_date = datetime.now() - timedelta(days=days)
    c.execute('DELETE FROM prices WHERE timestamp < ?', (cutoff_date,))
    deleted_count = c.rowcount
    conn.commit()
    
    # Only vacuum if explicitly requested and we deleted something
    if vacuum and deleted_count > 1000:
        try:
            logging.info("Starting database VACUUM...")
            c.execute('VACUUM')
            logging.info("Database VACUUM completed.")
        except sqlite3.OperationalError:
            logging.warning("Could not VACUUM database (locked?), skipping.")
            
    conn.close()
    return deleted_count

@retry_db_op()
def get_latest_timestamp() -> Optional[datetime]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    c.execute('SELECT MAX(timestamp) FROM prices')
    result = c.fetchone()
    conn.close()
    
    if result and result[0]:
        ts_str = result[0]
        # Handle potential format differences
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            pass
        
        # Try common formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f', # ISO format with T
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
    return None

@retry_db_op()
def clear_all_data() -> None:
    """
    Deletes ALL data from the prices table. Use with caution.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    c.execute('DELETE FROM prices')
    conn.commit()
    
    try:
        c.execute('VACUUM')
    except sqlite3.OperationalError:
        logging.warning("Could not VACUUM database (locked?), skipping.")
        
    conn.close()
