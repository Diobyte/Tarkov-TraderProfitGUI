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
    
    c.executemany('''
        INSERT INTO prices (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', items)
    
    conn.commit()
    conn.close()

@retry_db_op()
def get_latest_prices() -> List[Tuple]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    # Get the latest timestamp
    c = conn.cursor()
    c.execute('SELECT MAX(timestamp) FROM prices')
    result = c.fetchone()
    latest_time = result[0] if result else None
    
    if not latest_time:
        conn.close()
        return []

    c.execute('''
        SELECT item_id, name, flea_price, trader_price, trader_name, profit, timestamp, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category
        FROM prices
        WHERE timestamp = ?
        ORDER BY profit DESC
    ''', (latest_time,))
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
    
    time_threshold = datetime.now() - timedelta(hours=hours)
    
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
def cleanup_old_data(days: int = 7) -> int:
    """
    Deletes records older than the specified number of days to keep the DB size manageable.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    cutoff_date = datetime.now() - timedelta(days=days)
    c.execute('DELETE FROM prices WHERE timestamp < ?', (cutoff_date,))
    deleted_count = c.rowcount
    conn.commit()
    
    # Only vacuum if we actually deleted something significant to avoid locking
    if deleted_count > 1000:
        try:
            c.execute('VACUUM')
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
