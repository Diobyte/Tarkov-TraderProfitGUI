import sqlite3
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any

# Ensure DB is always created in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, 'tarkov_data.db')

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

def save_prices_batch(items: List[Tuple]) -> None:
    """
    Saves a list of items in a single transaction.
    items: list of tuples/dicts matching the schema
    """
    if not items:
        return
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
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
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise e

def get_latest_prices() -> List[Tuple]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    # Get the latest timestamp
    c = conn.cursor()
    c.execute('SELECT MAX(timestamp) FROM prices')
    latest_time = c.fetchone()[0]
    
    if not latest_time:
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

def get_market_trends(hours: int = 6) -> List[Tuple]:
    """
    Calculates volatility and average profit over the last X hours.
    Returns a dictionary keyed by item_id.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    
    # SQLite doesn't have built-in STDDEV, so we approximate or calculate variance manually if needed.
    # For simplicity in this environment, we'll fetch the data and calculate in Pandas in the app,
    # OR we can just get Min/Max/Avg here.
    # Let's get Min, Max, Avg Profit for the time window.
    
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
    # Vacuum to reclaim space
    c.execute('VACUUM')
    conn.close()
    return deleted_count

def get_latest_timestamp() -> Optional[datetime]:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    c = conn.cursor()
    try:
        c.execute('SELECT MAX(timestamp) FROM prices')
        result = c.fetchone()[0]
        conn.close()
        if result:
            # Handle potential format differences if any, but usually it's iso format string
            try:
                return datetime.fromisoformat(result)
            except ValueError:
                # Fallback for older python versions or different formats if needed
                # But fromisoformat is standard for what sqlite3 adapter stores
                return datetime.strptime(result, '%Y-%m-%d %H:%M:%S.%f')
        return None
    except Exception:
        conn.close()
        return None
