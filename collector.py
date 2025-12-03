import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import schedule
import time
import database
from datetime import datetime, timedelta
import logging
import sys
import signal
import argparse
import os
from typing import Optional, Dict, Any
from types import FrameType
import pandas as pd

import config
from ml_engine import get_ml_engine

# Configuration
# Loaded from config.py

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("collector.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def handle_exit(signum: int, frame: Optional[FrameType]) -> None:
    """Handle termination signals gracefully."""
    logging.info(f"Collector stopped by signal {signum}.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)
if sys.platform != 'win32':
    # SIGHUP is not available on Windows
    import signal as sig_module
    signal.signal(sig_module.SIGHUP, handle_exit)  # type: ignore[attr-defined]

# Session singleton for connection reuse
_session: Optional[requests.Session] = None

def get_session() -> requests.Session:
    """Get or create a session with retry configuration.
    
    Returns a singleton session to enable connection pooling and reuse.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        retry = Retry(
            total=5,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session

def run_query(query: str, variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Execute a GraphQL query against the Tarkov API.
    
    Args:
        query: GraphQL query string.
        variables: Optional query variables.
        
    Returns:
        JSON response dict or None on failure.
    """
    headers = {"Content-Type": "application/json"}
    session = get_session()
    try:
        payload: Dict[str, Any] = {'query': query}
        if variables:
            payload['variables'] = variables
            
        response = session.post(config.API_URL, headers=headers, json=payload, timeout=config.API_TIMEOUT_SECONDS)
        if response.status_code == 200:
            try:
                data = response.json()
                # Check for GraphQL errors
                if 'errors' in data:
                    logging.error(f"GraphQL errors: {data['errors']}")
                    return None
                return data
            except ValueError:
                logging.error("Failed to decode JSON response.")
                return None
        else:
            logging.error(f"Query failed with code {response.status_code}: {response.text[:200]}")
            return None
    except requests.Timeout:
        logging.error(f"API request timed out after {config.API_TIMEOUT_SECONDS} seconds.")
        return None
    except requests.ConnectionError:
        logging.error("API connection error. Check your internet connection.")
        return None
    except Exception as e:
        logging.error(f"Error querying API: {e}")
        return None

def fetch_and_store_data() -> None:
    start_time = time.time()
    logging.info("Fetching data...")
    
    query = """
    query GetItems($offset: Int, $limit: Int) {
        items(lang: en, offset: $offset, limit: $limit) {
            id
            name
            shortName
            width
            height
            basePrice
            avg24hPrice
            low24hPrice
            high24hPrice
            changeLast48hPercent
            weight
            iconLink
            wikiLink
            types
            lastOfferCount
            updated
            category {
                name
            }
            sellFor {
                price
                source
                currency
                vendor {
                    name
                    ... on TraderOffer {
                        minTraderLevel
                        taskUnlock {
                            id
                            name
                        }
                    }
                }
            }
            buyFor {
                price
                source
                currency
                vendor {
                    name
                    ... on TraderOffer {
                        minTraderLevel
                        taskUnlock {
                            id
                            name
                        }
                    }
                }
            }
        }
    }
    """
    
    all_items = []
    offset = 0
    limit = 1000
    
    while True:
        logging.info(f"Fetching items offset={offset} limit={limit}...")
        result = run_query(query, variables={"offset": offset, "limit": limit})
        
        if not result or 'data' not in result or 'items' not in result['data']:
            logging.warning("No data returned from API or error occurred.")
            break

        items = result['data']['items']
        if not items:
            break
            
        all_items.extend(items)
        
        if len(items) < limit:
            # Less items than limit means we reached the end
            break
            
        offset += limit
        # Be nice to the API
        time.sleep(0.5)

    logging.info(f"Fetched total {len(all_items)} items from API. Processing...")
    
    batch_data = []
    current_time = datetime.now()
    
    for item in all_items:
        # Filter out items explicitly marked as noFlea
        types = item.get('types', [])
        if 'noFlea' in types:
            continue

        item_id = item['id']
        name = item['name']
        short_name = item.get('shortName', name)
        icon_link = item.get('iconLink', '')
        wiki_link = item.get('wikiLink', '')
        width = item.get('width', 1)
        height = item.get('height', 1)
        base_price = item.get('basePrice', 0) or 0
        avg_24h_price = item.get('avg24hPrice', 0) or 0
        low_24h_price = item.get('low24hPrice', 0) or 0
        high_24h_price = item.get('high24hPrice', 0) or 0
        change_last_48h = item.get('changeLast48hPercent', 0.0) or 0.0
        weight = item.get('weight', 0.0) or 0.0
        last_offer_count = item.get('lastOfferCount', 0) or 0
        updated = item.get('updated', '')
        category_data = item.get('category')
        category = 'Unknown'
        if category_data and isinstance(category_data, dict):
            category = category_data.get('name', 'Unknown') or 'Unknown'
        
        # Find best Trader Sell Price (We sell to Trader)
        best_trader_price = 0
        best_trader_name = None
        trader_level_required = 1
        trader_task_unlock = None
        
        sell_offers = item.get('sellFor') or []
        for sell_offer in sell_offers:
            if not isinstance(sell_offer, dict):
                continue
            if sell_offer.get('currency') == 'RUB' and sell_offer.get('source') != 'fleaMarket':
                price = sell_offer.get('price')
                if price is not None and isinstance(price, (int, float)) and price > best_trader_price:
                    best_trader_price = price
                    best_trader_name = sell_offer.get('source')
                    # Extract trader level requirement
                    vendor = sell_offer.get('vendor', {})
                    if vendor:
                        trader_level_required = vendor.get('minTraderLevel', 1) or 1
                        task_data = vendor.get('taskUnlock')
                        if task_data:
                            trader_task_unlock = task_data.get('name')
        
        # Find best Flea Buy Price (We buy from Flea)
        best_flea_price = float('inf')
        
        buy_offers = item.get('buyFor') or []
        for buy_offer in buy_offers:
            if not isinstance(buy_offer, dict):
                continue
            if buy_offer.get('currency') == 'RUB' and buy_offer.get('source') == 'fleaMarket':
                price = buy_offer.get('price')
                if price is not None and isinstance(price, (int, float)) and price < best_flea_price:
                    best_flea_price = price
        
        # If we found valid prices for both
        if best_trader_price > 0 and best_flea_price != float('inf') and best_flea_price > 0:
            profit = best_trader_price - best_flea_price
            
            # Calculate price velocity (how much the price differs from avg - opportunity indicator)
            price_velocity = 0.0
            if avg_24h_price > 0:
                price_velocity = ((avg_24h_price - best_flea_price) / avg_24h_price) * 100
            
            # Calculate liquidity score based on offer count (more offers = easier to buy)
            # Normalize: 0-10 offers = low, 10-50 = medium, 50+ = high liquidity
            liquidity_score = min(last_offer_count / 50.0, 1.0) * 100 if last_offer_count else 0
            
            # Add to batch - now with enhanced data
            batch_data.append((
                item_id, name, current_time, best_flea_price, best_trader_price, 
                best_trader_name, profit, icon_link, width, height, 
                avg_24h_price, low_24h_price, change_last_48h, weight, category,
                # New fields
                base_price, high_24h_price, last_offer_count, short_name, wiki_link,
                trader_level_required, trader_task_unlock, price_velocity, liquidity_score,
                updated
            ))
            
    if batch_data:
        database.save_prices_batch(batch_data)
        duration = time.time() - start_time
        logging.info(f"Stored {len(batch_data)} items in {duration:.2f} seconds.")
        
        # Train the ML model on the new data
        try:
            train_model_on_batch(batch_data)
        except Exception as e:
            logging.warning(f"Model training failed (non-critical): {e}")
    else:
        logging.info("No profitable items found or API error.")


def train_model_on_batch(batch_data: list) -> None:
    """
    Train the ML model on the newly fetched batch of data.
    
    This allows the model to continuously learn and improve
    recommendations over time. The learned state persists even
    after database cleanup.
    
    Args:
        batch_data: List of tuples with item data.
    """
    # Convert batch data to DataFrame for training
    columns = [
        'item_id', 'name', 'timestamp', 'flea_price', 'trader_price',
        'trader_name', 'profit', 'icon_link', 'width', 'height',
        'avg_24h_price', 'low_24h_price', 'change_last_48h', 'weight', 'category',
        'base_price', 'high_24h_price', 'last_offer_count', 'short_name', 'wiki_link',
        'trader_level_required', 'trader_task_unlock', 'price_velocity', 'liquidity_score',
        'api_updated'
    ]
    
    df = pd.DataFrame(batch_data, columns=columns)
    
    # Calculate ROI for training
    df['roi'] = (df['profit'] / df['flea_price'] * 100).where(df['flea_price'] > 0, 0)
    
    # Get ML engine and train
    ml_engine = get_ml_engine()
    result = ml_engine.train_on_data(df, save=True)
    
    logging.info(f"Model training: {result['items_processed']} items, "
                f"{result['profitable_count']} profitable")


def cleanup_job() -> None:
    """Remove old data records to manage database size."""
    try:
        deleted = database.cleanup_old_data(days=config.DATA_RETENTION_DAYS, vacuum=False)
        if deleted > 0:
            logging.info(f"Cleaned up {deleted} old records.")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")


def job() -> None:
    """Main collection job - fetches and stores market data."""
    try:
        fetch_and_store_data()
    except Exception as e:
        logging.error(f"Job failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode (uses collector_standalone.pid)")
    args = parser.parse_args()

    pid_file = "collector_standalone.pid" if args.standalone else "collector.pid"
    
    # Write PID file
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    database.init_db()
    logging.info(f"Starting collector (Standalone: {args.standalone})...")
    
    try:
        # Check last run time to respect rate limits
        last_run = database.get_latest_timestamp()
        should_run_immediately = True
        
        if last_run:
            time_since_last = datetime.now() - last_run
            if time_since_last < timedelta(minutes=config.COLLECTION_INTERVAL_MINUTES):
                should_run_immediately = False
                logging.info(f"Last run was {time_since_last} ago. Skipping immediate run to respect {config.COLLECTION_INTERVAL_MINUTES}-minute rate limit.")
        
        if should_run_immediately:
            job()
            
        cleanup_job() # Run cleanup on startup
        
        schedule.every(config.COLLECTION_INTERVAL_MINUTES).minutes.do(job)
        schedule.every(24).hours.do(cleanup_job) # Run cleanup daily
        
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Collector stopped by user.")
    except Exception as e:
        logging.critical(f"Collector crashed: {e}")
    finally:
        if os.path.exists(pid_file):
            try:
                os.remove(pid_file)
            except OSError:
                pass
        sys.exit(0)
