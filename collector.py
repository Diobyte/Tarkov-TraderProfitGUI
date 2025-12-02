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

import config

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

def handle_exit(signum: int, frame: Any) -> None:
    logging.info(f"Collector stopped by signal {signum}.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

def get_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def run_query(query: str, variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    session = get_session()
    try:
        payload: Dict[str, Any] = {'query': query}
        if variables:
            payload['variables'] = variables
            
        # Added timeout to prevent hanging indefinitely
        response = session.post(config.API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                logging.error("Failed to decode JSON response.")
                return None
        else:
            logging.error(f"Query failed with code {response.status_code}")
            return None
    except requests.Timeout:
        logging.error("API request timed out after 30 seconds.")
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
            width
            height
            avg24hPrice
            low24hPrice
            changeLast48hPercent
            weight
            iconLink
            types
            lastOfferCount
            category {
                name
            }
            sellFor {
                price
                source
                currency
            }
            buyFor {
                price
                source
                currency
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
        icon_link = item.get('iconLink', '')
        width = item.get('width', 1)
        height = item.get('height', 1)
        avg_24h_price = item.get('avg24hPrice', 0) or 0
        low_24h_price = item.get('low24hPrice', 0) or 0
        change_last_48h = item.get('changeLast48hPercent', 0.0) or 0.0
        weight = item.get('weight', 0.0) or 0.0
        category_data = item.get('category', {})
        category = category_data.get('name', 'Unknown') if category_data else 'Unknown'
        
        # Find best Trader Sell Price (We sell to Trader)
        best_trader_price = 0
        best_trader_name = None
        
        sell_offers = item.get('sellFor') or []
        for sell_offer in sell_offers:
            if sell_offer.get('currency') == 'RUB' and sell_offer.get('source') != 'fleaMarket':
                price = sell_offer.get('price', 0)
                if price is not None and price > best_trader_price:
                    best_trader_price = price
                    best_trader_name = sell_offer.get('source')
        
        # Find best Flea Buy Price (We buy from Flea)
        best_flea_price = float('inf')
        
        buy_offers = item.get('buyFor') or []
        for buy_offer in buy_offers:
            if buy_offer.get('currency') == 'RUB' and buy_offer.get('source') == 'fleaMarket':
                price = buy_offer.get('price', 0)
                if price is not None and price < best_flea_price:
                    best_flea_price = price
        
        # If we found valid prices for both
        if best_trader_price > 0 and best_flea_price != float('inf') and best_flea_price > 0:
            profit = best_trader_price - best_flea_price
            
            # Add to batch
            batch_data.append((
                item_id, name, current_time, best_flea_price, best_trader_price, 
                best_trader_name, profit, icon_link, width, height, 
                avg_24h_price, low_24h_price, change_last_48h, weight, category
            ))
            
    if batch_data:
        database.save_prices_batch(batch_data)
        duration = time.time() - start_time
        logging.info(f"Stored {len(batch_data)} items in {duration:.2f} seconds.")
    else:
        logging.info("No profitable items found or API error.")

def cleanup_job() -> None:
    try:
        # Retention set to manage DB size
        # Increase this if you need more historical data for ML, but be aware of DB size
        # We do NOT vacuum automatically here to prevent locking the DB for too long during collection cycles
        deleted = database.cleanup_old_data(days=config.DATA_RETENTION_DAYS, vacuum=False) 
        if deleted is not None and deleted > 0:
            logging.info(f"Cleaned up {deleted} old records.")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def job() -> None:
    try:
        fetch_and_store_data()
    except Exception as e:
        logging.error(f"Job failed: {e}")

# duplicate import removed

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
