"""
🎮 Tarkov Trader Profit Dashboard v3.0
The ultimate flea-to-trader arbitrage finder for Escape from Tarkov.
"""

import streamlit as st
import pandas as pd
import database
import utils
import config
from ml_engine import get_ml_engine
import time
import subprocess
import os
import sys
import logging
import plotly.express as px
import plotly.graph_objects as go
import psutil
from typing import Tuple, Optional
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8',
    force=True
)

# Initialize database
database.init_db()

# --- Constants ---
PID_FILE = "collector.pid"
STANDALONE_PID_FILE = "collector_standalone.pid"
COLORS = {
    'profit': '#00D26A',      # Vibrant green
    'loss': '#FF4757',        # Vibrant red
    'warning': '#FFA502',     # Orange
    'info': '#3742FA',        # Blue
    'accent': '#9C88FF',      # Purple
    'gold': '#FFD700',        # Gold
    'silver': '#C0C0C0',      # Silver
    'bronze': '#CD7F32',      # Bronze
    'bg_dark': '#0E1117',
    'bg_card': '#1a1a2e',
    'text': '#FAFAFA',
}

# =============================================================================
# COLLECTOR MANAGEMENT
# =============================================================================
def is_collector_running() -> Tuple[bool, Optional[int], Optional[str]]:
    """Check if the data collector is running."""
    for pid_file, mode in [(STANDALONE_PID_FILE, "standalone"), (PID_FILE, "session")]:
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                if psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    try:
                        if any("collector.py" in arg for arg in p.cmdline()):
                            return True, pid, mode
                    except (psutil.AccessDenied, psutil.ZombieProcess):
                        if "python" in p.name().lower():
                            return True, pid, mode
            except (OSError, ValueError, psutil.NoSuchProcess):
                pass
    return False, None, None

def start_collector() -> bool:
    """Start the data collector process."""
    if is_collector_running()[0]:
        return True
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open("collector_startup.log", "a") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "collector.py"],
                creationflags=flags, stdout=log, stderr=subprocess.STDOUT
            )
        time.sleep(2)
        if proc.poll() is None:
            with open(PID_FILE, 'w') as f:
                f.write(str(proc.pid))
            return True
    except Exception as e:
        logging.error("Failed to start collector: %s", e)
    return False

def stop_collector() -> bool:
    """Stop the data collector process."""
    running, pid, mode = is_collector_running()
    if not running:
        return True
    if mode == "standalone":
        return False
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            psutil.Process(pid).kill()
        except Exception:
            pass
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    return True

def find_all_collector_processes() -> list:
    """Find all running Python processes that are actual collector.py instances from this project."""
    collector_processes = []
    project_dir = os.path.dirname(os.path.abspath(__file__)).lower()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cwd']):
        try:
            pinfo = proc.info
            if pinfo['name'] and 'python' in pinfo['name'].lower():
                cmdline = pinfo.get('cmdline', []) or []
                cmdline_str = ' '.join(cmdline)
                
                # Check if this is actually running collector.py (not just has "collector" in path)
                is_collector = False
                
                # Method 1: Check if "collector.py" is explicitly in the command arguments
                for arg in cmdline:
                    if arg.endswith('collector.py'):
                        is_collector = True
                        break
                
                # Method 2: Check if running from our project directory
                if is_collector:
                    # Verify it's from our project, not some other collector.py
                    try:
                        proc_cwd = pinfo.get('cwd', '') or ''
                        if proc_cwd and project_dir in proc_cwd.lower():
                            pass  # Confirmed from our project
                        elif any(project_dir in arg.lower() for arg in cmdline):
                            pass  # Project path in cmdline
                        else:
                            # Check if the collector.py path contains our project
                            collector_path = next((arg for arg in cmdline if arg.endswith('collector.py')), '')
                            if collector_path and project_dir not in collector_path.lower():
                                # Not our project's collector - but still show it as it might be ours
                                # Just mark it differently
                                pass
                    except Exception:
                        pass
                
                if is_collector:
                    # Determine if it's from our project using cwd or command line hints
                    is_ours = False
                    proc_cwd = (pinfo.get('cwd', '') or '').lower()
                    if proc_cwd and project_dir in proc_cwd:
                        is_ours = True
                    elif any(project_dir in (arg or '').lower() for arg in cmdline):
                        is_ours = True
                    else:
                        # Fall back to checking the resolved collector.py path if cwd is missing
                        collector_path = next((arg for arg in cmdline if arg.endswith('collector.py')), '')
                        if collector_path:
                            resolved = collector_path
                            if not os.path.isabs(resolved) and proc_cwd:
                                resolved = os.path.abspath(os.path.join(proc_cwd, collector_path))
                            if project_dir in resolved.lower():
                                is_ours = True
                    
                    collector_processes.append({
                        'pid': pinfo['pid'],
                        'cmdline': cmdline_str[:100],
                        'create_time': datetime.fromtimestamp(pinfo['create_time']).strftime('%Y-%m-%d %H:%M:%S'),
                        'is_project': is_ours
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return collector_processes

def force_kill_all_collectors() -> Tuple[int, int]:
    """Force kill ALL collector processes. Returns (killed_count, failed_count)."""
    killed = 0
    failed = 0
    
    # First, clean up PID files
    for pid_file in [PID_FILE, STANDALONE_PID_FILE]:
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    proc = psutil.Process(pid)
                    proc.kill()
                    proc.wait(timeout=3)
                    killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                os.remove(pid_file)
            except (OSError, ValueError):
                pass
    
    # Then find and kill any remaining collector processes (only from this project)
    for proc_info in find_all_collector_processes():
        if not proc_info.get('is_project', False):
            continue  # Skip processes not from this project
        try:
            proc = psutil.Process(proc_info['pid'])
            proc.kill()
            proc.wait(timeout=3)
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            failed += 1
    
    return killed, failed

def get_database_stats() -> dict:
    """Get database file statistics."""
    db_path = database.DB_NAME
    stats: dict = {
        'exists': os.path.exists(db_path),
        'size_mb': 0.0,
        'size_str': '0 B',
        'path': db_path,
        'record_count': 0,
        'unique_items': 0,
        'oldest_record': None,
        'newest_record': None
    }
    
    if stats['exists']:
        size_bytes = os.path.getsize(db_path)
        stats['size_mb'] = size_bytes / (1024 * 1024)
        if size_bytes < 1024:
            stats['size_str'] = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            stats['size_str'] = f"{size_bytes / 1024:.1f} KB"
        else:
            stats['size_str'] = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=5)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM prices")
            stats['record_count'] = c.fetchone()[0]
            c.execute("SELECT COUNT(DISTINCT item_id) FROM prices")
            stats['unique_items'] = c.fetchone()[0]
            c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prices")
            row = c.fetchone()
            stats['oldest_record'] = row[0]
            stats['newest_record'] = row[1]
            conn.close()
        except Exception:
            pass
    
    return stats

def read_log_file(log_file: str, max_lines: int = config.LOG_MAX_LINES) -> str:
    """Read the last N lines from a log file efficiently.
    
    Uses a deque for memory-efficient reading of large log files.
    
    Args:
        log_file: Path to the log file to read.
        max_lines: Maximum number of lines to return (newest first).
    
    Returns:
        String containing the last max_lines of the log file, reversed.
    """
    from collections import deque
    
    if not os.path.exists(log_file):
        return f"Log file not found: {log_file}"
    
    try:
        # Use deque for memory-efficient tail reading
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            recent_lines = deque(f, maxlen=max_lines)
            # Return reversed so newest is first
            return ''.join(reversed(recent_lines))
    except PermissionError:
        return f"Permission denied: {log_file}"
    except Exception as e:
        return f"Error reading log: {e}"

def get_log_files() -> list:
    """Get list of available log files."""
    log_files = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_patterns = ['*.log', 'collector*.log', 'app*.log']
    for filename in os.listdir(base_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(base_dir, filename)
            try:
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                log_files.append({
                    'name': filename,
                    'path': filepath,
                    'size': size,
                    'modified': datetime.fromtimestamp(mtime)
                })
            except OSError:
                pass
    
    return sorted(log_files, key=lambda x: x['modified'], reverse=True)

# =============================================================================
# PAGE CONFIG & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Tarkov Profit Finder",
    layout="wide",
    page_icon="💰",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS for a sleek, modern look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 50%, #0E1117 100%);
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #1a1a2e;}
    ::-webkit-scrollbar-thumb {background: #4CAF50; border-radius: 4px;}
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FAFAFA !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2d3748;
        border-bottom: none;
        padding: 12px 24px;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
        color: #00D26A !important;
        border-color: #00D26A;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D26A 0%, #00B85C 100%);
        color: #0E1117;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,210,106,0.4);
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #1a1a2e;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        color: #FAFAFA;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #00D26A;
    }
    
    /* Custom classes */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D26A 0%, #00B85C 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    .hero-subtitle {
        color: #9CA3AF;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-live {
        background: rgba(0,210,106,0.2);
        color: #00D26A;
        border: 1px solid #00D26A;
    }
    
    .status-offline {
        background: rgba(255,71,87,0.2);
        color: #FF4757;
        border: 1px solid #FF4757;
    }
    
    .profit-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .item-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    
    .profit-amount {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00D26A;
    }
    
    .stat-label {
        color: #6B7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stat-value {
        color: #FAFAFA;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=config.STREAMLIT_CACHE_TTL_SECONDS)
def load_data() -> pd.DataFrame:
    """Load and process market data from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing processed market data with calculated metrics.
                     Returns empty DataFrame if no data available.
    """
    try:
        data = database.get_latest_prices()
        if not data:
            return pd.DataFrame()
        
        columns = [
            'item_id', 'name', 'flea_price', 'trader_price', 'trader_name', 'profit',
            'timestamp', 'icon_link', 'width', 'height', 'avg_24h_price', 'low_24h_price',
            'change_last_48h', 'weight', 'category', 'base_price', 'high_24h_price',
            'last_offer_count', 'short_name', 'wiki_link', 'trader_level_required',
            'trader_task_unlock', 'price_velocity', 'liquidity_score'
        ]
        
        # Handle old/new data formats
        if len(data[0]) == 15:
            df = pd.DataFrame(data, columns=columns[:15])
            for col in columns[15:]:
                if col == 'short_name':
                    df[col] = df['name']
                elif col == 'wiki_link' or col == 'trader_task_unlock':
                    df[col] = ''
                else:
                    df[col] = 0
        elif len(data[0]) >= 24:
            # Use available columns
            df = pd.DataFrame(data, columns=columns[:len(data[0])])
            # Add any missing columns
            for col in columns[len(data[0]):]:
                if col == 'short_name':
                    df[col] = df['name']
                elif col in ('wiki_link', 'trader_task_unlock'):
                    df[col] = ''
                else:
                    df[col] = 0
        else:
            df = pd.DataFrame(data, columns=columns)
        
        # Add trend data
        trends = database.get_market_trends(hours=168)
        if trends:
            trend_df = pd.DataFrame(trends, columns=['item_id', 'trend_avg', 'trend_min', 'trend_max', 'data_points'])
            df = df.merge(trend_df, on='item_id', how='left')
            df['volatility'] = (df['trend_max'].fillna(df['profit']) - df['trend_min'].fillna(df['profit'])).fillna(0)
            df['data_points'] = df['data_points'].fillna(0)
        else:
            df['volatility'] = 0
            df['data_points'] = 0
        
        # Calculate metrics
        df = utils.calculate_metrics(df)
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        return pd.DataFrame()

def get_filtered_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply user-defined filters to the market dataframe.
    
    Args:
        df: The source DataFrame containing market data.
        filters: Dictionary containing filter criteria:
            - min_profit: Minimum profit threshold
            - min_roi: Minimum ROI percentage
            - search: Item name search string
            - category: Category filter ('All' or specific category)
            - show_negative: Whether to include negative profit items
            - min_offers: Minimum offer count (default: 5)
            - min_profit_per_slot: Minimum profit per inventory slot
            - max_trader_level: Maximum allowed trader level
    
    Returns:
        pd.DataFrame: Filtered DataFrame matching all criteria.
    """
    if df.empty:
        return df
    
    # Ensure required columns exist
    required_cols = ['profit', 'roi', 'last_offer_count', 'name', 'category']
    for col in required_cols:
        if col not in df.columns:
            if col in ('profit', 'roi', 'last_offer_count'):
                df[col] = 0
            else:
                df[col] = ''
    
    # Get minimum offers from filters, default to config value
    min_offers = filters.get('min_offers', config.VOLUME_MIN_FOR_RECOMMENDATION)
    min_profit_per_slot = filters.get('min_profit_per_slot', 0)
    max_trader_level = filters.get('max_trader_level', None)
    
    # Ensure profit_per_slot column exists
    if 'profit_per_slot' not in df.columns:
        df['profit_per_slot'] = 0
    
    filtered = df[
        (df['profit'] >= filters['min_profit']) &
        (df['roi'] >= filters['min_roi']) &
        (df['last_offer_count'] >= min_offers) &
        (df['profit_per_slot'] >= min_profit_per_slot)
    ]
    
    if filters['search']:
        filtered = filtered[filtered['name'].str.contains(filters['search'], case=False, na=False)]
    
    if filters['category'] != 'All':
        filtered = filtered[filtered['category'] == filters['category']]

    if max_trader_level is not None and 'trader_level_required' in filtered.columns:
        filtered = filtered[filtered['trader_level_required'] <= max_trader_level]
    
    if not filters['show_negative']:
        filtered = filtered[filtered['profit'] > 0]
    
    return filtered

# =============================================================================
# HEADER COMPONENT
# =============================================================================
def render_header() -> None:
    """Render the hero header section with status indicators."""
    st.markdown('<h1 class="hero-title">💰 TARKOV PROFIT FINDER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Buy from Flea → Sell to Traders → Stack Roubles</p>', unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1.2])
    
    with col1:
        running, pid, mode = is_collector_running()
        if running:
            st.markdown('<span class="status-badge status-live">● LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">○ OFFLINE</span>', unsafe_allow_html=True)
    
    with col2:
        last_update = database.get_latest_timestamp()
        if last_update is not None:
            time_ago = datetime.now() - last_update
            total_mins = int(time_ago.total_seconds() / 60)
            if total_mins < 1:
                st.caption("📡 Updated just now")
            elif total_mins < 60:
                st.caption(f"📡 Updated {total_mins}m ago")
            else:
                st.caption(f"📡 Updated {total_mins // 60}h {total_mins % 60}m ago")
        else:
            st.caption("📡 No data yet")
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Database & ML health indicator
    with col4:
        try:
            db_health = database.get_database_health()
        except Exception as e:  # pragma: no cover - defensive
            logging.error("Failed to get database health: %s", e)
            db_health = {
                'status': 'error',
                'data_age_hours': 0,
                'total_records': 0,
                'errors': ['Failed to read database health'],
            }

        from ml_engine import get_ml_engine  # local import to avoid cycles

        try:
            ml_engine = get_ml_engine()
            learning_status = ml_engine.get_trend_learning_status()
        except Exception as e:  # pragma: no cover - defensive
            logging.error("Failed to get ML learning status: %s", e)
            learning_status = {
                'enabled': False,
                'learning_quality': 0,
                'items_with_history': 0,
            }

        status = str(db_health.get('status', 'unknown')).lower()
        color = '#00D26A' if status == 'healthy' else ('#FFA502' if status == 'warning' else '#FF4757')
        age_hours = float(db_health.get('data_age_hours', 0) or 0)
        ml_quality = float(learning_status.get('learning_quality', 0) or 0)

        st.markdown(
            f"""
            <div style="border-radius: 10px; padding: 8px 10px; border: 1px solid #2d3748; background: #111827;">
              <div style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.06em;">System Health</div>
              <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span style="font-size: 0.85rem; color: {color}; font-weight: 600;">DB: {status.title()}</span>
                <span style="font-size: 0.75rem; color: #9CA3AF;">Age: {age_hours:.1f}h</span>
              </div>
              <div style="margin-top: 4px; font-size: 0.75rem; color: #9CA3AF;">
                ML Learning: {ml_quality:.0f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# STATS OVERVIEW
# =============================================================================
def render_stats(df: pd.DataFrame) -> None:
    """Render the main statistics overview with key metrics.
    
    Args:
        df: DataFrame containing filtered market data.
    """
    if df.empty:
        st.warning("No data available. Start the collector to begin tracking prices.")
        return
    
    profitable = df[df['profit'] > 0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Items",
            f"{len(df):,}",
            help="Items being tracked"
        )
    
    with col2:
        st.metric(
            "Profitable",
            f"{len(profitable):,}",
            delta=f"{len(profitable)/len(df)*100:.0f}%" if len(df) > 0 else "0%",
            help="Items with positive profit"
        )
    
    with col3:
        best_profit = df['profit'].max() if not df.empty else 0
        st.metric(
            "Best Flip",
            f"₽{best_profit:,.0f}",
            help="Highest profit item"
        )
    
    with col4:
        avg_roi = profitable['roi'].mean() if len(profitable) > 0 else 0
        st.metric(
            "Avg ROI",
            f"{avg_roi:.1f}%",
            help="Average return on profitable items"
        )
    
    with col5:
        total_potential = profitable['profit'].sum() if len(profitable) > 0 else 0
        st.metric(
            "Total Potential",
            f"₽{total_potential:,.0f}",
            help="Sum of all profitable flips"
        )

# =============================================================================
# TOP OPPORTUNITIES CARDS
# =============================================================================
def render_top_opportunities(df: pd.DataFrame) -> None:
    """Render top 6 trading opportunities as visual cards with ML insights.
    
    Uses persistent ML learning to rank opportunities by learned
    profitability, consistency, and trend direction.
    
    Args:
        df: DataFrame containing filtered market data sorted by profit.
    """
    if df.empty:
        return

    st.markdown("### 🏆 Top ML-Ranked Opportunities")
    st.caption("Ranked by learned profitability, consistency, and market trends. ML Score blends profit, ROI, liquidity, and historical consistency.")
    
    # Filter to reliable volume items only (>= 5 offers)
    reliable_df = df[df['last_offer_count'] >= config.VOLUME_MIN_FOR_RECOMMENDATION]
    if reliable_df.empty:
        st.info("No items with sufficient market volume found.")
        return
    
    # Use ML engine for ranking
    ml_engine = get_ml_engine()
    
    try:
        # Get ML-ranked recommendations
        ml_df = ml_engine.enrich_with_learned_data(reliable_df.copy())
        ml_df = ml_engine.calculate_opportunity_score_ml(ml_df)
        
        # Sort by ML score instead of just profit
        if 'ml_opportunity_score' in ml_df.columns:
            top = ml_df.nlargest(6, 'ml_opportunity_score')
        else:
            top = reliable_df.nlargest(6, 'profit')
    except Exception as e:
        logging.warning("ML ranking failed, using profit: %s", e)
        top = reliable_df.nlargest(6, 'profit')
    
    cols = st.columns(3)
    for i, (_, item) in enumerate(top.iterrows()):
        with cols[i % 3]:
            # Medal for top 3
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "💎"
            
            profit_color = COLORS['profit'] if item['profit'] > 0 else COLORS['loss']
            
            # Get trend indicator if available
            trend_icon = ""
            if 'learned_data_points' in item and item.get('learned_data_points', 0) >= 6:
                learned_score = item.get('learned_score_adjustment', 0)
                if learned_score > 5:
                    trend_icon = " 📈"
                elif learned_score < -5:
                    trend_icon = " 📉"
                else:
                    trend_icon = " ➡️"
            
            # Get ML score if available
            ml_score = item.get('ml_opportunity_score', 0)
            consistency = item.get('learned_consistency', 50)
            profit_per_slot = item.get('profit_per_slot', 0)
            trader_level = item.get('trader_level_required', 1)
            
            st.markdown(f"""
            <div class="profit-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5rem;">{medal}</span>
                    <span class="profit-amount" style="color: {profit_color};">+₽{item['profit']:,.0f}{trend_icon}</span>
                </div>
                <div class="item-name">{item['name'][:35]}{'...' if len(item['name']) > 35 else ''}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 12px;">
                    <div>
                        <div class="stat-label">Buy</div>
                        <div class="stat-value">₽{item['flea_price']:,.0f}</div>
                    </div>
                    <div>
                        <div class="stat-label">Sell</div>
                        <div class="stat-value">₽{item['trader_price']:,.0f}</div>
                    </div>
                    <div>
                        <div class="stat-label">ML Score</div>
                        <div class="stat-value" style="color: {profit_color};">{ml_score:.0f}</div>
                    </div>
                </div>
                <div style="margin-top: 8px; color: #6B7280; font-size: 0.8rem;">
                    {item['trader_name']} (L{trader_level}) • {item['category']} • {item.get('last_offer_count', 0):.0f} offers
                </div>
                <div style="margin-top: 4px; color: #9CA3AF; font-size: 0.75rem;">
                    Profit/slot: ₽{profit_per_slot:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# MAIN DATA TABLE
# =============================================================================
def render_data_table(df: pd.DataFrame) -> None:
    """Render the main data table with top 50 ML-recommended trades.
    
    Uses persistent learning to rank items by learned profitability
    and consistency rather than just current profit.
    
    Args:
        df: DataFrame containing filtered market data.
    """
    if df.empty:
        st.info("No items match your filters.")
        return

    st.markdown("### 🏆 Top ML-Recommended Trades")
    st.caption(
        f"Items with ≥{config.VOLUME_MIN_FOR_RECOMMENDATION} offers, ranked by learned profitability and consistency. "
        "ML Score blends profit, ROI, liquidity, and historical consistency."
    )
    
    # Filter to items with reliable volume
    reliable_df = df[df['last_offer_count'] >= config.VOLUME_MIN_FOR_RECOMMENDATION]
    
    if reliable_df.empty:
        st.warning("No items with sufficient market volume. Try adjusting filters.")
        return
    
    # Enrich with ML data
    ml_engine = get_ml_engine()
    
    try:
        ml_df = ml_engine.enrich_with_learned_data(reliable_df.copy())
        ml_df = ml_engine.calculate_opportunity_score_ml(ml_df)
        
        # Add trend indicator column
        def get_trend_emoji(row):
            learned_adj = row.get('learned_score_adjustment', 0)
            if learned_adj > 5:
                return '📈'
            elif learned_adj < -5:
                return '📉'
            elif row.get('learned_data_points', 0) >= 6:
                return '➡️'
            else:
                return '🆕'
        
        ml_df['trend'] = ml_df.apply(get_trend_emoji, axis=1)
        
        # Sort by ML score
        if 'ml_opportunity_score' in ml_df.columns:
            ml_df = ml_df.sort_values('ml_opportunity_score', ascending=False)
        else:
            ml_df = ml_df.sort_values('profit', ascending=False)
    except Exception as e:
        logging.warning("ML enrichment failed: %s", e)
        ml_df = reliable_df.copy()
        ml_df['trend'] = '❓'
        ml_df = ml_df.sort_values('profit', ascending=False)
    
    # Prepare display dataframe
    display_cols = ['icon_link', 'name', 'trend', 'profit', 'roi', 'flea_price', 
                   'trader_price', 'trader_name', 'category', 'last_offer_count']
    
    # Add ML score if available
    if 'ml_opportunity_score' in ml_df.columns:
        display_cols.insert(4, 'ml_opportunity_score')
    
    # Only include columns that exist
    display_cols = [c for c in display_cols if c in ml_df.columns]
    
    display_df = ml_df[display_cols].head(50)
    
    # Calculate dynamic max ROI for progress bar (ROI can exceed 100%)
    max_roi = max(display_df['roi'].max() if 'roi' in display_df.columns else 100, 100)
    
    column_config = {
        "icon_link": st.column_config.ImageColumn("", width=50),
        "name": st.column_config.TextColumn("Item", width=180),
        "trend": st.column_config.TextColumn("Trend", width=50),
        "profit": st.column_config.NumberColumn("Profit", format="₽%d", width=90),
        "roi": st.column_config.ProgressColumn("ROI", format="%.1f%%", min_value=0, max_value=max_roi, width=90),
        "flea_price": st.column_config.NumberColumn("Flea", format="₽%d", width=90),
        "trader_price": st.column_config.NumberColumn("Trader", format="₽%d", width=90),
        "trader_name": st.column_config.TextColumn("Trader", width=80),
        "category": st.column_config.TextColumn("Category", width=100),
        "last_offer_count": st.column_config.NumberColumn("Offers", width=65),
    }
    
    if 'ml_opportunity_score' in display_cols:
        column_config["ml_opportunity_score"] = st.column_config.ProgressColumn(
            "ML Score", format="%.0f", min_value=0, max_value=100, width=80
        )
    
    st.dataframe(
        display_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=500
    )

# =============================================================================
# MARKET EXPLORER - FULL DATABASE WITH SELF-FILTERING
# =============================================================================
def render_market_explorer(df: pd.DataFrame) -> None:
    """Render a comprehensive market explorer with built-in filtering.
    
    Args:
        df: Full unfiltered DataFrame containing all market data.
    """
    if df.empty:
        st.info("No market data available.")
        return
    
    st.markdown("### 🌐 Complete Market Explorer")
    st.caption("Browse and filter ALL items in the flea market database.")
    
    # Built-in filters for this table
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        explorer_search = st.text_input("🔍 Search Items", placeholder="Item name...", key="explorer_search")
    
    with filter_col2:
        categories = ['All Categories'] + sorted(df['category'].dropna().unique().tolist())
        explorer_cat = st.selectbox("Category", categories, key="explorer_cat")
    
    with filter_col3:
        traders = ['All Traders'] + sorted(df['trader_name'].dropna().unique().tolist())
        explorer_trader = st.selectbox("Trader", traders, key="explorer_trader")
    
    with filter_col4:
        sort_options = {
            'Profit (High→Low)': ('profit', False),
            'Profit (Low→High)': ('profit', True),
            'ROI (High→Low)': ('roi', False),
            'Flea Price (High→Low)': ('flea_price', False),
            'Flea Price (Low→High)': ('flea_price', True),
            'Name (A→Z)': ('name', True),
            'Offers (High→Low)': ('last_offer_count', False),
        }
        sort_by = st.selectbox("Sort By", list(sort_options.keys()), key="explorer_sort")
    
    # Price range filters
    price_col1, price_col2, price_col3, price_col4 = st.columns(4)
    
    with price_col1:
        min_flea = st.number_input("Min Flea Price", value=0, step=10000, key="min_flea")
    with price_col2:
        max_flea = st.number_input("Max Flea Price", value=int(df['flea_price'].max()) if not df.empty else 10000000, step=10000, key="max_flea")
    with price_col3:
        min_profit_explorer = st.number_input("Min Profit", value=-1000000, step=1000, key="min_profit_explorer")
    with price_col4:
        show_all_profits = st.checkbox("Include Negative Profits", value=True, key="show_all_explorer")
    
    # Apply filters
    explorer_df = df.copy()
    
    if explorer_search:
        explorer_df = explorer_df[explorer_df['name'].str.contains(explorer_search, case=False, na=False)]
    
    if explorer_cat != 'All Categories':
        explorer_df = explorer_df[explorer_df['category'] == explorer_cat]
    
    if explorer_trader != 'All Traders':
        explorer_df = explorer_df[explorer_df['trader_name'] == explorer_trader]
    
    explorer_df = explorer_df[
        (explorer_df['flea_price'] >= min_flea) &
        (explorer_df['flea_price'] <= max_flea) &
        (explorer_df['profit'] >= min_profit_explorer)
    ]
    
    if not show_all_profits:
        explorer_df = explorer_df[explorer_df['profit'] > 0]
    
    # Sort
    sort_col, sort_asc = sort_options[sort_by]
    explorer_df = explorer_df.sort_values(sort_col, ascending=sort_asc)
    
    # Display stats
    st.markdown(f"**Showing {len(explorer_df):,} of {len(df):,} items**")
    
    # Full table with all columns
    display_cols = [
        'icon_link', 'name', 'profit', 'roi', 'flea_price', 'trader_price',
        'trader_name', 'category', 'avg_24h_price', 'low_24h_price', 'high_24h_price',
        'change_last_48h', 'last_offer_count', 'weight'
    ]
    
    # Only include columns that exist
    display_cols = [c for c in display_cols if c in explorer_df.columns]
    
    st.dataframe(
        explorer_df[display_cols],
        column_config={
            "icon_link": st.column_config.ImageColumn("", width=50),
            "name": st.column_config.TextColumn("Item", width=180),
            "profit": st.column_config.NumberColumn("Profit", format="₽%d", width=90),
            "roi": st.column_config.NumberColumn("ROI", format="%.1f%%", width=70),
            "flea_price": st.column_config.NumberColumn("Flea", format="₽%d", width=90),
            "trader_price": st.column_config.NumberColumn("Trader", format="₽%d", width=90),
            "trader_name": st.column_config.TextColumn("Sell To", width=80),
            "category": st.column_config.TextColumn("Category", width=100),
            "avg_24h_price": st.column_config.NumberColumn("24h Avg", format="₽%d", width=90),
            "low_24h_price": st.column_config.NumberColumn("24h Low", format="₽%d", width=85),
            "high_24h_price": st.column_config.NumberColumn("24h High", format="₽%d", width=85),
            "change_last_48h": st.column_config.NumberColumn("48h Δ%", format="%.1f%%", width=70),
            "last_offer_count": st.column_config.NumberColumn("Offers", width=65),
            "weight": st.column_config.NumberColumn("Weight", format="%.2f kg", width=75),
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    # Export option
    st.download_button(
        "📥 Export to CSV",
        explorer_df.to_csv(index=False).encode('utf-8'),
        "tarkov_market_data.csv",
        "text/csv",
        use_container_width=False
    )

# =============================================================================
# VISUAL MARKET ANALYTICS
# =============================================================================
def render_visual_analytics(df: pd.DataFrame) -> None:
    """Render comprehensive visual analytics for the entire market.
    
    Args:
        df: Full DataFrame containing all market data for analysis.
    """
    if df.empty or len(df) < 5:
        st.info("Not enough data for visual analytics.")
        return
    
    st.markdown("### 🎨 Flea Market Visual Analytics")
    st.caption("Comprehensive market visualizations and insights.")
    
    viz_tabs = st.tabs([
        "📊 Price Distribution", 
        "🥧 Category Analysis",
        "📈 Trader Breakdown",
        "🌡️ Market Heatmap",
        "📉 Trends & Patterns"
    ])
    
    # --- Price Distribution Tab ---
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Flea price distribution histogram
            fig = px.histogram(
                df[df['flea_price'] < df['flea_price'].quantile(0.95)],
                x='flea_price',
                nbins=50,
                title='Flea Market Price Distribution',
                template='plotly_dark',
                color_discrete_sequence=[COLORS['profit']]
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                xaxis_title='Price (₽)',
                yaxis_title='Number of Items',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit distribution
            fig = px.histogram(
                df,
                x='profit',
                nbins=50,
                title='Profit Distribution (All Items)',
                template='plotly_dark',
                color_discrete_sequence=[COLORS['warning']]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                xaxis_title='Profit (₽)',
                yaxis_title='Number of Items',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plots
        col3, col4 = st.columns(2)
        
        with col3:
            # Price box plot by category (top 8)
            top_cats = df['category'].value_counts().head(8).index.tolist()
            cat_df = df[df['category'].isin(top_cats)]
            
            fig = px.box(
                cat_df,
                x='category',
                y='flea_price',
                title='Price Range by Category',
                template='plotly_dark',
                color='category'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                showlegend=False,
                height=350
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # ROI vs Price scatter
            fig = px.scatter(
                df[df['profit'] > 0],
                x='flea_price',
                y='roi',
                size='profit',
                color='category',
                hover_name='name',
                title='ROI vs Price (Profitable Items)',
                template='plotly_dark',
                size_max=30
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Category Analysis Tab ---
    with viz_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Total items per category
            cat_counts = df.groupby('category').size().reset_index(name='count')
            cat_counts = cat_counts.sort_values('count', ascending=True).tail(12)
            
            fig = px.bar(
                cat_counts,
                x='count',
                y='category',
                orientation='h',
                title='Items per Category',
                template='plotly_dark',
                color='count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                coloraxis_showscale=False,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average profit by category
            cat_profit = df.groupby('category').agg({
                'profit': 'mean',
                'roi': 'mean',
                'name': 'count'
            }).round(0).reset_index()
            cat_profit.columns = ['Category', 'Avg Profit', 'Avg ROI', 'Items']
            cat_profit = cat_profit.sort_values('Avg Profit', ascending=True).tail(12)
            
            fig = px.bar(
                cat_profit,
                x='Avg Profit',
                y='Category',
                orientation='h',
                title='Average Profit by Category',
                template='plotly_dark',
                color='Avg Profit',
                color_continuous_scale=['#FF5252', '#FFD700', '#4CAF50']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                coloraxis_showscale=False,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category summary table
        st.markdown("#### Category Performance Summary")
        cat_summary = df.groupby('category').agg({
            'profit': ['sum', 'mean', 'max'],
            'roi': 'mean',
            'flea_price': 'mean',
            'name': 'count'
        }).round(0)
        cat_summary.columns = ['Total Profit', 'Avg Profit', 'Max Profit', 'Avg ROI', 'Avg Price', 'Items']
        cat_summary = cat_summary.sort_values('Total Profit', ascending=False)
        
        st.dataframe(
            cat_summary,
            column_config={
                "Total Profit": st.column_config.NumberColumn(format="₽%d"),
                "Avg Profit": st.column_config.NumberColumn(format="₽%d"),
                "Max Profit": st.column_config.NumberColumn(format="₽%d"),
                "Avg ROI": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Price": st.column_config.NumberColumn(format="₽%d"),
            },
            use_container_width=True,
            height=400
        )
    
    # --- Trader Breakdown Tab ---
    with viz_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Items by trader pie
            trader_counts = df.groupby('trader_name').size().reset_index(name='count')
            
            fig = px.pie(
                trader_counts,
                values='count',
                names='trader_name',
                title='Items by Best Trader',
                hole=0.4,
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total profit potential by trader
            trader_profit = df[df['profit'] > 0].groupby('trader_name').agg({
                'profit': 'sum',
                'name': 'count'
            }).reset_index()
            trader_profit.columns = ['Trader', 'Total Profit', 'Profitable Items']
            
            fig = px.bar(
                trader_profit.sort_values('Total Profit', ascending=True),
                x='Total Profit',
                y='Trader',
                orientation='h',
                title='Total Profit Potential by Trader',
                template='plotly_dark',
                color='Total Profit',
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trader stats table
        st.markdown("#### Trader Statistics")
        trader_stats = df.groupby('trader_name').agg({
            'profit': ['mean', 'max', 'sum'],
            'roi': 'mean',
            'trader_price': 'mean',
            'name': 'count'
        }).round(0)
        trader_stats.columns = ['Avg Profit', 'Best Profit', 'Total Profit', 'Avg ROI', 'Avg Payout', 'Items']
        trader_stats = trader_stats.sort_values('Total Profit', ascending=False)
        
        st.dataframe(
            trader_stats,
            column_config={
                "Avg Profit": st.column_config.NumberColumn(format="₽%d"),
                "Best Profit": st.column_config.NumberColumn(format="₽%d"),
                "Total Profit": st.column_config.NumberColumn(format="₽%d"),
                "Avg ROI": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Payout": st.column_config.NumberColumn(format="₽%d"),
            },
            use_container_width=True
        )
    
    # --- Market Heatmap Tab ---
    with viz_tabs[3]:
        st.markdown("#### Category × Trader Profit Heatmap")
        
        # Create pivot table for heatmap
        top_cats = df['category'].value_counts().head(10).index.tolist()
        heatmap_df = df[df['category'].isin(top_cats)]
        
        pivot = heatmap_df.pivot_table(
            values='profit',
            index='category',
            columns='trader_name',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            pivot,
            title='Average Profit: Category × Trader',
            template='plotly_dark',
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Treemap of profitable items
        st.markdown("#### Profit Treemap (Profitable Items)")
        profitable = df[df['profit'] > 0].copy()
        
        if len(profitable) > 0:
            # Limit to top items for readability
            profitable = profitable.nlargest(100, 'profit')
            
            fig = px.treemap(
                profitable,
                path=['category', 'trader_name', 'name'],
                values='profit',
                title='Profit Breakdown (Top 100 Items)',
                template='plotly_dark',
                color='profit',
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Trends & Patterns Tab ---
    with viz_tabs[4]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Trader value scatter
            fig = px.scatter(
                df,
                x='flea_price',
                y='trader_price',
                color='profit',
                hover_name='name',
                title='Flea Price vs Trader Price',
                template='plotly_dark',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            # Add break-even line
            max_price = max(df['flea_price'].max(), df['trader_price'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_price],
                y=[0, max_price],
                mode='lines',
                name='Break Even',
                line=dict(dash='dash', color='gray')
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weight efficiency (profit per kg)
            weight_df = df[df['weight'] > 0].copy()
            if len(weight_df) > 0:
                weight_df['profit_per_kg'] = weight_df['profit'] / weight_df['weight']
                weight_df = weight_df.nlargest(20, 'profit_per_kg')
                
                fig = px.bar(
                    weight_df,
                    x='profit_per_kg',
                    y='name',
                    orientation='h',
                    title='Top 20: Profit per KG (Weight Efficiency)',
                    template='plotly_dark',
                    color='profit_per_kg',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=COLORS['text'],
                    coloraxis_showscale=False,
                    yaxis={'categoryorder': 'total ascending'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weight data available.")
        
        # 24h price change analysis
        if 'change_last_48h' in df.columns:
            st.markdown("#### 48h Price Movement")
            
            change_df = df[df['change_last_48h'].notna() & (df['change_last_48h'] != 0)].copy()
            
            if len(change_df) > 0:
                col3, col4 = st.columns(2)
                
                with col3:
                    # Biggest gainers
                    gainers = change_df.nlargest(10, 'change_last_48h')
                    fig = px.bar(
                        gainers,
                        x='change_last_48h',
                        y='name',
                        orientation='h',
                        title='📈 Biggest Price Gainers (48h)',
                        template='plotly_dark',
                        color_discrete_sequence=[COLORS['profit']]
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color=COLORS['text'],
                        yaxis={'categoryorder': 'total ascending'},
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    # Biggest losers
                    losers = change_df.nsmallest(10, 'change_last_48h')
                    fig = px.bar(
                        losers,
                        x='change_last_48h',
                        y='name',
                        orientation='h',
                        title='📉 Biggest Price Drops (48h)',
                        template='plotly_dark',
                        color_discrete_sequence=[COLORS['loss']]
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color=COLORS['text'],
                        yaxis={'categoryorder': 'total descending'},
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price change data available.")

# =============================================================================
# ANALYTICS CHARTS
# =============================================================================
def render_analytics(df: pd.DataFrame) -> None:
    """Render analytics visualizations with ML insights and trend learning.
    
    Args:
        df: DataFrame containing filtered market data.
    """
    if df.empty or len(df) < 3:
        st.info("Not enough data for analytics.")
        return
    
    st.markdown("### 📈 ML-Powered Market Analytics")
    
    ml_engine = get_ml_engine()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Profit Analysis", 
        "🧠 Learning Status",
        "📈 Trend Insights",
        "🎯 Category Performance"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit distribution
            fig = px.histogram(
                df, x='profit', nbins=50,
                title='Profit Distribution',
                color_discrete_sequence=[COLORS['profit']],
                template='plotly_dark'
            )
            fig.add_vline(x=0, line_dash="dash", line_color=COLORS['loss'], 
                         annotation_text="Break Even")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                xaxis_title="Profit (₽)",
                yaxis_title="Count",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI vs Profit scatter
            fig = px.scatter(
                df[df['profit'] > 0].head(100),
                x='profit', y='roi',
                size='last_offer_count',
                color='profit',
                hover_data=['name', 'trader_name'],
                title='ROI vs Profit (Top 100)',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ML Analysis if enough items
        if len(df) >= 10:
            ml_df = ml_engine.calculate_opportunity_score_ml(df.copy())
            ml_df = ml_engine.calculate_risk_score(ml_df)
            ml_df = ml_engine.cluster_items(ml_df)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Risk vs Opportunity Matrix
                fig = px.scatter(
                    ml_df, x='risk_score', y='ml_opportunity_score',
                    color='cluster_label', size='profit',
                    hover_data=['name', 'profit'],
                    title='Risk vs Opportunity Matrix',
                    template='plotly_dark'
                )
                fig.add_hline(y=50, line_dash="dash", line_color="gray")
                fig.add_vline(x=50, line_dash="dash", line_color="gray")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=COLORS['text'],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Cluster distribution
                cluster_stats = ml_df.groupby('cluster_label').agg({
                    'profit': ['mean', 'count'],
                    'risk_score': 'mean'
                }).round(0)
                cluster_stats.columns = ['Avg Profit', 'Count', 'Avg Risk']
                cluster_stats = cluster_stats.sort_values('Avg Profit', ascending=False)
                
                st.markdown("#### Strategy Tiers")
                st.dataframe(
                    cluster_stats,
                    use_container_width=True,
                    column_config={
                        "Avg Profit": st.column_config.NumberColumn(format="₽%.0f"),
                        "Avg Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                    }
                )
    
    with tab2:
        # Learning Status Dashboard
        st.markdown("#### 🧠 Persistent Model Learning Status")
        st.caption("The model continuously learns and improves, surviving database cleanups.")
        
        learning_status = ml_engine.get_persistent_learning_status()
        
        # Learning quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            quality = learning_status.get('overall_quality', 0)
            st.metric(
                "Learning Quality",
                f"{quality:.0f}%",
                help="Overall model learning quality based on data volume"
            )
        
        with col2:
            st.metric(
                "Items Learned",
                f"{learning_status.get('unique_items_learned', 0):,}",
                help="Number of unique items the model has learned about"
            )
        
        with col3:
            st.metric(
                "Training Sessions",
                f"{learning_status.get('total_sessions', 0):,}",
                help="Number of data collection cycles used for training"
            )
        
        with col4:
            st.metric(
                "Total Samples",
                f"{learning_status.get('total_samples', 0):,}",
                help="Total number of data points processed"
            )
        
        # Quality breakdown
        st.markdown("#### Learning Quality Breakdown")
        
        quality_data = pd.DataFrame({
            'Metric': ['Items Coverage', 'Session History', 'Sample Volume'],
            'Quality': [
                learning_status.get('items_quality', 0),
                learning_status.get('sessions_quality', 0),
                learning_status.get('samples_quality', 0)
            ]
        })
        
        fig = px.bar(
            quality_data,
            x='Metric', y='Quality',
            title='Learning Quality by Component',
            template='plotly_dark',
            color='Quality',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            height=300,
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Last updated
        if learning_status.get('last_updated'):
            st.caption(f"Last model update: {learning_status['last_updated']}")
    
    with tab3:
        # Trend Insights
        st.markdown("#### 📈 Learned Item Trends")
        st.caption("Items ranked by learned profitability and consistency over time.")
        
        top_learned = ml_engine.get_top_learned_items(20)
        
        if top_learned:
            trend_df = pd.DataFrame(top_learned)
            
            # Display top learned items - consistency score is 0-100
            fig = px.bar(
                trend_df.head(15),
                x='profit_mean',
                y='item_id',
                orientation='h',
                color='consistency_score',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],  # Consistency is 0-100 scale
                title='Top 15 Items by Learned Profitability',
                template='plotly_dark',
                hover_data=['category', 'data_points']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=450,
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_colorbar_title='Consistency %'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("#### Detailed Trend Data")
            st.dataframe(
                trend_df,
                column_config={
                    "item_id": st.column_config.TextColumn("Item ID", width=200),
                    "profit_mean": st.column_config.NumberColumn("Avg Profit", format="₽%.0f"),
                    "consistency_score": st.column_config.ProgressColumn("Consistency", min_value=0, max_value=100),
                    "data_points": st.column_config.NumberColumn("Data Points"),
                    "category": st.column_config.TextColumn("Category"),
                    "trader": st.column_config.TextColumn("Best Trader"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No trend data available yet. Keep the collector running to build trend history.")
    
    with tab4:
        # Category Performance
        st.markdown("#### 🎯 Learned Category Performance")
        st.caption("Category rankings based on historical profitability patterns.")
        
        category_trends = ml_engine.get_category_performance()
        
        if category_trends:
            cat_df = pd.DataFrame(category_trends)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category profit chart - color by profitability rate (0-100%)
                cat_df_profit = cat_df.head(12).copy()
                
                fig = px.bar(
                    cat_df_profit,
                    x='avg_profit',
                    y='category',
                    orientation='h',
                    color='profitable_rate',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 1],  # Ensure full 0-100% range is used
                    title='Top Categories by Avg Profit',
                    template='plotly_dark'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=COLORS['text'],
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_colorbar_title='Profit %',
                    coloraxis_colorbar_tickformat='.0%'  # Show as percentage
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category weight chart - weight is a multiplier (1.0 = neutral)
                # Color by how far from neutral (1.0) the weight is
                cat_df_chart = cat_df.head(12).copy()
                cat_df_chart['weight_deviation'] = cat_df_chart['weight'] - 1.0  # Positive = boosted, negative = penalized
                
                fig = px.bar(
                    cat_df_chart,
                    x='weight',
                    y='category',
                    orientation='h',
                    color='weight_deviation',
                    color_continuous_scale='RdYlGn',  # Red for low weight, Green for high weight
                    color_continuous_midpoint=0,  # Center on neutral (1.0 weight = 0 deviation)
                    title='Category Learned Weights (1.0 = Neutral)',
                    template='plotly_dark'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=COLORS['text'],
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_colorbar_title='Boost'  # Shows deviation from neutral weight
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Full table - convert profitable_rate to percentage for display
            cat_df_display = cat_df.copy()
            cat_df_display['profitable_pct'] = cat_df_display['profitable_rate'] * 100
            
            st.dataframe(
                cat_df_display,
                column_config={
                    "category": st.column_config.TextColumn("Category"),
                    "avg_profit": st.column_config.NumberColumn("Avg Profit", format="₽%.0f"),
                    "profitable_pct": st.column_config.ProgressColumn("Profit Rate", min_value=0, max_value=100, format="%.0f%%"),
                    "weight": st.column_config.NumberColumn("Weight", format="%.2f"),
                    "total_items": st.column_config.NumberColumn("Items Tracked"),
                    "profitable_rate": None,  # Hide the raw 0-1 column
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No category data available yet. Keep the collector running to build category insights.")

# =============================================================================
# ITEM DETAIL VIEW
# =============================================================================
def render_item_detail(df: pd.DataFrame) -> None:
    """Render detailed item analysis with price history.
    
    Args:
        df: DataFrame containing filtered market data.
    """
    if df.empty:
        return
    
    st.markdown("### 🔍 Item Deep Dive")
    
    item_name = st.selectbox(
        "Select an item to analyze",
        options=df.sort_values('profit', ascending=False)['name'].tolist(),
        index=0
    )
    
    item = df[df['name'] == item_name].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        icon_link = item.get('icon_link', '')
        if icon_link and isinstance(icon_link, str) and icon_link.startswith('http'):
            st.image(icon_link, width=128)
        st.markdown(f"**{item['name']}**")
        st.caption(f"{item['category']}")
        
        wiki_link = item.get('wiki_link')
        if wiki_link and isinstance(wiki_link, str) and wiki_link.startswith('http'):
            st.link_button("📖 Wiki", wiki_link)
    
    with col2:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Profit", f"₽{item['profit']:,.0f}")
        with metrics_col2:
            st.metric("ROI", f"{item['roi']:.1f}%")
        with metrics_col3:
            st.metric("Flea Price", f"₽{item['flea_price']:,.0f}")
        with metrics_col4:
            st.metric("Trader Price", f"₽{item['trader_price']:,.0f}")
        
        # Price history chart
        history = database.get_item_history(item['item_id'])
        if history and len(history) > 1:
            hist_df = pd.DataFrame(history, columns=['timestamp', 'flea_price', 'trader_price', 'profit'])
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df['timestamp'], y=hist_df['profit'],
                mode='lines+markers', name='Profit',
                line=dict(color=COLORS['profit'], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=hist_df['timestamp'], y=hist_df['flea_price'],
                mode='lines', name='Flea Price',
                line=dict(color=COLORS['warning'], dash='dash')
            ))
            fig.update_layout(
                title='Price History',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                height=250,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SYSTEM CONTROL PANEL
# =============================================================================
def render_system_panel() -> None:
    """Render the system control and monitoring panel.
    
    Provides collector management, database statistics, log viewing,
    and system maintenance controls.
    """
    st.markdown("### ⚙️ System Control Panel")
    
    # Create tabs for different system functions
    sys_tab1, sys_tab2, sys_tab3, sys_tab4 = st.tabs([
        "🔄 Collector Status", 
        "🗄️ Database", 
        "📜 Logs",
        "🔧 Maintenance"
    ])
    
    # --- Collector Status Tab ---
    with sys_tab1:
        st.markdown("#### Data Collector Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            running, pid, mode = is_collector_running()
            
            if running:
                st.success(f"✅ Collector is **RUNNING**")
                mode_display = mode.title() if mode else "Unknown"
                st.markdown(f"""
                <div class="profit-card">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div>
                            <div class="stat-label">Process ID</div>
                            <div class="stat-value">{pid}</div>
                        </div>
                        <div>
                            <div class="stat-label">Mode</div>
                            <div class="stat-value">{mode_display}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Collector is **STOPPED**")
                st.caption("Start the collector to begin fetching market data.")
        
        with col2:
            st.markdown("**Quick Actions**")
            
            if running:
                if mode != "standalone":
                    if st.button("⏹️ Stop Collector", use_container_width=True, type="secondary"):
                        if stop_collector():
                            st.success("Collector stopped!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to stop collector")
                else:
                    st.warning("Standalone mode - use Force Kill")
            else:
                if st.button("▶️ Start Collector", use_container_width=True, type="primary"):
                    if start_collector():
                        st.success("Collector started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to start collector")
        
        st.markdown("---")
        
        # Show all running collector processes
        st.markdown("#### Running Collector Processes")
        collector_procs = find_all_collector_processes()
        
        # Separate project vs external processes
        project_procs = [p for p in collector_procs if p.get('is_project', False)]
        external_procs = [p for p in collector_procs if not p.get('is_project', False)]
        
        if project_procs:
            st.markdown("**🟢 This Project's Collectors:**")
            for proc in project_procs:
                with st.container():
                    pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
                    with pcol1:
                        st.code(f"PID: {proc['pid']}")
                    with pcol2:
                        st.caption(f"Started: {proc['create_time']}")
                        st.text(proc['cmdline'][:60] + "..." if len(proc['cmdline']) > 60 else proc['cmdline'])
                    with pcol3:
                        if st.button("Kill", key=f"kill_{proc['pid']}", type="secondary"):
                            try:
                                psutil.Process(proc['pid']).kill()
                                st.success(f"Killed PID {proc['pid']}")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
        
        if external_procs:
            st.markdown("**🟡 Other collector.py Processes (different projects):**")
            st.caption("These are collector.py scripts from other directories - not from this project.")
            for proc in external_procs:
                with st.container():
                    pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
                    with pcol1:
                        st.code(f"PID: {proc['pid']}")
                    with pcol2:
                        st.caption(f"Started: {proc['create_time']}")
                        st.text(proc['cmdline'][:60] + "..." if len(proc['cmdline']) > 60 else proc['cmdline'])
                    with pcol3:
                        if st.button("Kill", key=f"kill_ext_{proc['pid']}", type="secondary"):
                            try:
                                psutil.Process(proc['pid']).kill()
                                st.success(f"Killed PID {proc['pid']}")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
        
        if not project_procs and not external_procs:
            st.info("No collector processes found running.")
        
        st.markdown("---")
        
        # Force kill all
        st.markdown("#### ⚠️ Emergency Controls")
        st.caption("Use these if the collector is stuck or unresponsive.")
        
        if st.button("🔴 Force Kill ALL Collectors", type="secondary", use_container_width=True):
            killed, failed = force_kill_all_collectors()
            if killed > 0:
                st.success(f"Killed {killed} process(es)")
            if failed > 0:
                st.warning(f"Failed to kill {failed} process(es)")
            if killed == 0 and failed == 0:
                st.info("No collector processes found to kill")
            time.sleep(1)
            st.rerun()
    
    # --- Database Tab ---
    with sys_tab2:
        st.markdown("#### Database Statistics")
        
        db_stats = get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Database Size", db_stats['size_str'])
        with col2:
            st.metric("Total Records", f"{db_stats['record_count']:,}")
        with col3:
            st.metric("Unique Items", f"{db_stats['unique_items']:,}")
        with col4:
            if db_stats['newest_record']:
                try:
                    newest = datetime.fromisoformat(db_stats['newest_record'])
                    age = datetime.now() - newest
                    age_str = f"{int(age.total_seconds() / 60)}m ago"
                except Exception:
                    age_str = "Unknown"
            else:
                age_str = "No data"
            st.metric("Last Update", age_str)
        
        st.markdown("---")
        
        # Data range info
        if db_stats['oldest_record'] and db_stats['newest_record']:
            st.markdown("#### Data Time Range")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 **Oldest:** {db_stats['oldest_record']}")
            with col2:
                st.info(f"📅 **Newest:** {db_stats['newest_record']}")
        
        st.markdown("---")
        
        # Database actions
        st.markdown("#### Database Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Clear All Data**")
            st.caption("Delete all price history. Cannot be undone!")
            if st.button("🗑️ Clear Database", type="secondary", use_container_width=True, key="clear_db_btn"):
                st.session_state['confirm_clear'] = True
            
            if st.session_state.get('confirm_clear'):
                st.warning("⚠️ Are you sure? This will delete ALL data!")
                ccol1, ccol2 = st.columns(2)
                with ccol1:
                    if st.button("✅ Yes, Clear", type="primary"):
                        database.clear_all_data()
                        st.cache_data.clear()
                        st.session_state['confirm_clear'] = False
                        st.success("Database cleared!")
                        time.sleep(1)
                        st.rerun()
                with ccol2:
                    if st.button("❌ Cancel"):
                        st.session_state['confirm_clear'] = False
                        st.rerun()
        
        with col2:
            st.markdown("**Optimize Database**")
            st.caption("Run VACUUM to reclaim space and optimize.")
            if st.button("🔧 Optimize", use_container_width=True):
                try:
                    import sqlite3
                    conn = sqlite3.connect(database.DB_NAME, timeout=30)
                    conn.execute("VACUUM")
                    conn.close()
                    st.success("Database optimized!")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
        
        with col3:
            st.markdown("**Cleanup Old Data**")
            st.caption("Remove records older than 7 days.")
            if st.button("🧹 Cleanup", use_container_width=True):
                try:
                    deleted = database.cleanup_old_data(days=7, vacuum=True)
                    st.success(f"Cleaned up {deleted:,} old records!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
    
    # --- Logs Tab ---
    with sys_tab3:
        st.markdown("#### Log Viewer")
        
        log_files = get_log_files()
        
        if not log_files:
            st.info("No log files found.")
        else:
            # Log file selector
            col1, col2 = st.columns([3, 1])
            
            with col1:
                log_options = [f['name'] for f in log_files]
                selected_log = st.selectbox(
                    "Select Log File",
                    options=log_options,
                    index=0
                )
            
            with col2:
                max_lines = st.number_input("Lines", min_value=10, max_value=500, value=50, step=10)
            
            # Find selected log file
            selected_file = next((f for f in log_files if f['name'] == selected_log), None)
            
            if selected_file:
                st.caption(f"Modified: {selected_file['modified'].strftime('%Y-%m-%d %H:%M:%S')} | Size: {selected_file['size'] / 1024:.1f} KB")
                
                # Refresh button
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔄 Refresh Logs"):
                        st.rerun()
                
                # Log content
                log_content = read_log_file(selected_file['path'], max_lines)
                
                st.code(log_content, language="log", line_numbers=True)
                
                # Clear log option
                st.markdown("---")
                if st.button("🗑️ Clear This Log File", type="secondary"):
                    try:
                        with open(selected_file['path'], 'w') as f:
                            f.write(f"Log cleared at {datetime.now().isoformat()}\n")
                        st.success("Log file cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear log: {e}")
    
    # --- Maintenance Tab ---
    with sys_tab4:
        st.markdown("#### System Maintenance")
        
        # System info
        st.markdown("**System Information**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            st.metric("CPU Usage", f"{cpu_percent}%")
        
        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")
        
        with col3:
            disk = psutil.disk_usage('.')
            st.metric("Disk Usage", f"{disk.percent}%")
        
        st.markdown("---")
        
        # Cache management
        st.markdown("**Cache Management**")
        
        if st.button("🔄 Clear All Caches", use_container_width=True):
            st.cache_data.clear()
            st.success("All caches cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # PID file cleanup
        st.markdown("**PID File Cleanup**")
        st.caption("Clean up stale PID files if processes crashed.")
        
        pid_files = [f for f in [PID_FILE, STANDALONE_PID_FILE] if os.path.exists(f)]
        
        if pid_files:
            for pf in pid_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    try:
                        with open(pf, 'r') as f:
                            pid_content = f.read().strip()
                        st.text(f"📄 {pf} (PID: {pid_content})")
                    except Exception:
                        st.text(f"📄 {pf}")
                with col2:
                    if st.button("Delete", key=f"del_{pf}"):
                        try:
                            os.remove(pf)
                            st.success(f"Deleted {pf}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
        else:
            st.info("No PID files found.")
        
        st.markdown("---")
        
        # Quick restart
        st.markdown("**Application Restart**")
        st.caption("Force a page refresh to reload all components.")
        
        if st.button("🔄 Refresh Application", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

# =============================================================================
# SETTINGS SIDEBAR
# =============================================================================
def render_sidebar() -> dict:
    """Render the settings sidebar with quick filters.
    
    Returns:
        dict: Filter settings including min_profit, min_roi, category, search, show_negative.
    """
    with st.sidebar:
        st.markdown("## 🎯 Quick Filters")
        
        # Quick collector status indicator
        running, pid, mode = is_collector_running()
        if running:
            st.success(f"✅ Collector Running (PID: {pid})")
        else:
            st.error("❌ Collector Stopped")
            st.caption("Go to System tab to start")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 📊 Data Filters")
        
        min_profit = st.number_input("Min Profit (₽)", value=0, step=1000)
        min_roi = st.number_input("Min ROI (%)", value=0.0, step=1.0)
        
        df = load_data()
        categories = ['All'] + sorted(df['category'].dropna().unique().tolist()) if not df.empty else ['All']
        category = st.selectbox("Category", categories)
        
        search = st.text_input("🔍 Search", placeholder="Item name...")
        
        show_negative = st.checkbox("Show Negative Profit", value=False)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if running:
                if st.button("⏹️ Stop", use_container_width=True, type="secondary"):
                    stop_collector()
                    st.rerun()
            else:
                if st.button("▶️ Start", use_container_width=True, type="primary"):
                    if start_collector():
                        st.rerun()
        
        return {
            'min_profit': min_profit,
            'min_roi': min_roi,
            'category': category,
            'search': search,
            'show_negative': show_negative
        }

# =============================================================================
# MAIN APP
# =============================================================================
def main() -> None:
    """Main application entry point - orchestrates all dashboard components."""
    # Sidebar filters
    filters = render_sidebar()
    
    # Load data
    df = load_data()
    filtered_df = get_filtered_data(df, filters)
    
    # Main content
    render_header()
    
    st.markdown("---")
    
    render_stats(filtered_df)
    
    st.markdown("---")
    
    render_top_opportunities(filtered_df)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Top Trades", 
        "🌐 Market Explorer",
        "🎨 Visual Analytics",
        "📈 ML Insights", 
        "🔍 Item Details", 
        "⚙️ System"
    ])
    
    with tab1:
        render_data_table(filtered_df)
    
    with tab2:
        render_market_explorer(df)  # Use full df, not filtered
    
    with tab3:
        render_visual_analytics(df)  # Use full df for comprehensive analysis
    
    with tab4:
        render_analytics(filtered_df)
    
    with tab5:
        render_item_detail(filtered_df)
    
    with tab6:
        render_system_panel()
    
    # Footer
    st.markdown("---")
    st.caption("💡 Data from tarkov.dev API • Updated every 5 minutes • Not affiliated with Battlestate Games")

if __name__ == "__main__":
    main()
