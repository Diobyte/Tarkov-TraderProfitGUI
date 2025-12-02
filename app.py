import streamlit as st
import pandas as pd
import database
import utils
import time
import subprocess
import os
import signal
import sys
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import psutil
from typing import Tuple, Optional, Any

# --- Configuration ---
# Flea Market Level Requirements (Based on Patch 0.15+ changes)

CATEGORY_LOCKS = {
    "Sniper rifle": 20,
    "Assault rifle": 25,
    "Assault carbine": 25,
    "Marksman rifle": 25,
    "Backpack": 25,
    "Foregrip": 20,
    "Comb. tact. device": 25,
    "Flashlight": 25,
    "Auxiliary Mod": 25,
    "Comb. muzzle device": 20,
    "Flashhider": 20,
    "Silencer": 20,
    "Building material": 30,
    "Electronics": 30,
    "Household goods": 30,
    "Jewelry": 30,
    "Tool": 30,
    "Battery": 30,
    "Lubricant": 30,
    "Medical supplies": 30,
    "Fuel": 30,
    "Drug": 30, 
    "Info": 30, 
}

ITEM_LOCKS = {
    "PS12B": 40,
    "M80": 35,
    "Blackout CJB": 40,
}

# Configure Logging for the Streamlit App
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8',
    force=True
)

# Initialize DB (ensure tables exist and WAL mode is on)
database.init_db()

PID_FILE = "collector.pid"
STANDALONE_PID_FILE = "collector_standalone.pid"

def is_collector_running() -> Tuple[bool, Optional[int], Optional[str]]:
    # Check standalone first
    if os.path.exists(STANDALONE_PID_FILE):
        try:
            with open(STANDALONE_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                # Verify it's actually python
                if "python" in p.name().lower() or "collector" in p.name().lower():
                    return True, pid, "standalone"
        except (OSError, ValueError, psutil.NoSuchProcess):
            pass # Fall through to check normal PID

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                if "python" in p.name().lower() or "collector" in p.name().lower():
                    return True, pid, "session"
        except (OSError, ValueError, psutil.NoSuchProcess):
            # Process dead or file corrupt
            return False, None, None
    return False, None, None

def start_collector() -> None:
    running, _, _ = is_collector_running()
    if running:
        return
    # Start collector.py in a separate process
    # Redirect stdout and stderr to a log file
    # Use -u for unbuffered output so logs appear immediately
    # Note: collector.py now handles its own logging to collector.log, 
    # but we still redirect stdout/stderr to catch crashes that happen before logging is setup
    
    # Windows flags to detach process and prevent signal propagation (Ctrl+C)
    # We use CREATE_NO_WINDOW so the collector runs in the background without a popup
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    # Redirect stdout/stderr to a file to capture startup errors
    # This is crucial for debugging if the collector fails to start (e.g. missing imports)
    try:
        log_file = open("collector_startup.log", "a")
        proc = subprocess.Popen(
            [sys.executable, "-u", "collector.py"], 
            creationflags=creation_flags,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait a moment to see if it crashes immediately
        time.sleep(2)
        if proc.poll() is not None:
            # Process exited immediately
            st.error(f"Collector failed to start. Check collector_startup.log for details. Return code: {proc.returncode}")
            logging.error(f"Collector failed to start. Return code: {proc.returncode}")
            return
        
        with open(PID_FILE, 'w') as f:
            f.write(str(proc.pid))
        logging.info(f"Started collector with PID {proc.pid}")
    except Exception as e:
        logging.error(f"Failed to start collector process: {e}")
        st.error(f"Failed to start collector process: {e}")

def stop_collector() -> None:
    running, pid, mode = is_collector_running()
    if running and pid:
        if mode == "standalone":
            st.warning("Collector is running in Standalone Mode. Please stop it from the terminal running 'run_collector.bat'.")
            return

        try:
            os.kill(pid, signal.SIGTERM) # Try graceful termination
            logging.info(f"Stopped collector with PID {pid}")
        except OSError as e:
            logging.error(f"Error stopping collector: {e}")
            pass
        # Clean up PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

def force_kill_all_collectors() -> None:
    # 1. Try standard stop first
    stop_collector()
    
    # 2. Force kill any lingering processes by name/command line
    try:
        if sys.platform == "win32":
            # Use PowerShell to find and kill python processes running collector.py
            # We use Get-CimInstance to reliably get the command line arguments
            cmd = "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*collector.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
            subprocess.run(["powershell", "-Command", cmd], creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.run(["pkill", "-f", "collector.py"])
        
        logging.info("Force killed all collector processes.")
    except Exception as e:
        logging.error(f"Error force killing collectors: {e}")

    # 3. Final cleanup of PID file
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except:
            pass

st.set_page_config(page_title="Tarkov Trader Profit", layout="wide")

st.title("Tarkov Trader Profit Dashboard (RUB)")

# --- Data Loading ---
@st.cache_data(ttl=10) # Cache data for 10 seconds to prevent DB spam
def load_data(trend_hours: int = 168) -> pd.DataFrame:
    try:
        # 1. Get Latest Snapshot
        data = database.get_latest_prices()
        if not data:
            logging.warning("No data returned from database.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=['item_id', 'name', 'flea_price', 'trader_price', 'trader_name', 'profit', 'timestamp', 'icon_link', 'width', 'height', 'avg_24h_price', 'low_24h_price', 'change_last_48h', 'weight', 'category'])
        
        # 2. Get Historical Trends
        trend_df = pd.DataFrame()
        try:
            trends = database.get_market_trends(hours=trend_hours)
            if trends:
                trend_df = pd.DataFrame(trends, columns=['item_id', 'trend_avg_profit', 'trend_min_profit', 'trend_max_profit', 'data_points'])
                
                # Merge trends into main dataframe
                df = pd.merge(df, trend_df, on='item_id', how='left')
                
                # Calculate Volatility (Max - Min) as a simple proxy for risk
                # Fill NaNs before calculation to avoid issues
                df['trend_max_profit'] = df['trend_max_profit'].fillna(df['profit'])
                df['trend_min_profit'] = df['trend_min_profit'].fillna(df['profit'])
                
                df['volatility'] = df['trend_max_profit'] - df['trend_min_profit']
                df['volatility'] = df['volatility'].fillna(0)
            else:
                df['volatility'] = 0
                df['trend_avg_profit'] = df['profit']
        except Exception as e:
            logging.warning(f"Could not load market trends: {e}")
            # Don't show warning to user, just log it and proceed with partial data
            df['volatility'] = 0
            df['trend_avg_profit'] = df['profit']

        return df
    except Exception as e:
        logging.error(f"Error loading data from database: {e}")
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

# --- Sidebar: Collector Control ---
st.sidebar.header("Data Collector")
try:
    collector_running, pid, collector_mode = is_collector_running()
except Exception:
    collector_running, pid, collector_mode = False, None, None

if collector_running:
    if collector_mode == "standalone":
        st.sidebar.success(f"Collector Running (Standalone Mode, PID: {pid})")
        st.sidebar.info("Managed by external script.")
    else:
        st.sidebar.success(f"Collector Running (PID: {pid})")
        if st.sidebar.button("Stop Collector"):
            stop_collector()
            st.rerun()
else:
    st.sidebar.warning("Collector Stopped")
    if st.sidebar.button("Start Collector (Every 5m)"):
        try:
            start_collector()
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start collector: {e}")
            logging.error(f"Failed to start collector: {e}")

with st.sidebar.expander("Maintenance"):
    if st.button("Force Kill All Collectors"):
        force_kill_all_collectors()
        st.success("Attempted to kill all collector processes.")
        time.sleep(1)
        st.rerun()
        
st.sidebar.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
player_level = st.sidebar.slider("Your Player Level", min_value=1, max_value=70, value=15, help="Filters items based on Flea Market level requirements.")
show_locked = st.sidebar.checkbox("Show Locked Items", value=False, help="Show items even if you don't meet the level requirement.")
min_profit = st.sidebar.number_input("Min Profit (RUB)", value=0, step=1000, min_value=-1000000)
min_roi = st.sidebar.number_input("Min ROI (%)", value=0.0, step=1.0)
min_pps = st.sidebar.number_input("Min Profit Per Slot (RUB)", value=0, step=1000)
min_discount = st.sidebar.number_input("Min Discount from Avg (%)", value=0.0, step=5.0)
trend_window_hours = st.sidebar.number_input("Trend Analysis Window (Hours)", value=168, step=24, min_value=24, max_value=720)
search_term = st.sidebar.text_input("Search Item Name")

# --- Auto-Refresh Configuration ---
# Dashboard automatically refreshes every 60 seconds to check for new data
refresh_interval = 60

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.toast("Refreshing data...", icon="🔄")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.caption("Tarkov Trader Profit v1.1")
st.sidebar.caption("Data provided by tarkov.dev")

@st.fragment(run_every=30)
def render_sidebar_status():
    # Show last DB update in sidebar
    try:
        last_db_update = database.get_latest_timestamp()
        if last_db_update:
            st.sidebar.info(f"DB Last Updated:\n{last_db_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.sidebar.info("DB Status: Empty")
    except Exception:
        pass

render_sidebar_status()

def get_filtered_data():
    try:
        df = load_data(trend_hours=trend_window_hours)
    except Exception as e:
        logging.critical(f"Critical error loading application data: {e}")
        st.error(f"Critical error loading application data: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        # Calculate Metrics using shared utility
        df = utils.calculate_metrics(df)
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return pd.DataFrame()

    # Apply Filters
    filtered_df = df[
        (df['profit'] >= min_profit) & 
        (df['roi'] >= min_roi) &
        (df['profit_per_slot'] >= min_pps)
    ]
    
    if min_discount > 0:
        filtered_df = filtered_df[filtered_df['discount_percent'] >= min_discount]

    if search_term:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]
    
    # Apply Level Filters
    if not show_locked:
        if player_level < 15:
            # Flea market is locked below level 15
            st.warning("Flea Market is locked below level 15. No items available.")
            return pd.DataFrame(columns=filtered_df.columns)
            
        def is_item_unlocked(row):
            name = row['name']
            category = row['category']
            
            # Check specific item overrides first
            for restricted_item, level_req in ITEM_LOCKS.items():
                if restricted_item in name:
                    if player_level < level_req:
                        return False
            
            # Check Category
            if category in CATEGORY_LOCKS:
                if player_level < CATEGORY_LOCKS[category]:
                    return False
                    
            return True

        filtered_df = filtered_df[filtered_df.apply(is_item_unlocked, axis=1)]
        
    return filtered_df

@st.fragment(run_every=refresh_interval)
def render_header_metrics():
    filtered_df = get_filtered_data()
    if filtered_df.empty:
        # Check if collector is running
        is_running, _, _ = is_collector_running()
        if is_running:
            st.info("Collector is running... Waiting for initial data fetch (this may take 10-20 seconds).")
            time.sleep(5) # Poll every 5 seconds
            st.rerun()
        else:
            st.warning("No data found. Please run the collector script first.")
            if st.button("Refresh Data Now"):
                st.cache_data.clear()
                st.rerun()
        return

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Opportunities", len(filtered_df))
    
    if len(filtered_df) < 5 and not show_locked and min_profit >= 0:
        st.info("Tip: Few items found? Try enabling 'Show Locked Items' or lowering 'Min Profit' to see more opportunities.")

    if not filtered_df.empty:
        col2.metric("Max Profit", f"{filtered_df['profit'].max():,.0f} ₽")
        col3.metric("Avg ROI", f"{filtered_df['roi'].mean():.1f}%")
        col4.metric("Best Item", filtered_df.sort_values('profit', ascending=False).iloc[0]['name'])

    # --- Recommendation Engine ---
    st.markdown("### 🏆 Daily Top 10 Recommendations")
    st.markdown("Based on a weighted score of Profit (35%), Efficiency (25%), ROI (20%), Stability (10%), and Discounts (10%).")
    
    if not filtered_df.empty:
        try:
            rec_df = filtered_df.copy()
            scaler = MinMaxScaler()
            
            if rec_df['volatility'].max() > 0:
                rec_df['stability_norm'] = 1 - scaler.fit_transform(rec_df[['volatility']])
            else:
                rec_df['stability_norm'] = 1.0
                
            metrics = ['profit', 'roi', 'profit_per_slot', 'discount_percent']
            rec_df[metrics] = rec_df[metrics].fillna(0)
            
            if len(rec_df) > 1:
                norm_data = scaler.fit_transform(rec_df[metrics])
                norm_df = pd.DataFrame(norm_data, columns=[f'{c}_norm' for c in metrics], index=rec_df.index)
            else:
                # If only one item, give it max score
                norm_df = pd.DataFrame(1.0, index=rec_df.index, columns=[f'{c}_norm' for c in metrics])
            
            rec_df = pd.concat([rec_df, norm_df], axis=1)
            
            rec_df['trader_score'] = (
                (rec_df['profit_norm'] * 0.35) +
                (rec_df['profit_per_slot_norm'] * 0.25) +
                (rec_df['roi_norm'] * 0.20) +
                (rec_df['stability_norm'] * 0.10) +
                (rec_df['discount_percent_norm'] * 0.10)
            ) * 100
            
            top_10 = rec_df.sort_values('trader_score', ascending=False).head(10)
            
            st.dataframe(
                top_10,
                column_order=['name', 'trader_score', 'profit', 'profit_per_slot', 'roi', 'volatility', 'category'],
                column_config={
                    "name": "Item Name",
                    "trader_score": st.column_config.ProgressColumn("Trader Score", format="%.1f", min_value=0, max_value=100),
                    "profit": st.column_config.NumberColumn("Profit", format="%d ₽"),
                    "profit_per_slot": st.column_config.NumberColumn("Profit/Slot", format="%d ₽"),
                    "roi": st.column_config.NumberColumn("ROI", format="%.1f %%"),
                    "volatility": st.column_config.NumberColumn("Volatility (Risk)", format="%d ₽"),
                    "category": "Category"
                },
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    else:
        st.info("No items match your filters to generate recommendations.")
    
    st.write(f"Last updated: {filtered_df['timestamp'].iloc[0] if not filtered_df.empty else 'Never'}")

@st.fragment(run_every=refresh_interval)
def render_market_table():
    filtered_df = get_filtered_data()
    if filtered_df.empty: return

    st.subheader("Current Market Opportunities")
    st.dataframe(
        filtered_df.sort_values(by='profit', ascending=False),
        column_config={
            "icon_link": st.column_config.ImageColumn("Icon"),
            "flea_price": st.column_config.NumberColumn("Flea Price", format="%d ₽"),
            "trader_price": st.column_config.NumberColumn("Trader Price", format="%d ₽"),
            "profit": st.column_config.NumberColumn("Profit", format="%d ₽"),
            "profit_per_slot": st.column_config.NumberColumn("Profit/Slot", format="%d ₽"),
            "roi": st.column_config.ProgressColumn("ROI", format="%.2f %%", min_value=0, max_value=100),
            "discount_percent": st.column_config.NumberColumn("Discount (vs Avg)", format="%.1f %%"),
            "avg_24h_price": st.column_config.NumberColumn("24h Avg", format="%d ₽"),
            "timestamp": st.column_config.DatetimeColumn("Last Updated", format="D MMM, HH:mm:ss"),
            "category": st.column_config.TextColumn("Category"),
            "width": None,
            "height": None,
            "slots": None,
            "weight": None,
            "change_last_48h": None,
            "low_24h_price": None,
            "discount_from_avg": None,
            "profit_per_kg": None
        },
        hide_index=True,
    )

@st.fragment(run_every=refresh_interval)
def render_visual_analysis():
    filtered_df = get_filtered_data()
    if filtered_df.empty: return
    
    # We need the full dataset for clustering context, but filtered_df is what we have.
    # Ideally we should use the full dataset for clustering, but let's stick to filtered for now or reload full.
    # The original code used 'df' (full data) for clustering.
    # Let's reload full data for clustering to be accurate.
    try:
        df = load_data(trend_hours=trend_window_hours)
        if df.empty:
            df = filtered_df
        else:
            # Recalculate metrics for full df
            df['roi'] = df.apply(lambda x: (x['profit'] / x['flea_price'] * 100) if x['flea_price'] > 0 else 0, axis=1)
            df['slots'] = df['width'] * df['height']
            df['profit_per_slot'] = df.apply(lambda x: x['profit'] / x['slots'] if x['slots'] > 0 else 0, axis=1)
            df['discount_from_avg'] = df['avg_24h_price'] - df['flea_price']
            df['discount_percent'] = df.apply(lambda x: (x['discount_from_avg'] / x['avg_24h_price'] * 100) if x['avg_24h_price'] > 0 else 0, axis=1)
    except:
        df = filtered_df

    if df.empty:
        return

    st.subheader("Market Analysis (ML Clustering)")
    if len(df) >= 3:
        try:
            # Simple K-Means clustering
            # Added 'volatility' and 'change_last_48h' to the clustering features
            features = ['profit', 'roi', 'flea_price', 'profit_per_slot', 'discount_percent', 'volatility', 'change_last_48h']
            
            # Ensure all features exist (handle missing trend data)
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            
            X = df[features].copy().fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Dynamic number of clusters based on data size, max 4
            n_clusters = min(4, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters to assign meaningful labels
            cluster_stats = df.groupby('cluster')[['profit', 'volatility']].mean()
            
            # Sort by profit to determine potential
            sorted_by_profit = cluster_stats.sort_values('profit', ascending=True)
            
            cluster_map = {}
            if n_clusters == 4:
                # If we have 4 clusters, we can try to map them to our specific categories
                # The top 2 profit clusters
                top_2_clusters = sorted_by_profit.index[-2:]
                bottom_2_clusters = sorted_by_profit.index[:-2]
                
                # Analyze top 2 for volatility
                top_c1 = top_2_clusters[0]
                top_c2 = top_2_clusters[1]
                
                # Use type: ignore to suppress Pylance Scalar errors
                vol1 = float(cluster_stats.loc[top_c1, 'volatility']) # type: ignore
                vol2 = float(cluster_stats.loc[top_c2, 'volatility']) # type: ignore
                
                if vol1 > vol2:
                    cluster_map[top_c1] = "High Potential (Volatile)"
                    cluster_map[top_c2] = "High Potential (Stable)"
                else:
                    cluster_map[top_c1] = "High Potential (Stable)"
                    cluster_map[top_c2] = "High Potential (Volatile)"
                    
                # Map bottom 2
                cluster_map[bottom_2_clusters[0]] = "Low Potential"
                cluster_map[bottom_2_clusters[1]] = "Medium Potential"
            else:
                # Fallback for fewer clusters
                for i, cluster_id in enumerate(sorted_by_profit.index):
                    cluster_map[cluster_id] = f"Tier {i+1} (Profit: {sorted_by_profit.loc[cluster_id, 'profit']:.0f})"

            df['cluster_label'] = df['cluster'].map(cluster_map).fillna("Unclassified")
            
            col_a, col_b = st.columns(2)
            with col_a:
                # Interactive Scatter Plot: Profit vs Volatility
                # Ensure size is positive for Plotly, but reflect magnitude of profit/loss
                df['plot_size'] = df['profit_per_slot'].abs().clip(lower=10)

                fig = px.scatter(
                    df, 
                    x='volatility', 
                    y='profit', 
                    color='cluster_label',
                    size='plot_size',
                    hover_data=['name', 'roi', 'profit_per_slot', 'discount_percent', 'avg_24h_price', 'change_last_48h', 'volatility'],
                    title='Risk vs Reward: Profit vs Volatility (7 Days)',
                    labels={'volatility': 'Volatility (Risk)', 'profit': 'Profit (Reward)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                if 'category' in df.columns:
                    pos_df = df[df['profit'] > 0]
                    if not pos_df.empty:
                        fig_tree = px.treemap(
                            pos_df,
                            path=['category', 'name'],
                            values='profit',
                            color='volatility',
                            color_continuous_scale='RdYlGn_r',
                            title='Profit Opportunities by Category (Color = Risk/Volatility)'
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)
            
            st.write("""
            **Strategy Guide (Updated):**
            *   **High Potential (Stable)**: The "Holy Grail". High profit, low volatility. Safe bets.
            *   **High Potential (Volatile)**: High profit, but prices swing wildly. Good for sniping, bad for holding.
            *   **Volatility**: Calculated as `Max Profit - Min Profit` over the last 7 days. A larger range means higher risk/instability.
            """)
            
            # New Chart: Profit Distribution
            st.markdown("### 📈 Market Profit Distribution")
            fig_hist = px.histogram(
                df, 
                x="profit", 
                nbins=50, 
                title="Distribution of Profit Margins",
                labels={'profit': 'Profit (RUB)'},
                color_discrete_sequence=['#3366cc']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("### 📊 Category Performance")
            cat_stats = df.groupby('category')[['profit', 'roi']].mean().reset_index()
            cat_stats = cat_stats.sort_values('profit', ascending=False).head(15)
            
            fig_cat = px.bar(
                cat_stats, 
                x='category', 
                y='profit',
                color='roi',
                title='Average Profit by Category (Color = Avg ROI)',
                labels={'profit': 'Avg Profit (RUB)', 'roi': 'Avg ROI (%)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        except Exception as e:
            st.error(f"Error in ML Analysis: {e}")
    else:
        st.info("Not enough data for ML analysis yet.")

@st.fragment(run_every=refresh_interval)
def render_item_history():
    filtered_df = get_filtered_data()
    if filtered_df.empty: return

    st.subheader("Item Analysis")
    unique_names = filtered_df['name'].unique()
    
    if len(unique_names) > 0:
        selected_item_name = st.selectbox("Select Item for Detailed History", unique_names)
        
        if selected_item_name:
            try:
                item_row = filtered_df[filtered_df['name'] == selected_item_name].iloc[0]
                item_id = item_row['item_id']
                
                history_data = database.get_item_history(item_id)
                if history_data:
                    hist_df = pd.DataFrame(history_data, columns=['timestamp', 'flea_price', 'trader_price', 'profit'])
                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                    
                    # Add summary metrics
                    latest = hist_df.iloc[-1]
                    avg_profit = hist_df['profit'].mean()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Profit", f"{latest['profit']:,} ₽")
                    m2.metric("Avg Profit (Period)", f"{avg_profit:,.0f} ₽")
                    m3.metric("Lowest Price", f"{hist_df['flea_price'].min():,.0f} ₽")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['profit'], mode='lines+markers', name='Profit'))
                    fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['flea_price'], mode='lines', name='Flea Price', line=dict(dash='dash')))
                    fig.update_layout(title=f"Profit History: {selected_item_name}", xaxis_title="Time", yaxis_title="RUB")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No history available for this item.")
            except Exception as e:
                st.error(f"Error fetching item history: {e}")
    else:
        st.info("No items match your filters.")

@st.fragment(run_every=refresh_interval)
def render_logs():
    st.subheader("System Logs")
    log_type = st.radio("Select Log File", ["Collector Logs (Background Process)", "App Logs (Dashboard Errors)", "Startup Logs (Collector Startup)"], horizontal=True)
    
    col_c1, col_c2 = st.columns([1, 5])
    with col_c1:
        if st.button("Refresh Logs"):
            pass # Fragment will rerun automatically, no need for st.rerun()
    
    if "Collector" in log_type:
        log_file_path = "collector.log"
    elif "Startup" in log_type:
        log_file_path = "collector_startup.log"
    else:
        log_file_path = "app.log"

    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
                last_lines = lines[-200:]
                log_content = "".join(last_lines)
                if not log_content.strip():
                    st.info(f"{log_file_path} is empty.")
                else:
                    st.code(log_content, language="text")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.info(f"No log file found at {log_file_path}.")

# Render Header
render_header_metrics()

# Render Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Table", "Visual Analysis", "Item History", "Console"])

with tab1:
    render_market_table()
with tab2:
    render_visual_analysis()
with tab3:
    render_item_history()
with tab4:
    render_logs()
