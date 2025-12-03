# config.py
"""Configuration constants for Tarkov Trader Profit application."""

from typing import Dict

# API Configuration
API_URL: str = 'https://api.tarkov.dev/graphql'
COLLECTION_INTERVAL_MINUTES: int = 5
DATA_RETENTION_DAYS: int = 7
API_TIMEOUT_SECONDS: int = 30

# Database Configuration
DB_LOOKBACK_WINDOW_MINUTES: int = 45  # Window for fetching "latest" prices
LIQUIDITY_NORMALIZATION_THRESHOLD: int = 50  # Offer count considered "high liquidity"
MAX_LIQUIDITY_SCORE: int = 100  # Maximum liquidity score

# Cache Configuration
STREAMLIT_CACHE_TTL_SECONDS: int = 60  # How long to cache data in Streamlit
LOG_MAX_LINES: int = 100  # Maximum lines to display from log files

# ML Engine Configuration
ML_ANOMALY_CONTAMINATION: float = 0.05  # Fraction of outliers for anomaly detection
ML_ESTIMATORS: int = 100  # Number of estimators for ensemble models
ML_MIN_ITEMS_FOR_ANALYSIS: int = 10  # Minimum items required for ML analysis
ML_MIN_ITEMS_FOR_ANOMALY: int = 20  # Minimum items for anomaly detection

# Trend Learning Configuration
TREND_LOOKBACK_HOURS: int = 24  # Hours of history to analyze for trends
TREND_MIN_DATA_POINTS: int = 6  # Minimum data points for valid trend
TREND_PROFIT_MOMENTUM_WEIGHT: float = 0.20  # Weight of profit trend in scoring
TREND_VOLATILITY_PENALTY: float = 0.15  # Penalty factor for high volatility items
TREND_CONSISTENCY_BONUS: float = 0.25  # Bonus for consistently profitable items
TREND_IMPROVEMENT_THRESHOLD: float = 0.05  # 5% improvement = positive trend

# Volume/Offers Thresholds
VOLUME_MIN_FOR_RECOMMENDATION: int = 5  # Minimum offers required to recommend
VOLUME_LOW_THRESHOLD: int = 10  # Below this = low volume/hard to buy
VOLUME_MEDIUM_THRESHOLD: int = 50  # Below this = medium volume
VOLUME_HIGH_THRESHOLD: int = 100  # Above this = high volume/saturated
VOLUME_VERY_HIGH_THRESHOLD: int = 200  # Above this = very high volume
VOLUME_WEIGHT_IN_SCORE: float = 0.15  # Weight of volume in opportunity scoring

# Flea Market Level Requirements (Based on Patch 0.15+ changes)
CATEGORY_LOCKS: Dict[str, int] = {
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

ITEM_LOCKS: Dict[str, int] = {
    "PS12B": 40,
    "M80": 35,
    "Blackout CJB": 40,
}

# Alert Configuration
ALERT_DEFAULT_COOLDOWN_MINUTES: int = 30  # Cooldown between alert triggers
ALERT_HIGH_PROFIT_THRESHOLD: int = 10000  # Threshold for high profit alerts
ALERT_HIGH_ROI_THRESHOLD: float = 50.0  # Threshold for high ROI alerts
ALERT_MAX_HISTORY: int = 500  # Maximum alert history entries

# Export Configuration
EXPORT_MAX_ROWS: int = 1000  # Maximum rows to export at once
EXPORT_CLEANUP_DAYS: int = 7  # Delete exports older than this

# Performance Configuration
DATABASE_CONNECTION_TIMEOUT: int = 30  # Database connection timeout in seconds
DATABASE_RETRY_ATTEMPTS: int = 5  # Number of retry attempts for DB operations
DATABASE_RETRY_DELAY: float = 1.0  # Delay between retries in seconds

# UI Configuration
UI_REFRESH_INTERVAL_SECONDS: int = 60  # Auto-refresh interval
UI_MAX_TABLE_ROWS: int = 100  # Maximum rows in data tables
UI_CHART_HEIGHT: int = 400  # Default chart height in pixels

# Data Quality Configuration
MIN_PROFIT_FOR_DISPLAY: int = 0  # Minimum profit to show item
MIN_OFFERS_FOR_RELIABLE: int = 5  # Minimum offers for reliable data
MAX_PRICE_AGE_HOURS: int = 1  # Maximum age of price data to consider fresh
