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
