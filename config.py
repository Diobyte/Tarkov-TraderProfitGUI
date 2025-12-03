# config.py
"""Configuration constants for Tarkov Trader Profit application.

All configuration values can be overridden via environment variables
using the pattern TARKOV_{CONSTANT_NAME}.

Example:
    export TARKOV_COLLECTION_INTERVAL_MINUTES=10
    export TARKOV_DATA_RETENTION_DAYS=14

Note:
    Values are validated and logged when invalid values are provided.
    Invalid values fall back to defaults to ensure application stability.
"""

import os
import logging
from typing import Dict, Final

logger = logging.getLogger(__name__)


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable or use default.
    
    Args:
        key: Environment variable suffix (will be prefixed with TARKOV_).
        default: Default value if env var is not set or invalid.
        
    Returns:
        Integer value from environment or default.
    """
    env_value = os.environ.get(f'TARKOV_{key}')
    if env_value is None:
        return default
    try:
        result = int(env_value)
        # Validate non-negative for most settings
        if result < 0 and key not in ('MIN_PROFIT_FOR_DISPLAY',):
            logger.warning("Negative value for TARKOV_%s: %d, using default %d", key, result, default)
            return default
        return result
    except ValueError:
        logger.warning("Invalid integer for TARKOV_%s: %r, using default %d", key, env_value, default)
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable or use default.
    
    Args:
        key: Environment variable suffix (will be prefixed with TARKOV_).
        default: Default value if env var is not set or invalid.
        
    Returns:
        Float value from environment or default.
    """
    env_value = os.environ.get(f'TARKOV_{key}')
    if env_value is None:
        return default
    try:
        return float(env_value)
    except ValueError:
        logger.warning("Invalid float for TARKOV_%s: %r, using default %s", key, env_value, default)
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string value from environment variable or use default."""
    return os.environ.get(f'TARKOV_{key}', default)


# API Configuration
API_URL: Final[str] = _get_env_str('API_URL', 'https://api.tarkov.dev/graphql')
COLLECTION_INTERVAL_MINUTES: Final[int] = _get_env_int('COLLECTION_INTERVAL_MINUTES', 5)
DATA_RETENTION_DAYS: Final[int] = _get_env_int('DATA_RETENTION_DAYS', 7)
API_TIMEOUT_SECONDS: Final[int] = _get_env_int('API_TIMEOUT_SECONDS', 30)

# Database Configuration
DB_LOOKBACK_WINDOW_MINUTES: int = _get_env_int('DB_LOOKBACK_WINDOW_MINUTES', 45)
LIQUIDITY_NORMALIZATION_THRESHOLD: int = _get_env_int('LIQUIDITY_NORMALIZATION_THRESHOLD', 50)
MAX_LIQUIDITY_SCORE: int = _get_env_int('MAX_LIQUIDITY_SCORE', 100)

# Cache Configuration
STREAMLIT_CACHE_TTL_SECONDS: int = _get_env_int('STREAMLIT_CACHE_TTL_SECONDS', 60)
LOG_MAX_LINES: int = _get_env_int('LOG_MAX_LINES', 100)

# ML Engine Configuration
ML_ANOMALY_CONTAMINATION: float = _get_env_float('ML_ANOMALY_CONTAMINATION', 0.05)
ML_ESTIMATORS: int = _get_env_int('ML_ESTIMATORS', 100)
ML_MIN_ITEMS_FOR_ANALYSIS: int = _get_env_int('ML_MIN_ITEMS_FOR_ANALYSIS', 10)
ML_MIN_ITEMS_FOR_ANOMALY: int = _get_env_int('ML_MIN_ITEMS_FOR_ANOMALY', 20)

# Trend Learning Configuration
TREND_LOOKBACK_HOURS: int = _get_env_int('TREND_LOOKBACK_HOURS', 24)
TREND_MIN_DATA_POINTS: int = _get_env_int('TREND_MIN_DATA_POINTS', 6)
TREND_PROFIT_MOMENTUM_WEIGHT: float = _get_env_float('TREND_PROFIT_MOMENTUM_WEIGHT', 0.20)
TREND_VOLATILITY_PENALTY: float = _get_env_float('TREND_VOLATILITY_PENALTY', 0.15)
TREND_CONSISTENCY_BONUS: float = _get_env_float('TREND_CONSISTENCY_BONUS', 0.25)
TREND_IMPROVEMENT_THRESHOLD: float = _get_env_float('TREND_IMPROVEMENT_THRESHOLD', 0.05)

# Volume/Offers Thresholds
VOLUME_MIN_FOR_RECOMMENDATION: int = _get_env_int('VOLUME_MIN_FOR_RECOMMENDATION', 5)
VOLUME_LOW_THRESHOLD: int = _get_env_int('VOLUME_LOW_THRESHOLD', 10)
VOLUME_MEDIUM_THRESHOLD: int = _get_env_int('VOLUME_MEDIUM_THRESHOLD', 50)
VOLUME_HIGH_THRESHOLD: int = _get_env_int('VOLUME_HIGH_THRESHOLD', 100)
VOLUME_VERY_HIGH_THRESHOLD: int = _get_env_int('VOLUME_VERY_HIGH_THRESHOLD', 200)
VOLUME_WEIGHT_IN_SCORE: float = _get_env_float('VOLUME_WEIGHT_IN_SCORE', 0.15)

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
ALERT_DEFAULT_COOLDOWN_MINUTES: int = _get_env_int('ALERT_DEFAULT_COOLDOWN_MINUTES', 30)
ALERT_HIGH_PROFIT_THRESHOLD: int = _get_env_int('ALERT_HIGH_PROFIT_THRESHOLD', 10000)
ALERT_HIGH_ROI_THRESHOLD: float = _get_env_float('ALERT_HIGH_ROI_THRESHOLD', 50.0)
ALERT_MAX_HISTORY: int = _get_env_int('ALERT_MAX_HISTORY', 500)

# Export Configuration
EXPORT_MAX_ROWS: int = _get_env_int('EXPORT_MAX_ROWS', 1000)
EXPORT_CLEANUP_DAYS: int = _get_env_int('EXPORT_CLEANUP_DAYS', 7)

# Performance Configuration
DATABASE_CONNECTION_TIMEOUT: int = _get_env_int('DATABASE_CONNECTION_TIMEOUT', 30)
DATABASE_RETRY_ATTEMPTS: int = _get_env_int('DATABASE_RETRY_ATTEMPTS', 5)
DATABASE_RETRY_DELAY: float = _get_env_float('DATABASE_RETRY_DELAY', 1.0)
DATABASE_BUSY_TIMEOUT_MS: int = _get_env_int('DATABASE_BUSY_TIMEOUT_MS', 30000)

# UI Configuration
UI_REFRESH_INTERVAL_SECONDS: int = _get_env_int('UI_REFRESH_INTERVAL_SECONDS', 60)
UI_MAX_TABLE_ROWS: int = _get_env_int('UI_MAX_TABLE_ROWS', 100)
UI_CHART_HEIGHT: int = _get_env_int('UI_CHART_HEIGHT', 400)

# Data Quality Configuration
MIN_PROFIT_FOR_DISPLAY: int = _get_env_int('MIN_PROFIT_FOR_DISPLAY', 0)
MIN_OFFERS_FOR_RELIABLE: int = _get_env_int('MIN_OFFERS_FOR_RELIABLE', 5)
MAX_PRICE_AGE_HOURS: int = _get_env_int('MAX_PRICE_AGE_HOURS', 1)
