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
from typing import Dict, Final, List

__all__: List[str] = [
    'API_URL', 'COLLECTION_INTERVAL_MINUTES', 'DATA_RETENTION_DAYS', 'API_TIMEOUT_SECONDS',
    'DB_LOOKBACK_WINDOW_MINUTES', 'LIQUIDITY_NORMALIZATION_THRESHOLD', 'MAX_LIQUIDITY_SCORE',
    'STREAMLIT_CACHE_TTL_SECONDS', 'LOG_MAX_LINES',
    'ML_ANOMALY_CONTAMINATION', 'ML_ESTIMATORS', 'ML_MIN_ITEMS_FOR_ANALYSIS', 'ML_MIN_ITEMS_FOR_ANOMALY',
    'TREND_LOOKBACK_HOURS', 'TREND_MIN_DATA_POINTS', 'TREND_PROFIT_MOMENTUM_WEIGHT',
    'TREND_VOLATILITY_PENALTY', 'TREND_CONSISTENCY_BONUS', 'TREND_IMPROVEMENT_THRESHOLD',
    'VOLUME_MIN_FOR_RECOMMENDATION', 'VOLUME_LOW_THRESHOLD', 'VOLUME_MEDIUM_THRESHOLD',
    'VOLUME_HIGH_THRESHOLD', 'VOLUME_VERY_HIGH_THRESHOLD', 'VOLUME_WEIGHT_IN_SCORE',
    'CATEGORY_LOCKS', 'ITEM_LOCKS', 'FLEA_MARKET_UNLOCK_LEVEL',
    'ALERT_DEFAULT_COOLDOWN_MINUTES', 'ALERT_HIGH_PROFIT_THRESHOLD', 'ALERT_HIGH_ROI_THRESHOLD', 'ALERT_MAX_HISTORY',
    'EXPORT_MAX_ROWS', 'EXPORT_CLEANUP_DAYS',
    'DATABASE_CONNECTION_TIMEOUT', 'DATABASE_RETRY_ATTEMPTS', 'DATABASE_RETRY_DELAY', 'DATABASE_BUSY_TIMEOUT_MS',
    'UI_REFRESH_INTERVAL_SECONDS', 'UI_MAX_TABLE_ROWS', 'UI_CHART_HEIGHT',
    'MIN_PROFIT_FOR_DISPLAY', 'MIN_OFFERS_FOR_RELIABLE', 'MAX_PRICE_AGE_HOURS',
]

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

# Flea Market Level Requirements (Based on Patch 1.0 changes)
# Default flea market unlock is level 15. Categories/items below require higher levels.
# Format: category/item name -> minimum player level required

CATEGORY_LOCKS: Dict[str, int] = {
    # === WEAPONS ===
    "Assault rifle": 25,
    "Assault carbine": 25,
    "Bolt-action rifle": 20,
    "Sniper rifle": 20,
    "Marksman rifle": 25,
    "Submachine gun": 20,
    "Machine gun": 25,
    "Throwable weapon": 25,
    
    # === WEAPON MODS ===
    "Foregrip": 20,
    "Auxiliary part": 25,
    "Flashlight": 25,
    "Tactical combo device": 25,
    "Laser sight": 25,
    "Muzzle device": 20,
    "Muzzle adapter": 20,
    "Flash hider": 20,
    "Suppressor": 25,
    "Silencer": 25,
    
    # === AMMO ===
    "Ammo pack": 25,
    "Ammunition pack": 25,
    
    # === GEAR ===
    "Backpack": 25,
    "Headwear": 20,
    "Eyewear": 20,
    "Armor component": 30,
    "Gear component": 30,
    "Armored equipment": 25,
    "Container": 25,
    "Secure container": 25,
    
    # === BARTER ITEMS ===
    "Electronics": 20,
    "Energy element": 20,
    "Battery": 20,
    "Flammable": 20,
    "Household goods": 20,
    "Household material": 20,
    "Medical supply": 20,
    "Medical supplies": 20,
    "Valuable": 30,
    "Valuables": 30,
    "Other": 25,
    
    # === MEDICAL ===
    "Stimulant": 30,
    "Injector": 30,
    "Injury treatment": 20,
    "Medkit": 20,
    "Pills": 20,
    
    # === KEYS ===
    "Mechanical key": 25,
    "Electronic key": 30,
    "Keycard": 30,
}

# Specific item level locks (overrides category locks)
# These are specific high-value or restricted items
ITEM_LOCKS: Dict[str, int] = {
    # === ELECTRONICS (Level 20 base, specific items higher) ===
    "Graphics card": 40,
    "GPU": 40,
    "Military circuit board": 35,
    "Military power filter": 35,
    "Phased array element": 40,
    "Tetriz portable game console": 35,
    "UHF RFID Reader": 35,
    "VPX Flash Storage Module": 20,
    "Virtex programmable processor": 35,
    
    # === ENERGY ELEMENTS ===
    "6-STEN-140-M military battery": 40,
    "GreenBat lithium battery": 35,
    
    # === FILTERS ===
    "FP-100 filter absorber": 35,
    
    # === FLAMMABLE / GUNPOWDER ===
    "Gunpowder Eagle": 30,
    "Gunpowder Hawk": 30,
    
    # === MEDICAL SUPPLIES ===
    "LEDX Skin Transilluminator": 35,
    
    # === MEDICAL - INJURY TREATMENT ===
    "CALOK-B hemostatic applicator": 30,
    "Surv12 field surgical kit": 35,
    
    # === STIMULANTS / INJECTORS ===
    "MULE stimulant injector": 40,
    "eTG-change regenerative stimulant injector": 40,
    
    # === GEAR COMPONENTS / ARMOR PLATES ===
    "Granit 4 ballistic plate": 40,
    "Korund-VM ballistic plate": 40,
    
    # === HEADGEAR ===
    "DevTac Ronin ballistic helmet": 40,
    "Vulkan-5 (LShZ-5) heavy helmet": 40,
    
    # === CONTAINERS ===
    "Injector case": 35,
    "S I C C organizational pouch": 40,
    "SICC organizational pouch": 40,
    
    # === WEAPONS (specific high-tier) ===
    "Accuracy International AXMC .338 LM bolt-action sniper rifle": 40,
    "SWORD International Mk-18 .338 LM marksman rifle": 40,
    
    # === AMMO PACKS ===
    ".300 Blackout CBJ ammo pack": 40,
    ".308 M80 ammo pack": 35,
    ".366 TKM AP-M ammo pack": 35,
    "12.7x55mm PS12B ammo pack": 40,
    "12.7x55mm PS12B": 40,
    "PS12B": 40,
    "23x75mm Zvezda flashbang round ammo pack": 40,
    "5.45x39mm 7N40 ammo pack": 35,
    "5.56x45mm M855A1 ammo pack": 40,
    "7.62x39mm PP gzh ammo pack": 30,
    "7.62x51mm M80 ammo pack": 35,
    "M80": 35,
    "7.62x54mm PS gzh ammo pack": 40,
    "9x39mm PAB-9 gs ammo pack": 40,
    ".300 Blackout": 40,
    "Blackout CBJ": 40,
    
    # === MARKED KEYS ===
    "Abandoned factory marked key": 35,
    "Dorm room 314 marked key": 35,
    "Mysterious room marked key": 35,
    "RB-BK marked key": 35,
    "RB-PKPM marked key": 35,
    "Shared bedroom marked key": 35,
}

# Default flea market unlock level
FLEA_MARKET_UNLOCK_LEVEL: int = 15

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
