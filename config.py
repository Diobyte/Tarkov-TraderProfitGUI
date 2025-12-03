# config.py

# API Configuration
API_URL = 'https://api.tarkov.dev/graphql'
COLLECTION_INTERVAL_MINUTES = 5
DATA_RETENTION_DAYS = 7

# Database Configuration
DB_LOOKBACK_WINDOW_MINUTES = 45  # Window for fetching "latest" prices
LIQUIDITY_NORMALIZATION_THRESHOLD = 50  # Offer count considered "high liquidity"
MAX_LIQUIDITY_SCORE = 100  # Maximum liquidity score

# Cache Configuration  
STREAMLIT_CACHE_TTL_SECONDS = 60  # How long to cache data in Streamlit

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
