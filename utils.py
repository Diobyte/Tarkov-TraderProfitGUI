import pandas as pd
import numpy as np
from typing import Optional, Union

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived metrics for the prices dataframe.
    Modifies the dataframe in-place.
    """
    if df.empty:
        return df

    # ROI - Return on Investment (Profit / Cost * 100)
    # Avoid division by zero
    df['roi'] = df.apply(lambda x: (x['profit'] / x['flea_price'] * 100) if x['flea_price'] > 0 else 0, axis=1)
    
    # Slots & Profit per Slot
    df['slots'] = df['width'] * df['height']
    df['profit_per_slot'] = df.apply(lambda x: x['profit'] / x['slots'] if x['slots'] > 0 else 0, axis=1)
    
    # Discount from Average (how much below avg 24h price we're buying)
    df['discount_from_avg'] = df['avg_24h_price'] - df['flea_price']
    df['discount_percent'] = df.apply(
        lambda x: (x['discount_from_avg'] / x['avg_24h_price'] * 100) if x['avg_24h_price'] > 0 else 0, 
        axis=1
    )
    
    # Profit per Kg (useful for weight-limited runs)
    df['profit_per_kg'] = df.apply(lambda x: x['profit'] / x['weight'] if x['weight'] > 0 else 0, axis=1)
    
    # --- Enhanced Metrics (v2) ---
    
    # Base Price Ratio - How flea price compares to game's base value
    # Lower ratio = potentially undervalued on flea
    if 'base_price' in df.columns:
        df['base_price'] = df['base_price'].fillna(0)
        df['flea_to_base_ratio'] = df.apply(
            lambda x: (x['flea_price'] / x['base_price']) if x['base_price'] > 0 else 0, 
            axis=1
        )
    
    # Price Range (24h High - Low) - Market volatility indicator
    if 'high_24h_price' in df.columns:
        df['high_24h_price'] = df['high_24h_price'].fillna(0)
        df['price_range_24h'] = df['high_24h_price'] - df['low_24h_price']
        df['price_range_percent'] = df.apply(
            lambda x: (x['price_range_24h'] / x['avg_24h_price'] * 100) if x['avg_24h_price'] > 0 else 0,
            axis=1
        )
    
    # Liquidity indicator (offer count normalized)
    if 'last_offer_count' in df.columns:
        df['last_offer_count'] = df['last_offer_count'].fillna(0)
        # Classify: Low (0-10), Medium (10-50), High (50+)
        df['liquidity_tier'] = df['last_offer_count'].apply(
            lambda x: 'High' if x >= 50 else ('Medium' if x >= 10 else 'Low')
        )
    
    # Trader Level Accessibility
    if 'trader_level_required' in df.columns:
        df['trader_level_required'] = df['trader_level_required'].fillna(1).astype(int)
    
    # Combined "Opportunity Score" - Higher = Better opportunity
    # Factors: profit margin, liquidity, price velocity, stability
    if 'liquidity_score' in df.columns and 'price_velocity' in df.columns:
        df['liquidity_score'] = df['liquidity_score'].fillna(0)
        df['price_velocity'] = df['price_velocity'].fillna(0)
        
        # Normalize components to 0-1 scale for scoring
        # Handle edge cases where max could be 0 or negative
        max_profit = max(df['profit'].max(), 1)
        max_roi = max(df['roi'].max(), 1)
        
        # Clip negative values to 0 for normalization to prevent negative scores
        df['opportunity_score'] = (
            (df['profit'].clip(lower=0) / max_profit * 0.30) +  # 30% weight on raw profit
            (df['roi'].clip(lower=0) / max_roi * 0.25) +         # 25% weight on ROI
            (df['liquidity_score'].clip(0, 100) / 100 * 0.20) + # 20% weight on liquidity
            (df['price_velocity'].clip(0, 100) / 100 * 0.15) +  # 15% weight on price velocity (clipped)
            (df['discount_percent'].clip(0, 50) / 50 * 0.10)    # 10% weight on discount
        ) * 100
    
    return df


def calculate_flea_market_fee(base_price: int, sell_price: int, intel_center_level: int = 0) -> int:
    """
    Calculate the flea market fee for listing an item.
    
    Args:
        base_price: The game's base price for the item
        sell_price: The price you want to list at
        intel_center_level: 0, 1, 2, or 3 (reduces fee by 0%, 5%, 10%, 15%)
    
    Returns:
        The fee in rubles
    """
    if base_price <= 0 or sell_price <= 0:
        return 0
    
    # Fee reduction based on intel center
    fee_reduction = {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.85}
    modifier = fee_reduction.get(intel_center_level, 1.0)
    
    # Tarkov fee formula (simplified approximation)
    # Real formula is more complex but this is a reasonable approximation
    vo = base_price
    vr = sell_price
    
    # Base fee rate
    if vr >= vo:
        # Selling above base price
        q = 1.0
        fee = vo * 0.05 * 4 ** (q * (vr / vo - 1))
    else:
        # Selling below base price
        fee = vo * 0.05
    
    return int(fee * modifier)


def format_roubles(value: Union[int, float, None]) -> str:
    """Format a number as roubles with thousand separators."""
    if value is None or pd.isna(value):
        return "0 ₽"
    try:
        return f"{int(value):,} ₽"
    except (ValueError, TypeError):
        return "0 ₽"
