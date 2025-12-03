"""Shared utility functions for Tarkov Trader Profit application."""

import pandas as pd
from typing import Optional, Union

__all__ = ['calculate_metrics', 'calculate_flea_market_fee', 'format_roubles']

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived metrics for the prices dataframe.
    
    Note: Creates a copy to avoid SettingWithCopyWarning.
    
    Args:
        df: DataFrame with price data including profit, flea_price, width, height, etc.
        
    Returns:
        DataFrame with added calculated metrics (roi, slots, profit_per_slot, etc.).
    """
    if df.empty:
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # ROI - Return on Investment (Profit / Cost * 100)
    # Avoid division by zero using vectorized operations (faster than apply)
    df['roi'] = df['profit'] / df['flea_price'].replace(0, float('inf')) * 100
    df.loc[df['flea_price'] == 0, 'roi'] = 0
    
    # Slots & Profit per Slot (vectorized for performance)
    df['slots'] = df['width'] * df['height']
    df['profit_per_slot'] = df['profit'] / df['slots'].replace(0, float('inf'))
    df.loc[df['slots'] == 0, 'profit_per_slot'] = 0
    
    # Discount from Average (how much below avg 24h price we're buying)
    df['discount_from_avg'] = df['avg_24h_price'] - df['flea_price']
    df['discount_percent'] = df['discount_from_avg'] / df['avg_24h_price'].replace(0, float('inf')) * 100
    df.loc[df['avg_24h_price'] == 0, 'discount_percent'] = 0
    
    # Profit per Kg (useful for weight-limited runs)
    df['profit_per_kg'] = df['profit'] / df['weight'].replace(0, float('inf'))
    df.loc[df['weight'] == 0, 'profit_per_kg'] = 0
    
    # --- Enhanced Metrics (v2) ---
    
    # Base Price Ratio - How flea price compares to game's base value
    # Lower ratio = potentially undervalued on flea
    if 'base_price' in df.columns:
        df['base_price'] = df['base_price'].fillna(0)
        df['flea_to_base_ratio'] = df['flea_price'] / df['base_price'].replace(0, float('inf'))
        df.loc[df['base_price'] == 0, 'flea_to_base_ratio'] = 0
    
    # Price Range (24h High - Low) - Market volatility indicator
    if 'high_24h_price' in df.columns:
        df['high_24h_price'] = df['high_24h_price'].fillna(0)
        df['price_range_24h'] = df['high_24h_price'] - df['low_24h_price']
        df['price_range_percent'] = df['price_range_24h'] / df['avg_24h_price'].replace(0, float('inf')) * 100
        df.loc[df['avg_24h_price'] == 0, 'price_range_percent'] = 0
    
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
