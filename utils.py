"""Shared utility functions for Tarkov Trader Profit application."""

import pandas as pd
import numpy as np
from typing import Optional, Union, List

__all__: List[str] = ['calculate_metrics', 'calculate_flea_market_fee', 'format_roubles']


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived metrics for the prices dataframe.
    
    Creates a copy of the input DataFrame to avoid SettingWithCopyWarning
    and adds the following calculated columns:
    
    - roi: Return on Investment (profit/flea_price * 100)
    - slots: Total inventory slots (width * height)
    - profit_per_slot: Profit divided by inventory slots
    - discount_from_avg: Difference between avg_24h_price and flea_price
    - discount_percent: Discount as percentage of avg_24h_price
    - profit_per_kg: Profit divided by item weight
    - flea_to_base_ratio: Ratio of flea_price to base_price (if available)
    - price_range_24h: High - Low price over 24h (if available)
    - price_range_percent: Price range as percentage of avg
    - liquidity_tier: Categorical tier based on offer count
    - opportunity_score: Combined score based on multiple factors
    
    Args:
        df: DataFrame with price data including profit, flea_price, width, height, etc.
        
    Returns:
        DataFrame with added calculated metrics. Returns empty DataFrame if input is empty.
        
    Note:
        Division by zero is handled gracefully by replacing with 0 or inf as appropriate.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure required columns exist; if missing, fill with safe defaults
    required_numeric = [
        'profit', 'flea_price', 'width', 'height', 'avg_24h_price',
        'low_24h_price', 'weight'
    ]
    for col in required_numeric:
        if col not in df.columns:
            df[col] = 0

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
        df['last_offer_count'] = pd.to_numeric(df['last_offer_count'], errors='coerce').fillna(0)
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
        profit_max = df['profit'].max()
        max_profit = profit_max if pd.notna(profit_max) and profit_max > 0 else 1
        roi_max = df['roi'].max()
        max_roi = roi_max if pd.notna(roi_max) and roi_max > 0 else 1
        
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
    # Validate inputs - base_price must be positive for meaningful calculation
    if not isinstance(base_price, (int, float)) or not isinstance(sell_price, (int, float)):
        return 0
    if base_price <= 0 or sell_price <= 0:
        return 0
    
    # Validate intel_center_level
    intel_center_level = max(0, min(3, int(intel_center_level)))
    
    # Fee reduction based on intel center
    fee_reduction = {0: 1.0, 1: 0.95, 2: 0.90, 3: 0.85}
    modifier = fee_reduction.get(intel_center_level, 1.0)
    
    # Tarkov fee formula (simplified approximation)
    # Real formula is more complex but this is a reasonable approximation
    vo = float(base_price)  # Ensure float division
    vr = float(sell_price)
    
    # Base fee rate
    try:
        if vr >= vo:
            # Selling above base price
            q = 1.0
            ratio = vr / vo  # Safe: vo > 0 checked above
            # Cap the exponent to prevent overflow
            exponent = min(q * (ratio - 1), 10)
            fee = vo * 0.05 * (4 ** exponent)
        else:
            # Selling below base price
            fee = vo * 0.05
        
        # Ensure fee doesn't exceed reasonable bounds
        return int(min(fee * modifier, 2_000_000_000))  # Cap at 2B rubles
    except (OverflowError, ValueError, ZeroDivisionError):
        # Handle extreme values gracefully
        return int(vo * 0.05 * modifier)


def format_roubles(value: Union[int, float, None]) -> str:
    """Format a number as roubles with thousand separators.
    
    Args:
        value: Numeric value to format (int, float, or None).
        
    Returns:
        Formatted string with rouble symbol and thousand separators.
    """
    if value is None:
        return "0 ₽"
    try:
        # Handle numpy/pandas types and special float values first
        # Check for NaN/inf before item() conversion to avoid conversion errors
        if hasattr(value, 'item'):
            # For numpy types, check isnan/isinf before converting
            if np.isnan(value) or np.isinf(value):  # type: ignore[arg-type]
                return "0 ₽"
            value = value.item()  # type: ignore[union-attr]
        if isinstance(value, float) and (pd.isna(value) or np.isinf(value)):
            return "0 ₽"
        return f"{int(value):,} ₽"
    except (ValueError, TypeError, OverflowError):
        return "0 ₽"
