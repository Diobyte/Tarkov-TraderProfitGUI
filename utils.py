import pandas as pd
import numpy as np

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived metrics for the prices dataframe.
    Modifies the dataframe in-place.
    """
    if df.empty:
        return df

    # ROI
    # Avoid division by zero
    df['roi'] = df.apply(lambda x: (x['profit'] / x['flea_price'] * 100) if x['flea_price'] > 0 else 0, axis=1)
    
    # Slots & Profit per Slot
    df['slots'] = df['width'] * df['height']
    df['profit_per_slot'] = df.apply(lambda x: x['profit'] / x['slots'] if x['slots'] > 0 else 0, axis=1)
    
    # Discount from Average
    df['discount_from_avg'] = df['avg_24h_price'] - df['flea_price']
    df['discount_percent'] = df.apply(
        lambda x: (x['discount_from_avg'] / x['avg_24h_price'] * 100) if x['avg_24h_price'] > 0 else 0, 
        axis=1
    )
    
    # Profit per Kg
    df['profit_per_kg'] = df.apply(lambda x: x['profit'] / x['weight'] if x['weight'] > 0 else 0, axis=1)
    
    return df
