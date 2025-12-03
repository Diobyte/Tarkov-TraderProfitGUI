"""
Advanced Machine Learning Engine for Tarkov Trader Profit Analysis.

This module provides sophisticated ML algorithms for:
- Price prediction and trend forecasting
- Anomaly detection for arbitrage opportunities  
- Item clustering and similarity analysis
- Risk assessment and portfolio optimization
- Time-series analysis for optimal trading windows
- Historical trend learning for improved recommendations over time
- Persistent model training that survives database cleanups
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import warnings
import logging
from datetime import datetime, timedelta

# Suppress sklearn convergence warnings that are expected with small datasets
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import config
from model_persistence import get_model_persistence, ModelPersistence

__all__ = ['TarkovMLEngine', 'get_ml_engine']

# Configure logging
logger = logging.getLogger(__name__)


class TarkovMLEngine:
    """
    Advanced ML engine for Tarkov market analysis.
    
    Thread-safe singleton providing comprehensive ML-powered market analysis
    for Escape from Tarkov trading. Combines multiple ML techniques for
    optimal trading recommendations.
    
    Features:
        - Feature engineering for game economy metrics
        - Opportunity scoring with adaptive weights  
        - Anomaly detection for arbitrage opportunities (Isolation Forest)
        - Item clustering for strategy grouping (K-Means)
        - Profit trend prediction (Ridge regression)
        - Risk assessment with multi-factor scoring
        - Historical trend learning for improved recommendations
        - Persistent model state that survives database cleanups
    
    Attributes:
        scaler (RobustScaler): For handling outliers common in game economies.
        price_predictor: Reserved for future advanced prediction models.
        anomaly_detector (IsolationForest): For detecting unusual pricing.
        item_clusterer (KMeans): For grouping similar trading opportunities.
        trend_data (pd.DataFrame): Cached historical trend data for items.
        profit_stats (Dict): Global profit statistics for calibration.
        persistence (ModelPersistence): Model persistence layer for saving/loading state.
        
    Example:
        >>> engine = get_ml_engine()
        >>> df = engine.calculate_opportunity_score_ml(market_data)
        >>> recommendations = engine.generate_trading_recommendations(df)
        
    Thread Safety:
        Use get_ml_engine() to get the singleton instance. The class uses
        double-check locking for thread-safe initialization.
    """
    
    def __init__(self) -> None:
        """Initialize the ML engine with default scalers and empty model slots."""
        self.scaler = RobustScaler()
        self.price_predictor: Optional[Any] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.item_clusterer: Optional[KMeans] = None
        self._is_fitted: bool = False
        # Trend learning state
        self.trend_data: Optional[pd.DataFrame] = None
        self.profit_stats: Optional[Dict[str, Any]] = None
        self._last_trend_update: Optional[datetime] = None
        # Persistent model storage
        self.persistence: ModelPersistence = get_model_persistence()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for ML models.
        
        Combines domain knowledge about Tarkov economy with statistical features
        to create meaningful predictors for trading analysis.
        
        Args:
            df: DataFrame containing raw market data with columns like
                profit, flea_price, trader_price, width, height, weight, etc.
        
        Returns:
            DataFrame with added feature columns including profit_margin,
            capital_efficiency, slots, profit_per_slot, density, price_spread,
            liquidity_score, and more.
        """
        if df.empty:
            return df
        
        features = df.copy()
        
        # --- Core Financial Metrics ---
        # Profit margin as percentage of trader price (how much of sale is profit)
        features['profit_margin'] = np.where(
            features['trader_price'] > 0,
            features['profit'] / features['trader_price'] * 100,
            0
        )
        
        # Capital efficiency - profit relative to investment
        features['capital_efficiency'] = np.where(
            features['flea_price'] > 0,
            features['profit'] / features['flea_price'],
            0
        )
        
        # --- Slot Efficiency Metrics (Critical for Tarkov inventory management) ---
        features['slots'] = features['width'] * features['height']
        features['profit_per_slot'] = np.where(
            features['slots'] > 0,
            features['profit'] / features['slots'],
            0
        )
        
        # Density score - combines weight and slots for backpack optimization
        features['density'] = np.where(
            features['slots'] > 0,
            features['weight'] / features['slots'],
            0
        )
        
        # Profit density - profit per unit of "space-weight"
        features['profit_density'] = np.where(
            (features['slots'] * features['weight']) > 0,
            features['profit'] / (features['slots'] * np.maximum(features['weight'], 0.1)),
            features['profit_per_slot']
        )
        
        # --- Price Dynamics ---
        # Volatility indicators
        if 'high_24h_price' in features.columns and 'low_24h_price' in features.columns:
            features['price_spread'] = features['high_24h_price'] - features['low_24h_price']
            features['price_spread_pct'] = np.where(
                features['avg_24h_price'] > 0,
                features['price_spread'] / features['avg_24h_price'] * 100,
                0
            )
            
            # Current price position within 24h range (0 = at low, 1 = at high)
            features['price_position'] = np.where(
                features['price_spread'] > 0,
                (features['flea_price'] - features['low_24h_price']) / features['price_spread'],
                0.5
            )
        
        # Price momentum (change direction and magnitude)
        if 'change_last_48h' in features.columns:
            features['momentum'] = features['change_last_48h'].fillna(0)
            features['momentum_abs'] = features['momentum'].abs()
            features['is_declining'] = (features['momentum'] < -5).astype(int)
            features['is_rising'] = (features['momentum'] > 5).astype(int)
        
        # --- Value Indicators ---
        if 'base_price' in features.columns:
            features['base_price'] = features['base_price'].fillna(0)
            # Flea premium over base (negative = discount)
            features['flea_premium'] = np.where(
                features['base_price'] > 0,
                (features['flea_price'] - features['base_price']) / features['base_price'] * 100,
                0
            )
            
            # Trader premium over base
            features['trader_premium'] = np.where(
                features['base_price'] > 0,
                (features['trader_price'] - features['base_price']) / features['base_price'] * 100,
                0
            )
        
        # --- Liquidity & Volume Metrics ---
        if 'last_offer_count' in features.columns:
            features['last_offer_count'] = features['last_offer_count'].fillna(0)
            
            # Log-transform for better distribution (many items have few offers)
            features['log_offers'] = np.log1p(features['last_offer_count'])
            
            # Liquidity score (0-100) - higher is better for buying
            # Items with < 5 offers get heavily penalized
            features['liquidity_score'] = np.where(
                features['last_offer_count'] < config.VOLUME_MIN_FOR_RECOMMENDATION,
                features['last_offer_count'] * 2,  # 0-10 score for 0-5 offers
                np.minimum(
                    features['last_offer_count'] / config.VOLUME_MEDIUM_THRESHOLD * 100, 
                    config.MAX_LIQUIDITY_SCORE
                )
            )
            
            # Volume tier classification
            features['volume_tier'] = pd.cut(
                features['last_offer_count'],
                bins=[-1, config.VOLUME_MIN_FOR_RECOMMENDATION, config.VOLUME_LOW_THRESHOLD, 
                      config.VOLUME_MEDIUM_THRESHOLD, config.VOLUME_HIGH_THRESHOLD, 
                      config.VOLUME_VERY_HIGH_THRESHOLD, float('inf')],
                labels=['Unreliable', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            # Is this item reliable enough to trade?
            features['volume_reliable'] = features['last_offer_count'] >= config.VOLUME_MIN_FOR_RECOMMENDATION
            
            # Competition indicator - high offers might mean saturated market
            features['market_saturation'] = np.where(
                features['last_offer_count'] > config.VOLUME_HIGH_THRESHOLD,
                1.0,
                features['last_offer_count'] / config.VOLUME_HIGH_THRESHOLD
            )
            
            # Volume-adjusted profit - profit weighted by how easy it is to buy
            # Items with very few offers get their profit heavily discounted
            # This prevents 1-offer items from ranking high
            volume_multiplier = np.where(
                features['last_offer_count'] < config.VOLUME_MIN_FOR_RECOMMENDATION,
                0.1,  # 90% penalty for unreliable volume
                np.clip(features['last_offer_count'] / config.VOLUME_MEDIUM_THRESHOLD, 0.2, 2.0)
            )
            features['volume_adjusted_profit'] = features['profit'] * volume_multiplier
            
            # Profit per offer - how much profit relative to competition
            features['profit_per_offer'] = np.where(
                features['last_offer_count'] > 0,
                features['profit'] / np.log1p(features['last_offer_count']),
                features['profit']
            )
            
            # Buy feasibility score - combines volume with price position
            # High volume + low price position = easy to buy at good price
            if 'price_position' in features.columns:
                features['buy_feasibility'] = np.where(
                    features['last_offer_count'] < config.VOLUME_MIN_FOR_RECOMMENDATION,
                    10,  # Very low feasibility for unreliable items
                    features['liquidity_score'] * 0.6 + (1 - features['price_position']) * 100 * 0.4
                )
        
        # --- Tarkov-Specific Features ---
        # Trader accessibility (lower level = more accessible)
        if 'trader_level_required' in features.columns:
            features['trader_level_required'] = features['trader_level_required'].fillna(1)
            features['trader_accessibility'] = 5 - features['trader_level_required']
        
        # Category encoding for ML (will be used for embeddings)
        if 'category' in features.columns:
            # Create category popularity score based on average profit
            cat_profits = features.groupby('category')['profit'].mean()
            features['category_avg_profit'] = features['category'].map(cat_profits).fillna(0)
        
        return features
    
    def load_trend_data(self, hours: Optional[int] = None) -> bool:
        """
        Load historical trend data from the database for trend learning.
        
        This method fetches aggregated historical data to understand
        how each item's profit has evolved over time.
        
        Args:
            hours: Hours of history to analyze. Defaults to config.TREND_LOOKBACK_HOURS.
                   Must be positive if provided.
            
        Returns:
            True if trend data was successfully loaded, False otherwise.
        """
        if hours is None:
            hours = config.TREND_LOOKBACK_HOURS
        elif not isinstance(hours, int) or hours <= 0:
            logger.warning("Invalid hours value %s, using default", hours)
            hours = config.TREND_LOOKBACK_HOURS
            
        try:
            # Import here to avoid circular dependency
            from database import get_item_trend_data, get_profit_statistics
            
            # Get trend data
            trend_rows = get_item_trend_data(item_ids=None, hours=hours)
            
            if not trend_rows:
                logger.warning("No trend data available in database")
                return False
            
            # Convert to DataFrame
            self.trend_data = pd.DataFrame(trend_rows, columns=[
                'item_id', 'data_points', 'avg_profit', 'min_profit', 'max_profit',
                'avg_flea_price', 'avg_trader_price', 'avg_offers',
                'first_seen', 'last_seen'
            ])
            
            # Get global profit statistics
            self.profit_stats = get_profit_statistics(hours=hours)
            
            self._last_trend_update = datetime.now()
            
            total_records = self.profit_stats.get('total_records', 0) if self.profit_stats else 0
            logger.info("Loaded trend data for %d items (%d records)",
                       len(self.trend_data), total_records)
            return True
            
        except Exception as e:
            logger.error("Failed to load trend data: %s", e)
            return False
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich current data with historical trend features.
        
        Uses the loaded trend data to add features like:
        - Profit trend direction (improving/declining/stable)
        - Historical volatility
        - Consistency score (how reliably profitable)
        - Momentum indicators
        
        Args:
            df: Current market data DataFrame with item_id column.
            
        Returns:
            DataFrame with added trend-based feature columns.
        """
        if df.empty:
            return df
            
        features = df.copy()
        
        # If no trend data loaded, try to load it
        if self.trend_data is None or len(self.trend_data) == 0:
            self.load_trend_data()
        
        # If still no trend data, add default values
        if self.trend_data is None or len(self.trend_data) == 0:
            features['profit_trend'] = 0.0
            features['historical_volatility'] = 0.0
            features['consistency_score'] = 50.0
            features['trend_confidence'] = 0.0
            features['trend_direction'] = 'Unknown'
            return features
        
        # Merge trend data with current data
        trend_df = self.trend_data.copy()
        
        # Calculate trend metrics per item
        # Profit volatility (normalized std dev)
        trend_df['profit_range'] = trend_df['max_profit'] - trend_df['min_profit']
        avg_profit_abs = trend_df['avg_profit'].abs()
        trend_df['historical_volatility'] = np.where(
            avg_profit_abs > 1,
            trend_df['profit_range'] / avg_profit_abs,
            0
        )
        # Replace any inf or NaN values
        trend_df['historical_volatility'] = trend_df['historical_volatility'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate consistency score (0-100)
        # Items that are always profitable and have low volatility score higher
        min_consistent = (trend_df['min_profit'] > 0).astype(float) * 30
        low_volatility = (1 - np.clip(trend_df['historical_volatility'], 0, 1)) * 40
        has_data = np.clip(trend_df['data_points'] / config.TREND_MIN_DATA_POINTS, 0, 1) * 30
        trend_df['consistency_score'] = min_consistent + low_volatility + has_data
        
        # Trend confidence based on data points
        trend_df['trend_confidence'] = np.clip(
            trend_df['data_points'] / (config.TREND_MIN_DATA_POINTS * 2),
            0, 1
        ) * 100
        
        # Merge with current data
        features = features.merge(
            trend_df[['item_id', 'avg_profit', 'historical_volatility', 
                     'consistency_score', 'trend_confidence', 'data_points',
                     'min_profit', 'max_profit']].rename(columns={
                         'avg_profit': 'historical_avg_profit',
                         'data_points': 'historical_data_points',
                         'min_profit': 'historical_min_profit',
                         'max_profit': 'historical_max_profit'
                     }),
            on='item_id',
            how='left'
        )
        
        # Fill missing values for items without history
        features['historical_volatility'] = features['historical_volatility'].fillna(0.5)
        features['consistency_score'] = features['consistency_score'].fillna(50)
        features['trend_confidence'] = features['trend_confidence'].fillna(0)
        features['historical_data_points'] = features['historical_data_points'].fillna(0)
        features['historical_avg_profit'] = features['historical_avg_profit'].fillna(features['profit'])
        
        # Calculate profit trend (current vs historical)
        # Positive = improving, Negative = declining
        historical_avg_abs = features['historical_avg_profit'].abs()
        features['profit_trend'] = np.where(
            historical_avg_abs > 1,
            (features['profit'] - features['historical_avg_profit']) / historical_avg_abs,
            0
        )
        # Replace any inf or NaN values
        features['profit_trend'] = features['profit_trend'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Classify trend direction
        def classify_trend(trend_val: float) -> str:
            try:
                if trend_val is None or pd.isna(trend_val) or np.isinf(trend_val):
                    return 'Unknown'
                trend_float = float(trend_val)
                if trend_float > config.TREND_IMPROVEMENT_THRESHOLD:
                    return 'Improving'
                elif trend_float < -config.TREND_IMPROVEMENT_THRESHOLD:
                    return 'Declining'
                else:
                    return 'Stable'
            except (ValueError, TypeError):
                return 'Unknown'
        
        features['trend_direction'] = features['profit_trend'].apply(classify_trend)
        
        # Trend-adjusted opportunity score
        # Bonus for improving trends, penalty for declining
        features['trend_bonus'] = np.where(
            features['profit_trend'] > config.TREND_IMPROVEMENT_THRESHOLD,
            features['profit_trend'] * config.TREND_PROFIT_MOMENTUM_WEIGHT * 100,
            np.where(
                features['profit_trend'] < -config.TREND_IMPROVEMENT_THRESHOLD,
                features['profit_trend'] * config.TREND_PROFIT_MOMENTUM_WEIGHT * 100,
                0
            )
        )
        
        # Consistency bonus
        features['consistency_bonus'] = (
            (features['consistency_score'] - 50) / 50  # Normalize to -1 to 1
        ) * config.TREND_CONSISTENCY_BONUS * 100
        
        # Volatility penalty
        features['volatility_penalty'] = (
            features['historical_volatility'].clip(0, 1) 
            * config.TREND_VOLATILITY_PENALTY * 100
        )
        
        return features
    
    def get_trend_enhanced_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate an enhanced opportunity score that incorporates historical trends.
        
        This combines the base ML opportunity score with trend-based adjustments
        to favor items that are consistently profitable and improving.
        
        Args:
            df: DataFrame with market data.
            
        Returns:
            DataFrame with trend_enhanced_score column added.
        """
        if df.empty:
            return df
        
        # Get base ML score first
        features = self.calculate_opportunity_score_ml(df)
        
        # Add trend features
        features = self.calculate_trend_features(features)
        
        # Calculate trend-enhanced score
        base_score = features.get('ml_opportunity_score', 50)
        trend_bonus = features.get('trend_bonus', 0)
        consistency_bonus = features.get('consistency_bonus', 0)
        volatility_penalty = features.get('volatility_penalty', 0)
        
        # Weight based on how much historical data we have
        trend_weight = features.get('trend_confidence', 0) / 100
        
        # Blend base score with trend adjustments
        raw_enhanced = (
            base_score * (1 - trend_weight * 0.3) +  # Reduce base score weight as trend confidence grows
            (base_score + trend_bonus + consistency_bonus - volatility_penalty) * (trend_weight * 0.3)
        )
        features['trend_enhanced_score'] = np.clip(raw_enhanced, 0, 100)
        
        # Rank by enhanced score
        features['trend_rank'] = features['trend_enhanced_score'].rank(ascending=False)
        
        return features
    
    def train_on_data(self, df: pd.DataFrame, save: bool = True) -> Dict[str, Any]:
        """
        Train the model on new data and update persistent learned state.
        
        This method updates the model's learned parameters from the current
        market data. The learned state persists across database cleanups,
        allowing continuous improvement over time.
        
        Args:
            df: DataFrame with current market data.
            save: Whether to save state to disk after training.
            
        Returns:
            Dict with training statistics.
        """
        if df.empty:
            return {'status': 'no_data', 'items_processed': 0}
        
        items_processed = 0
        profitable_count = 0
        
        # Update item statistics
        for _, row in df.iterrows():
            item_id = row.get('item_id', '')
            if not item_id:
                continue
            
            profit = row.get('profit', 0)
            flea_price = row.get('flea_price', 0)
            offers = row.get('last_offer_count', 0)
            category = row.get('category', 'Unknown')
            trader = row.get('trader_name', 'Unknown')
            
            self.persistence.update_item_statistics(
                item_id=item_id,
                profit=profit,
                flea_price=flea_price,
                offers=int(offers),
                category=category,
                trader=trader
            )
            
            # Update category and trader stats
            is_profitable = profit > 0
            self.persistence.update_category_performance(category, profit, is_profitable)
            self.persistence.update_trader_reliability(trader, profit)
            
            items_processed += 1
            if is_profitable:
                profitable_count += 1
        
        # Update calibration
        if len(df) > 0:
            profit_mean = df['profit'].mean()
            profit_std = df['profit'].std() if len(df) > 1 else 1.0
            roi_mean = df['roi'].mean() if 'roi' in df.columns else 0
            roi_std = df['roi'].std() if 'roi' in df.columns and len(df) > 1 else 1.0
            
            # Handle NaN values from empty or single-row series
            if pd.isna(profit_mean) or np.isinf(profit_mean):
                profit_mean = 0
            if pd.isna(profit_std) or profit_std == 0 or np.isinf(profit_std):
                profit_std = 1.0
            if pd.isna(roi_mean) or np.isinf(roi_mean):
                roi_mean = 0
            if pd.isna(roi_std) or roi_std == 0 or np.isinf(roi_std):
                roi_std = 1.0
            
            self.persistence.update_calibration(
                profit_mean=float(profit_mean),
                profit_std=float(profit_std),
                roi_mean=float(roi_mean),
                roi_std=float(roi_std)
            )
        
        # Record training session
        avg_profit = df['profit'].mean() if len(df) > 0 else 0
        self.persistence.record_training_session(
            items_processed=items_processed,
            profitable_count=profitable_count,
            avg_profit=avg_profit
        )
        
        # Save to disk
        if save:
            self.persistence.save_state()
            self.persistence.save_history()
        
        logger.info("Trained on %d items, %d profitable", items_processed, profitable_count)
        
        return {
            'status': 'success',
            'items_processed': items_processed,
            'profitable_count': profitable_count,
            'avg_profit': avg_profit,
        }
    
    def get_learned_item_score(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get learned score adjustments for a specific item.
        
        Args:
            item_id: The item ID to look up.
            
        Returns:
            Dict with learned stats or None if item not tracked.
        """
        stats = self.persistence.get_item_learned_stats(item_id)
        if not stats:
            return None
        
        return {
            'item_id': item_id,
            'learned_profit_mean': stats['profit_mean'],
            'learned_consistency': stats['consistency_score'],
            'data_points': stats['count'],
            'first_seen': stats['first_seen'],
            'last_seen': stats['last_seen'],
        }
    
    def enrich_with_learned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich current data with persistently learned statistics.
        
        This adds columns from the learned model state that persist
        even after database cleanup.
        
        Args:
            df: Current market data DataFrame.
            
        Returns:
            DataFrame with additional learned columns.
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        features = df.copy()
        
        # Add learned profit mean
        learned_profit = []
        learned_consistency = []
        learned_data_points = []
        category_weights = []
        trader_reliability = []
        
        for _, row in features.iterrows():
            item_id = row.get('item_id', '')
            category = row.get('category', 'Unknown')
            trader = row.get('trader_name', 'Unknown')
            
            # Item-specific learned data
            item_stats = self.persistence.get_item_learned_stats(item_id)
            if item_stats and item_stats['count'] >= config.TREND_MIN_DATA_POINTS:
                learned_profit.append(item_stats['profit_mean'])
                learned_consistency.append(item_stats['consistency_score'])
                learned_data_points.append(item_stats['count'])
            else:
                learned_profit.append(None)
                learned_consistency.append(None)
                learned_data_points.append(0)
            
            # Category and trader weights
            category_weights.append(self.persistence.get_category_weight(category))
            trader_reliability.append(self.persistence.get_trader_reliability(trader))
        
        features['learned_profit_mean'] = learned_profit
        features['learned_consistency'] = learned_consistency
        features['learned_data_points'] = learned_data_points
        features['category_weight'] = category_weights
        features['trader_reliability'] = trader_reliability
        
        # Fill NaN learned values with current values (avoiding deprecation warning)
        learned_profit_mask = features['learned_profit_mean'].isna()
        features.loc[learned_profit_mask, 'learned_profit_mean'] = features.loc[learned_profit_mask, 'profit']
        consistency_mask = features['learned_consistency'].isna()
        features.loc[consistency_mask, 'learned_consistency'] = 50
        
        # Calculate learned-adjusted score
        # Items with high learned consistency get boosted
        has_learned = features['learned_data_points'] >= config.TREND_MIN_DATA_POINTS
        
        features['learned_score_adjustment'] = np.where(
            has_learned,
            (features['learned_consistency'] - 50) / 50 * 20,  # -20 to +20 adjustment
            0
        )
        
        # Profit trend vs learned (is current profit above/below historical average?)
        features['profit_vs_learned'] = np.where(
            has_learned & (features['learned_profit_mean'].abs() > 1),
            (features['profit'] - features['learned_profit_mean']) / features['learned_profit_mean'].abs(),
            0
        )
        
        return features
    
    def calculate_opportunity_score_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ML-enhanced opportunity score using gradient boosting.
        Learns optimal weights from historical data patterns.
        
        Args:
            df: DataFrame with market data including profit, prices, and item metadata.
            
        Returns:
            DataFrame with added ml_opportunity_score, opportunity_rank, and opportunity_percentile columns.
        """
        if df.empty or len(df) < config.ML_MIN_ITEMS_FOR_ANALYSIS:
            return df
        
        features = self.prepare_features(df)
        
        # Feature columns for scoring - core metrics
        score_features = [
            'profit', 'capital_efficiency', 'profit_per_slot', 'profit_density'
        ]
        
        # Add volume-related features if available
        volume_features = [
            'liquidity_score', 'volume_adjusted_profit', 'profit_per_offer', 'buy_feasibility'
        ]
        
        # Add other optional features
        other_features = [
            'price_position', 'momentum', 'trader_accessibility', 'market_saturation'
        ]
        
        for feat in volume_features + other_features:
            if feat in features.columns:
                score_features.append(feat)
        
        # Prepare feature matrix
        X = features[score_features].fillna(0).copy()
        
        # Normalize each feature to 0-1 scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adaptive weighting based on data characteristics
        # Items with high variance in a feature get less weight (unstable signal)
        feature_std = X.std()
        # Replace zeros and NaN with small value to avoid division by zero
        feature_std = feature_std.fillna(0.01).replace(0, 0.01)
        feature_stability = 1 / (feature_std + 0.01)
        stability_sum = feature_stability.sum()
        if stability_sum > 0 and not pd.isna(stability_sum) and not np.isinf(stability_sum):
            feature_stability = feature_stability / stability_sum
        else:
            # Fallback to uniform weights if sum is zero, NaN, or inf
            feature_stability = pd.Series(1.0 / len(score_features), index=feature_stability.index)
        
        # Base weights (domain knowledge) - volume gets significant weight
        base_weights = {
            'profit': 0.20,
            'capital_efficiency': 0.15,
            'profit_per_slot': 0.10,
            'profit_density': 0.08,
            'liquidity_score': config.VOLUME_WEIGHT_IN_SCORE,  # Volume is important!
            'volume_adjusted_profit': 0.10,
            'profit_per_offer': 0.05,
            'buy_feasibility': 0.05,
            'price_position': 0.04,  # Lower = better (buy low)
            'momentum': 0.03,
            'trader_accessibility': 0.03,
            'market_saturation': 0.02
        }
        
        # Combine base weights with adaptive weights
        weights = np.array([base_weights.get(f, 0.05) for f in score_features])
        # Ensure feature_stability values are valid floats
        stability_values = feature_stability.fillna(1.0 / len(score_features)).values
        weights = weights * stability_values
        weight_sum = weights.sum()
        if weight_sum > 0 and not np.isnan(weight_sum):
            weights = weights / weight_sum  # Renormalize
        else:
            weights = np.ones(len(score_features)) / len(score_features)
        
        # Calculate weighted score
        # Invert price_position (lower is better - buying near the low)
        for i, feat in enumerate(score_features):
            if feat == 'price_position':
                X_scaled[:, i] = 1 - X_scaled[:, i]
            elif feat == 'market_saturation':
                X_scaled[:, i] = 1 - X_scaled[:, i]  # Lower saturation is better
        
        # Handle potential NaN values in X_scaled
        X_scaled = np.nan_to_num(X_scaled, nan=0.5, posinf=1.0, neginf=0.0)
        
        features['ml_opportunity_score'] = np.clip((X_scaled @ weights) * 100, 0, 100)
        
        # Rank within dataset
        features['opportunity_rank'] = features['ml_opportunity_score'].rank(ascending=False)
        features['opportunity_percentile'] = features['ml_opportunity_score'].rank(pct=True) * 100
        
        return features
    
    def detect_arbitrage_anomalies(self, df: pd.DataFrame, contamination: float = config.ML_ANOMALY_CONTAMINATION) -> pd.DataFrame:
        """
        Use Isolation Forest to detect unusual pricing patterns that may indicate
        arbitrage opportunities or data errors.
        
        Anomalies can represent either exceptional opportunities (high profit
        with good volume) or risky situations (high profit with very low volume).
        The anomaly_type column helps distinguish between these cases.
        
        Args:
            df: DataFrame with prepared features for anomaly detection.
            contamination: Expected proportion of outliers (0.0-0.5). Default
                          is configured via ML_ANOMALY_CONTAMINATION.
            
        Returns:
            DataFrame with added columns:
                - is_anomaly: Boolean flag (1=anomaly, 0=normal)
                - anomaly_score: Continuous score (higher=more anomalous)
                - anomaly_type: Classification of anomaly type
                
        Note:
            Requires at least ML_MIN_ITEMS_FOR_ANOMALY items for detection.
        """
        if df.empty or len(df) < config.ML_MIN_ITEMS_FOR_ANOMALY:
            df['is_anomaly'] = False
            df['anomaly_score'] = 0
            return df
        
        features = self.prepare_features(df)
        
        # Features that indicate unusual pricing - now including volume
        anomaly_features = ['profit', 'capital_efficiency', 'profit_per_slot']
        
        # Volume can indicate anomalies - very high profit with low volume = suspicious
        if 'log_offers' in features.columns:
            anomaly_features.append('log_offers')
        if 'volume_adjusted_profit' in features.columns:
            anomaly_features.append('volume_adjusted_profit')
        if 'flea_premium' in features.columns:
            anomaly_features.append('flea_premium')
        if 'price_position' in features.columns:
            anomaly_features.append('price_position')
        
        X = features[anomaly_features].fillna(0)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=config.ML_ESTIMATORS
        )
        
        # Fit and predict
        features['anomaly_label'] = iso_forest.fit_predict(X)
        features['anomaly_score'] = -iso_forest.score_samples(X)  # Higher = more anomalous
        
        # Binary flag (1 = anomaly, 0 = normal)
        features['is_anomaly'] = (features['anomaly_label'] == -1).astype(int)
        
        # Classify anomaly type
        features['anomaly_type'] = features.apply(
            lambda row: self._classify_anomaly(row) if row['is_anomaly'] else 'Normal',
            axis=1
        )
        
        return features
    
    def _classify_anomaly(self, row: pd.Series) -> str:
        """
        Classify the type of anomaly based on feature values.
        
        Args:
            row: A pandas Series representing a single item's features.
            
        Returns:
            String label describing the anomaly type.
        """
        # Check for volume-related anomalies first
        offers = row.get('last_offer_count', 0)
        profit = row.get('profit', 0)
        
        if profit > 10000 and offers < config.VOLUME_LOW_THRESHOLD:
            return 'High Profit Low Volume (Risky)'
        elif profit > 5000 and offers > config.VOLUME_HIGH_THRESHOLD:
            return 'High Profit High Volume (Opportunity!)'
        elif row.get('profit', 0) > row.get('flea_price', 1) * 0.5:
            return 'High Profit Opportunity'
        elif row.get('capital_efficiency', 0) > 0.5:
            return 'High ROI'
        elif row.get('price_position', 0.5) < 0.1:
            return 'Price Dip'
        elif row.get('flea_premium', 0) < -30:
            return 'Undervalued'
        else:
            return 'Unusual Pattern'
    
    def cluster_items(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Cluster items into trading strategy groups using K-Means.
        
        Uses automatic optimal cluster detection via the elbow method
        to find meaningful groupings of items based on trading characteristics.
        
        Args:
            df: DataFrame with prepared features for clustering.
            n_clusters: Maximum number of clusters to create.
            
        Returns:
            DataFrame with cluster, cluster_label, and cluster_confidence columns added.
        """
        if df.empty or len(df) < n_clusters:
            df['cluster'] = 0
            df['cluster_label'] = 'Unknown'
            df['cluster_confidence'] = 0.5
            return df
        
        features = self.prepare_features(df)
        
        # Clustering features - including volume metrics
        cluster_features = ['profit', 'capital_efficiency', 'profit_per_slot']
        
        # Volume features are critical for clustering trading strategies
        if 'liquidity_score' in features.columns:
            cluster_features.append('liquidity_score')
        if 'log_offers' in features.columns:
            cluster_features.append('log_offers')
        if 'volume_adjusted_profit' in features.columns:
            cluster_features.append('volume_adjusted_profit')
        if 'momentum' in features.columns:
            cluster_features.append('momentum')
        if 'price_spread_pct' in features.columns:
            cluster_features.append('price_spread_pct')
        
        X = features[cluster_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal clusters using elbow method (simplified)
        if len(df) >= 20:
            inertias = []
            K_range = range(2, min(8, len(df) // 3))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow (simplified: largest drop)
            if len(inertias) > 1:
                drops = np.diff(inertias)
                optimal_k = list(K_range)[np.argmin(drops) + 1]
                n_clusters = min(optimal_k, n_clusters)
        
        # Final clustering - ensure n_clusters doesn't exceed sample count
        actual_n_clusters = min(n_clusters, len(df))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters and assign meaningful labels
        agg_dict = {
            'profit': 'mean',
            'capital_efficiency': 'mean',
            'profit_per_slot': 'mean'
        }
        # Add volume to cluster analysis if available
        if 'last_offer_count' in features.columns:
            agg_dict['last_offer_count'] = 'mean'
        
        cluster_stats = features.groupby('cluster').agg(agg_dict)
        
        # Sort clusters by profitability
        cluster_stats['rank'] = cluster_stats['profit'].rank(ascending=False)
        
        # Assign labels based on characteristics
        def label_cluster(cluster_id: int) -> str:
            if cluster_id not in cluster_stats.index:
                return '📊 Standard'
            stats = cluster_stats.loc[cluster_id]
            # Safely extract scalar values from potentially Series objects
            try:
                rank_val = stats['rank']
                profit_val = stats['profit']
                # Convert to numpy scalar first if needed, then to Python int/float
                if hasattr(rank_val, 'iloc'):
                    rank = int(rank_val.iloc[0])  # type: ignore[union-attr]
                elif hasattr(rank_val, 'item'):
                    rank = int(rank_val.item())  # type: ignore[union-attr]
                elif isinstance(rank_val, (int, float, np.number)):
                    rank = int(rank_val)
                else:
                    rank = 999
                    
                if hasattr(profit_val, 'iloc'):
                    profit = float(profit_val.iloc[0])  # type: ignore[union-attr]
                elif hasattr(profit_val, 'item'):
                    profit = float(profit_val.item())  # type: ignore[union-attr]
                elif isinstance(profit_val, (int, float, np.number)):
                    profit = float(profit_val)
                else:
                    profit = 0.0
            except (ValueError, AttributeError, TypeError, IndexError):
                rank = 999
                profit = 0.0
            
            if rank == 1:
                return '🥇 Elite Opportunities'
            elif rank == 2:
                return '🥈 High Value'
            elif rank == 3:
                return '🥉 Solid Picks'
            elif profit < 0:
                return '⚠️ Avoid (Negative)'
            else:
                return '📊 Standard'
        
        features['cluster_label'] = features['cluster'].apply(label_cluster)
        
        # Add cluster confidence (distance to centroid)
        distances = kmeans.transform(X_scaled)
        # Handle edge cases with distances
        min_distances = distances.min(axis=1)
        features['cluster_confidence'] = 1 / (1 + np.nan_to_num(min_distances, nan=1.0, posinf=1.0, neginf=0.0))
        
        return features
    
    def find_similar_items(self, df: pd.DataFrame, item_id: str, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Find items with similar trading characteristics to a given item.
        
        Uses k-nearest neighbors on normalized feature space to identify
        items with comparable profit potential and characteristics.
        
        Args:
            df: DataFrame containing all items for comparison.
            item_id: The ID of the target item to find similar items for.
            n_neighbors: Number of similar items to return. Must be positive.
            
        Returns:
            DataFrame of similar items with similarity_score column added.
            Returns empty DataFrame if item_id not found or n_neighbors invalid.
        """
        # Validate n_neighbors
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            logger.warning("find_similar_items called with invalid n_neighbors: %s", n_neighbors)
            return pd.DataFrame()
        
        if df.empty or item_id not in df['item_id'].values:
            return pd.DataFrame()
        
        features = self.prepare_features(df)
        
        # Similarity features
        sim_features = ['profit', 'capital_efficiency', 'profit_per_slot', 'slots', 'weight']
        
        if 'category' in features.columns:
            # One-hot encode category
            cat_dummies = pd.get_dummies(features['category'], prefix='cat')
            features = pd.concat([features, cat_dummies], axis=1)
            sim_features.extend(cat_dummies.columns.tolist())
        
        X = features[sim_features].fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Handle NaN/inf in scaled features
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Find neighbors - ensure at least 1 neighbor and don't exceed dataset size
        actual_neighbors = max(1, min(n_neighbors + 1, len(df)))
        nn = NearestNeighbors(n_neighbors=actual_neighbors)
        nn.fit(X_scaled)
        
        # Get index of target item
        target_idx = features[features['item_id'] == item_id].index[0]
        target_scaled = X_scaled[features.index.get_loc(target_idx)].reshape(1, -1)
        
        distances, indices = nn.kneighbors(target_scaled)
        
        # Get similar items (exclude the item itself)
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]
        
        result = features.iloc[similar_indices].copy()
        # Handle edge cases in similarity score calculation
        result['similarity_score'] = 1 / (1 + np.nan_to_num(similar_distances, nan=1.0, posinf=1.0, neginf=0.0))
        
        return result
    
    def predict_profit_trend(self, history_df: pd.DataFrame, 
                            periods_ahead: int = 12) -> Dict[str, Any]:
        """
        Predict future profit trends using time-series analysis.
        
        Uses Ridge regression for linear projection with moving average
        smoothing to detect trends and forecast future values.
        
        Args:
            history_df: DataFrame with timestamp and profit columns.
            periods_ahead: Number of future periods to predict. Must be positive.
            
        Returns:
            Dict containing:
                - predictions: List of predicted profit values
                - confidence: 0-100 confidence score based on model fit
                - trend: 'increasing', 'decreasing', or 'stable'
                - recent_avg: Average profit of last 5 periods
                - volatility: Normalized standard deviation of profits
        """
        # Validate periods_ahead
        if not isinstance(periods_ahead, int) or periods_ahead <= 0:
            logger.warning("predict_profit_trend called with invalid periods_ahead: %s, using default 12", periods_ahead)
            periods_ahead = 12
        
        if history_df.empty or len(history_df) < config.ML_MIN_ITEMS_FOR_ANALYSIS:
            return {'predictions': [], 'confidence': 0, 'trend': 'unknown', 'recent_avg': 0, 'volatility': 0}
        
        # Ensure sorted by timestamp
        df = history_df.sort_values('timestamp').copy()
        
        # Check for required columns
        if 'profit' not in df.columns:
            logger.warning("predict_profit_trend called without 'profit' column")
            return {'predictions': [], 'confidence': 0, 'trend': 'unknown', 'recent_avg': 0, 'volatility': 0}
        
        # Simple moving averages
        df['sma_short'] = df['profit'].rolling(window=3, min_periods=1).mean()
        df['sma_long'] = df['profit'].rolling(window=6, min_periods=1).mean()
        
        # Trend detection - handle NaN from diff()
        profit_diff = df['profit'].tail(5).diff().dropna()
        recent_trend = profit_diff.mean() if len(profit_diff) > 0 else 0
        if pd.isna(recent_trend):
            recent_trend = 0
        
        if recent_trend > 0:
            trend = 'increasing'
        elif recent_trend < 0:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Simple linear projection
        X = np.arange(len(df)).reshape(-1, 1)
        y = np.array(df['profit'].values, dtype=np.float64)  # Ensure numpy array type with explicit dtype
        
        # Replace any NaN/inf values in y
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(df), len(df) + periods_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Confidence based on R² and stability
        train_score = model.score(X, y)
        # Clamp train_score to valid range
        train_score = max(0.0, min(1.0, train_score)) if not np.isnan(train_score) else 0.0
        profit_mean = df['profit'].mean()
        profit_std = df['profit'].std()
        # Handle NaN from std() on single-element series
        if pd.isna(profit_std) or profit_std == 0:
            profit_std = 1.0
        if pd.isna(profit_mean):
            profit_mean = 0.0
        # Avoid division by zero when mean is -1 or close to it
        volatility = float(profit_std / max(abs(profit_mean) + 1, 1))
        confidence = max(0.0, min(100.0, float(train_score) * 100 * (1 - min(volatility, 1.0))))
        
        return {
            'predictions': predictions.tolist(),
            'confidence': confidence,
            'trend': trend,
            'recent_avg': float(df['profit'].tail(5).mean()),
            'volatility': volatility
        }
    
    def calculate_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive risk score for each item.
        
        Risk is computed from multiple factors including price volatility,
        liquidity, momentum, profit margins, and trader accessibility.
        Higher risk indicates more volatile/uncertain profit potential.
        
        Args:
            df: DataFrame with prepared features for risk calculation.
            
        Returns:
            DataFrame with risk_score (0-100) and risk_level ('Low', 'Medium', 'High') columns.
        """
        if df.empty:
            df['risk_score'] = 50
            df['risk_level'] = 'Medium'
            return df
        
        features = self.prepare_features(df)
        
        risk_components = []
        
        # Volatility risk (price spread)
        if 'price_spread_pct' in features.columns:
            vol_risk = features['price_spread_pct'].fillna(0) / 50  # Normalize to ~0-1
            risk_components.append(('volatility', vol_risk.clip(0, 1), 0.25))
        
        # Liquidity risk (low offers = harder to buy)
        if 'last_offer_count' in features.columns:
            liq_risk = 1 - (features['last_offer_count'].fillna(0) / 100).clip(0, 1)
            risk_components.append(('liquidity', liq_risk, 0.20))
        
        # Momentum risk (rapidly changing prices)
        if 'momentum_abs' in features.columns:
            mom_risk = (features['momentum_abs'].fillna(0) / 30).clip(0, 1)
            risk_components.append(('momentum', mom_risk, 0.20))
        
        # Margin risk (thin margins = higher risk of loss)
        if 'profit_margin' in features.columns:
            margin_risk = 1 - (features['profit_margin'].fillna(0) / 30).clip(0, 1)
            risk_components.append(('margin', margin_risk, 0.20))
        
        # Accessibility risk (higher trader level = fewer can access)
        if 'trader_level_required' in features.columns:
            acc_risk = (features['trader_level_required'].fillna(1) - 1) / 3
            risk_components.append(('accessibility', acc_risk, 0.15))
        
        # Calculate weighted risk score
        if risk_components:
            total_weight = sum(w for _, _, w in risk_components)
            if total_weight > 0:
                features['risk_score'] = sum(
                    score * weight / total_weight 
                    for _, score, weight in risk_components
                ) * 100
            else:
                features['risk_score'] = 50
        else:
            features['risk_score'] = 50
        
        # Classify risk level
        features['risk_level'] = pd.cut(
            features['risk_score'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        return features
    
    def generate_trading_recommendations(self, df: pd.DataFrame, 
                                         player_level: int = 15,
                                         capital: int = 1000000,
                                         risk_tolerance: str = 'medium',
                                         use_trends: bool = True,
                                         use_learned: bool = True) -> pd.DataFrame:
        """
        Generate personalized trading recommendations based on player profile.
        
        Applies all analysis methods and filters results based on player's
        level, available capital, and risk tolerance preferences. When trend
        learning is enabled, recommendations improve over time as more
        historical data is collected. Persistent learned data survives
        database cleanups.
        
        Args:
            df: DataFrame with market data for analysis.
            player_level: Player's current level (affects flea market access at 15).
            capital: Available roubles for trading.
            risk_tolerance: 'low', 'medium', or 'high' risk preference.
            use_trends: Whether to use historical trend learning (default True).
            use_learned: Whether to use persistently learned data (default True).
            
        Returns:
            DataFrame of recommended items sorted by recommendation score,
            with columns for max_units, potential_profit, rec_score, and rec_tier.
        """
        if df.empty:
            return df
        
        # Enrich with persistently learned data
        if use_learned:
            features = self.enrich_with_learned_data(df)
        else:
            features = df.copy()
        
        # Apply all analyses with optional trend learning
        if use_trends:
            # Use trend-enhanced scoring
            features = self.get_trend_enhanced_score(features)
        else:
            features = self.calculate_opportunity_score_ml(features)
            
        features = self.detect_arbitrage_anomalies(features)
        features = self.calculate_risk_score(features)
        features = self.cluster_items(features)
        
        # Filter by player level (flea market access based on Patch 1.0 restrictions)
        # Uses flea_level_required column if available, otherwise defaults to accessible
        if player_level < config.FLEA_MARKET_UNLOCK_LEVEL:
            # Player below flea market unlock level - nothing accessible
            features['accessible'] = False
        else:
            # Check both trader level and flea market level requirements
            trader_accessible = features.get('trader_level_required', 1) <= 4
            if 'flea_level_required' in features.columns:
                flea_accessible = features['flea_level_required'] <= player_level
                features['accessible'] = trader_accessible & flea_accessible
            else:
                features['accessible'] = trader_accessible
        
        # Filter by risk tolerance
        risk_thresholds = {'low': 40, 'medium': 70, 'high': 100}
        max_risk = risk_thresholds.get(risk_tolerance, 70)
        features['within_risk'] = features['risk_score'] <= max_risk
        
        # Calculate how many units can be bought with capital
        features['max_units'] = (capital // features['flea_price']).clip(lower=0)
        features['potential_profit'] = features['max_units'] * features['profit']
        
        # Final recommendation score - now uses trend-enhanced score and learned data
        base_score = features.get('trend_enhanced_score', features.get('ml_opportunity_score', 50))
        
        # Add trend bonuses to final score
        trend_direction_bonus = np.where(
            features.get('trend_direction', 'Unknown') == 'Improving', 5,
            np.where(features.get('trend_direction', 'Unknown') == 'Declining', -5, 0)
        )
        
        consistency_factor = features.get('consistency_score', 50) / 100  # 0-1 scale
        
        # Learned score adjustment from persistent data
        learned_adjustment = features.get('learned_score_adjustment', 0)
        
        # Category and trader weights from learned data
        cat_weight = features.get('category_weight', 1.0)
        trader_rel = features.get('trader_reliability', 0.5)
        
        features['rec_score'] = np.clip(
            base_score * 0.30 +
            (100 - features['risk_score']) * 0.20 +
            features['cluster_confidence'] * 100 * 0.08 +
            features.get('liquidity_score', 50) * 0.12 +
            consistency_factor * 12 +  # Up to 12 points for consistency
            trend_direction_bonus +  # ±5 points for trend
            learned_adjustment +  # Up to ±20 points from learned data
            (cat_weight - 1) * 10 +  # ±5 points from category performance
            (trader_rel - 0.5) * 10,  # ±5 points from trader reliability
            0, 100
        )
        
        # Filter to accessible items within risk tolerance AND sufficient volume
        # This filters out unreliable 1-5 offer items
        if 'volume_reliable' in features.columns:
            has_volume = features['volume_reliable'].astype(bool)
        else:
            has_volume = pd.Series([True] * len(features), index=features.index)
        
        recommended = features[
            features['accessible'] & 
            features['within_risk'] &
            (features['profit'] > 0) &
            has_volume
        ].copy()
        
        # Add recommendation tier with enhanced thresholds
        recommended['rec_tier'] = pd.cut(
            recommended['rec_score'],
            bins=[0, 45, 65, 80, 100],
            labels=['Consider', 'Good', 'Great', 'Excellent']
        )
        
        # Add trend indicator for UI
        if 'trend_direction' in recommended.columns:
            recommended['trend_indicator'] = recommended['trend_direction'].map({
                'Improving': '📈',
                'Declining': '📉', 
                'Stable': '➡️',
                'Unknown': '❓'
            }).fillna('❓')
        
        # Add learned data indicator
        recommended['has_learned_data'] = recommended.get('learned_data_points', 0) >= config.TREND_MIN_DATA_POINTS
        
        return recommended.sort_values('rec_score', ascending=False)
    
    def get_trend_learning_status(self) -> Dict[str, Any]:
        """
        Get the current status of trend learning.
        
        Returns information about how much historical data is available
        and how it's being used to improve recommendations.
        
        Returns:
            Dict with trend learning status including:
                - enabled: Whether trend learning is active
                - items_with_history: Number of items with trend data
                - total_data_points: Total historical records
                - last_update: When trend data was last loaded
                - avg_data_points: Average data points per item
                - learning_quality: Overall quality score (0-100)
        """
        status = {
            'enabled': self.trend_data is not None and len(self.trend_data) > 0,
            'items_with_history': 0,
            'total_data_points': 0,
            'last_update': None,
            'avg_data_points': 0.0,
            'learning_quality': 0,
            'profit_stats': None
        }
        
        if self.trend_data is not None and len(self.trend_data) > 0:
            status['items_with_history'] = len(self.trend_data)
            status['total_data_points'] = int(self.trend_data['data_points'].sum())
            status['avg_data_points'] = float(self.trend_data['data_points'].mean())
            status['last_update'] = self._last_trend_update
            
            # Learning quality based on data coverage
            # More data points = better learning
            min_good_data = config.TREND_MIN_DATA_POINTS * 2
            items_with_good_data = (self.trend_data['data_points'] >= min_good_data).sum()
            status['learning_quality'] = min(100, int(
                (items_with_good_data / max(len(self.trend_data), 1)) * 100
            ))
        
        if self.profit_stats:
            status['profit_stats'] = self.profit_stats
        
        return status
    
    def get_item_trend_summary(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trend summary for a specific item.
        
        Args:
            item_id: The item ID to get trend data for.
            
        Returns:
            Dict with item trend info or None if not found.
        """
        if self.trend_data is None or len(self.trend_data) == 0:
            return None
        
        item_trend = self.trend_data[self.trend_data['item_id'] == item_id]
        if item_trend.empty:
            return None
        
        row = item_trend.iloc[0]
        return {
            'item_id': item_id,
            'data_points': int(row['data_points']),
            'avg_profit': float(row['avg_profit']),
            'min_profit': float(row['min_profit']),
            'max_profit': float(row['max_profit']),
            'profit_range': float(row['max_profit'] - row['min_profit']),
            'avg_offers': float(row.get('avg_offers', 0)),
            'first_seen': row.get('first_seen'),
            'last_seen': row.get('last_seen')
        }
    
    def get_persistent_learning_status(self) -> Dict[str, Any]:
        """
        Get the status of persistent model learning.
        
        Returns comprehensive information about the persistently learned
        model state that survives database cleanups.
        
        Returns:
            Dict with persistent learning status and metrics.
        """
        return self.persistence.get_learning_progress()
    
    def get_top_learned_items(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get top items by learned profitability.
        
        Args:
            n: Number of top items to return.
            
        Returns:
            List of dicts with item trend information.
        """
        return self.persistence.get_item_trends(top_n=n)
    
    def get_category_performance(self) -> List[Dict[str, Any]]:
        """
        Get learned category performance data.
        
        Returns:
            List of dicts with category performance metrics.
        """
        return self.persistence.get_category_trends()
    
    def save_model(self) -> bool:
        """
        Save the current model state to disk.
        
        Returns:
            True if save was successful.
        """
        success1 = self.persistence.save_state()
        success2 = self.persistence.save_history()
        return success1 and success2


# Singleton instance with thread-safe initialization
import threading
_ml_engine: Optional[TarkovMLEngine] = None
_ml_engine_lock: threading.Lock = threading.Lock()


def get_ml_engine() -> TarkovMLEngine:
    """Get or create the ML engine singleton (thread-safe).
    
    Returns:
        The singleton TarkovMLEngine instance.
        
    Raises:
        RuntimeError: If ML engine initialization fails.
    """
    global _ml_engine
    if _ml_engine is None:
        with _ml_engine_lock:
            # Double-check locking pattern
            if _ml_engine is None:
                try:
                    _ml_engine = TarkovMLEngine()
                except Exception as e:
                    logger.error("Failed to initialize ML engine: %s", e)
                    raise RuntimeError(f"ML engine initialization failed: {e}") from e
    return _ml_engine
