"""
Advanced Machine Learning Engine for Tarkov Trader Profit Analysis.

This module provides sophisticated ML algorithms for:
- Price prediction and trend forecasting
- Anomaly detection for arbitrage opportunities  
- Item clustering and similarity analysis
- Risk assessment and portfolio optimization
- Time-series analysis for optimal trading windows
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import logging
from datetime import datetime, timedelta

import config

__all__ = ['TarkovMLEngine', 'get_ml_engine']

# Configure logging
logger = logging.getLogger(__name__)


class TarkovMLEngine:
    """
    Advanced ML engine for Tarkov market analysis.
    
    Combines multiple ML techniques for comprehensive trading insights including:
    - Feature engineering for game economy metrics
    - Opportunity scoring with adaptive weights
    - Anomaly detection for arbitrage opportunities
    - Item clustering for strategy grouping
    - Profit trend prediction
    - Risk assessment
    
    Attributes:
        scaler: RobustScaler for handling outliers common in game economies.
        price_predictor: Optional predictor model (reserved for future use).
        anomaly_detector: Optional anomaly detection model.
        item_clusterer: Optional clustering model.
    """
    
    def __init__(self) -> None:
        """Initialize the ML engine with default scalers and empty model slots."""
        self.scaler = RobustScaler()
        self.price_predictor: Optional[Any] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.item_clusterer: Optional[KMeans] = None
        self._is_fitted: bool = False
    
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
        
        # --- Liquidity Metrics ---
        if 'last_offer_count' in features.columns:
            features['last_offer_count'] = features['last_offer_count'].fillna(0)
            # Log-transform for better distribution (many items have few offers)
            features['log_offers'] = np.log1p(features['last_offer_count'])
            
            # Liquidity score (0-100)
            features['liquidity_score'] = np.minimum(features['last_offer_count'] / 50 * 100, 100)
            
            # Competition indicator - high offers might mean saturated market
            features['market_saturation'] = np.where(
                features['last_offer_count'] > 100,
                1,
                features['last_offer_count'] / 100
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
        
        # Feature columns for scoring
        score_features = [
            'profit', 'capital_efficiency', 'profit_per_slot', 'profit_density'
        ]
        
        # Add optional features if available
        optional_features = [
            'liquidity_score', 'price_position', 'momentum', 
            'trader_accessibility', 'market_saturation'
        ]
        
        for feat in optional_features:
            if feat in features.columns:
                score_features.append(feat)
        
        # Prepare feature matrix
        X = features[score_features].fillna(0).copy()
        
        # Normalize each feature to 0-1 scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adaptive weighting based on data characteristics
        # Items with high variance in a feature get less weight (unstable signal)
        feature_stability = 1 / (X.std() + 0.01)
        feature_stability = feature_stability / feature_stability.sum()
        
        # Base weights (domain knowledge)
        base_weights = {
            'profit': 0.25,
            'capital_efficiency': 0.20,
            'profit_per_slot': 0.15,
            'profit_density': 0.10,
            'liquidity_score': 0.10,
            'price_position': 0.05,  # Lower = better (buy low)
            'momentum': 0.05,
            'trader_accessibility': 0.05,
            'market_saturation': 0.05
        }
        
        # Combine base weights with adaptive weights
        weights = np.array([base_weights.get(f, 0.05) for f in score_features])
        weights = weights * feature_stability.values
        weights = weights / weights.sum()  # Renormalize
        
        # Calculate weighted score
        # Invert price_position (lower is better - buying near the low)
        for i, feat in enumerate(score_features):
            if feat == 'price_position':
                X_scaled[:, i] = 1 - X_scaled[:, i]
            elif feat == 'market_saturation':
                X_scaled[:, i] = 1 - X_scaled[:, i]  # Lower saturation is better
        
        features['ml_opportunity_score'] = (X_scaled @ weights) * 100
        
        # Rank within dataset
        features['opportunity_rank'] = features['ml_opportunity_score'].rank(ascending=False)
        features['opportunity_percentile'] = features['ml_opportunity_score'].rank(pct=True) * 100
        
        return features
    
    def detect_arbitrage_anomalies(self, df: pd.DataFrame, contamination: float = config.ML_ANOMALY_CONTAMINATION) -> pd.DataFrame:
        """
        Use Isolation Forest to detect unusual pricing patterns that may indicate
        arbitrage opportunities or data errors.
        
        Args:
            df: DataFrame with prepared features for anomaly detection.
            contamination: Expected proportion of outliers in the dataset.
            
        Returns:
            DataFrame with is_anomaly, anomaly_score, and anomaly_type columns added.
        """
        if df.empty or len(df) < config.ML_MIN_ITEMS_FOR_ANOMALY:
            df['is_anomaly'] = False
            df['anomaly_score'] = 0
            return df
        
        features = self.prepare_features(df)
        
        # Features that indicate unusual pricing
        anomaly_features = ['profit', 'capital_efficiency', 'profit_per_slot']
        
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
        if row.get('profit', 0) > row.get('flea_price', 1) * 0.5:
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
        
        # Clustering features
        cluster_features = ['profit', 'capital_efficiency', 'profit_per_slot']
        
        if 'liquidity_score' in features.columns:
            cluster_features.append('liquidity_score')
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
        
        # Final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters and assign meaningful labels
        cluster_stats = features.groupby('cluster').agg({
            'profit': 'mean',
            'capital_efficiency': 'mean',
            'profit_per_slot': 'mean'
        })
        
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
                else:
                    rank = int(rank_val)  # type: ignore[arg-type]
                    
                if hasattr(profit_val, 'iloc'):
                    profit = float(profit_val.iloc[0])  # type: ignore[union-attr]
                elif hasattr(profit_val, 'item'):
                    profit = float(profit_val.item())  # type: ignore[union-attr]
                else:
                    profit = float(profit_val)  # type: ignore[arg-type]
            except (ValueError, AttributeError, TypeError):
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
        features['cluster_confidence'] = 1 / (1 + distances.min(axis=1))
        
        return features
    
    def find_similar_items(self, df: pd.DataFrame, item_id: str, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Find items with similar trading characteristics to a given item.
        
        Uses k-nearest neighbors on normalized feature space to identify
        items with comparable profit potential and characteristics.
        
        Args:
            df: DataFrame containing all items for comparison.
            item_id: The ID of the target item to find similar items for.
            n_neighbors: Number of similar items to return.
            
        Returns:
            DataFrame of similar items with similarity_score column added.
            Returns empty DataFrame if item_id not found.
        """
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
        
        # Find neighbors
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(df)))
        nn.fit(X_scaled)
        
        # Get index of target item
        target_idx = features[features['item_id'] == item_id].index[0]
        target_scaled = X_scaled[features.index.get_loc(target_idx)].reshape(1, -1)
        
        distances, indices = nn.kneighbors(target_scaled)
        
        # Get similar items (exclude the item itself)
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]
        
        result = features.iloc[similar_indices].copy()
        result['similarity_score'] = 1 / (1 + similar_distances)
        
        return result
    
    def predict_profit_trend(self, history_df: pd.DataFrame, 
                            periods_ahead: int = 12) -> Dict[str, Any]:
        """
        Predict future profit trends using time-series analysis.
        
        Uses Ridge regression for linear projection with moving average
        smoothing to detect trends and forecast future values.
        
        Args:
            history_df: DataFrame with timestamp and profit columns.
            periods_ahead: Number of future periods to predict.
            
        Returns:
            Dict containing:
                - predictions: List of predicted profit values
                - confidence: 0-100 confidence score based on model fit
                - trend: 'increasing', 'decreasing', or 'stable'
                - recent_avg: Average profit of last 5 periods
                - volatility: Normalized standard deviation of profits
        """
        if history_df.empty or len(history_df) < config.ML_MIN_ITEMS_FOR_ANALYSIS:
            return {'predictions': [], 'confidence': 0, 'trend': 'unknown', 'recent_avg': 0, 'volatility': 0}
        
        # Ensure sorted by timestamp
        df = history_df.sort_values('timestamp').copy()
        
        # Simple moving averages
        df['sma_short'] = df['profit'].rolling(window=3, min_periods=1).mean()
        df['sma_long'] = df['profit'].rolling(window=6, min_periods=1).mean()
        
        # Trend detection
        recent_trend = df['profit'].tail(5).diff().mean()
        
        if recent_trend > 0:
            trend = 'increasing'
        elif recent_trend < 0:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Simple linear projection
        X = np.arange(len(df)).reshape(-1, 1)
        y = np.array(df['profit'].values)  # Ensure numpy array type
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(df), len(df) + periods_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Confidence based on R² and stability
        train_score = model.score(X, y)
        profit_mean = df['profit'].mean()
        profit_std = df['profit'].std()
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
                                         risk_tolerance: str = 'medium') -> pd.DataFrame:
        """
        Generate personalized trading recommendations based on player profile.
        
        Applies all analysis methods and filters results based on player's
        level, available capital, and risk tolerance preferences.
        
        Args:
            df: DataFrame with market data for analysis.
            player_level: Player's current level (affects flea market access at 15).
            capital: Available roubles for trading.
            risk_tolerance: 'low', 'medium', or 'high' risk preference.
            
        Returns:
            DataFrame of recommended items sorted by recommendation score,
            with columns for max_units, potential_profit, rec_score, and rec_tier.
        """
        if df.empty:
            return df
        
        # Apply all analyses
        features = self.calculate_opportunity_score_ml(df)
        features = self.detect_arbitrage_anomalies(features)
        features = self.calculate_risk_score(features)
        features = self.cluster_items(features)
        
        # Filter by player level (flea market access)
        if player_level < 15:
            features['accessible'] = False
        else:
            features['accessible'] = features.get('trader_level_required', 1) <= 4
        
        # Filter by risk tolerance
        risk_thresholds = {'low': 40, 'medium': 70, 'high': 100}
        max_risk = risk_thresholds.get(risk_tolerance, 70)
        features['within_risk'] = features['risk_score'] <= max_risk
        
        # Calculate how many units can be bought with capital
        features['max_units'] = (capital // features['flea_price']).clip(lower=0)
        features['potential_profit'] = features['max_units'] * features['profit']
        
        # Final recommendation score
        features['rec_score'] = (
            features['ml_opportunity_score'] * 0.4 +
            (100 - features['risk_score']) * 0.3 +
            features['cluster_confidence'] * 100 * 0.15 +
            features.get('liquidity_score', 50) * 0.15
        )
        
        # Filter to accessible items within risk tolerance
        recommended = features[
            features['accessible'] & 
            features['within_risk'] &
            (features['profit'] > 0)
        ].copy()
        
        # Add recommendation tier
        recommended['rec_tier'] = pd.cut(
            recommended['rec_score'],
            bins=[0, 50, 70, 85, 100],
            labels=['Consider', 'Good', 'Great', 'Excellent']
        )
        
        return recommended.sort_values('rec_score', ascending=False)


# Singleton instance
_ml_engine = None

def get_ml_engine() -> TarkovMLEngine:
    """Get or create the ML engine singleton."""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = TarkovMLEngine()
    return _ml_engine
