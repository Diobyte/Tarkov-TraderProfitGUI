"""
Tests for the ML Engine module.
"""

import pytest
import pandas as pd
import numpy as np
from ml_engine import TarkovMLEngine, get_ml_engine


class TestMLEngine:
    """Tests for the TarkovMLEngine class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5'],
            'name': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E'],
            'flea_price': [10000, 20000, 15000, 50000, 8000],
            'trader_price': [12000, 18000, 20000, 45000, 15000],
            'profit': [2000, -2000, 5000, -5000, 7000],
            'width': [1, 2, 1, 2, 1],
            'height': [2, 2, 1, 3, 1],
            'weight': [0.5, 1.0, 0.3, 2.0, 0.1],
            'category': ['Weapons', 'Gear', 'Meds', 'Weapons', 'Meds'],
            'avg_24h_price': [11000, 21000, 14000, 55000, 9000],
            'high_24h_price': [12000, 22000, 16000, 60000, 10000],
            'low_24h_price': [9000, 19000, 13000, 48000, 7000],
            'base_price': [8000, 15000, 12000, 40000, 6000],
            'last_offer_count': [50, 10, 100, 5, 200],
            'trader_level_required': [1, 2, 1, 4, 1],
            'change_last_48h': [5.0, -10.0, 2.0, -15.0, 8.0]
        })
    
    @pytest.fixture
    def history_df(self):
        """Create a sample history DataFrame for trend prediction testing."""
        timestamps = pd.date_range(start='2024-01-01', periods=20, freq='5min')
        profits = [1000 + i * 100 + np.random.randint(-50, 50) for i in range(20)]
        return pd.DataFrame({
            'timestamp': timestamps,
            'profit': profits,
            'flea_price': [10000 + i * 50 for i in range(20)],
            'trader_price': [11000 + i * 50 for i in range(20)]
        })
    
    @pytest.fixture
    def ml_engine(self):
        """Create a fresh ML engine instance."""
        return TarkovMLEngine()
    
    def test_prepare_features(self, ml_engine, sample_df):
        """Test that prepare_features adds all expected columns."""
        result = ml_engine.prepare_features(sample_df)
        
        # Check core financial metrics
        assert 'profit_margin' in result.columns
        assert 'capital_efficiency' in result.columns
        
        # Check slot efficiency metrics
        assert 'slots' in result.columns
        assert 'profit_per_slot' in result.columns
        assert 'density' in result.columns
        
        # Check price dynamics
        assert 'price_spread' in result.columns
        assert 'price_position' in result.columns
        
        # Check liquidity metrics
        assert 'log_offers' in result.columns
        assert 'liquidity_score' in result.columns
    
    def test_prepare_features_empty_df(self, ml_engine):
        """Test that prepare_features handles empty DataFrame."""
        empty_df = pd.DataFrame()
        result = ml_engine.prepare_features(empty_df)
        assert result.empty
    
    def test_calculate_opportunity_score_ml(self, ml_engine, sample_df):
        """Test ML opportunity score calculation."""
        # Need at least 10 items for opportunity scoring
        large_df = pd.concat([sample_df] * 3, ignore_index=True)
        large_df['item_id'] = [f'item{i}' for i in range(len(large_df))]
        
        result = ml_engine.calculate_opportunity_score_ml(large_df)
        
        assert 'ml_opportunity_score' in result.columns
        assert 'opportunity_rank' in result.columns
        assert 'opportunity_percentile' in result.columns
        
        # Scores should be 0-100
        assert result['ml_opportunity_score'].min() >= 0
        assert result['ml_opportunity_score'].max() <= 100
    
    def test_calculate_opportunity_score_small_df(self, ml_engine):
        """Test that opportunity score handles small datasets."""
        small_df = pd.DataFrame({
            'item_id': ['item1'],
            'flea_price': [10000],
            'trader_price': [12000],
            'profit': [2000]
        })
        result = ml_engine.calculate_opportunity_score_ml(small_df)
        # Should return original df with small datasets
        assert len(result) == 1
    
    def test_detect_arbitrage_anomalies(self, ml_engine, sample_df):
        """Test anomaly detection."""
        # Need at least 20 items for anomaly detection
        large_df = pd.concat([sample_df] * 5, ignore_index=True)
        large_df['item_id'] = [f'item{i}' for i in range(len(large_df))]
        
        result = ml_engine.detect_arbitrage_anomalies(large_df)
        
        assert 'is_anomaly' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'anomaly_type' in result.columns
    
    def test_detect_anomalies_small_df(self, ml_engine, sample_df):
        """Test that anomaly detection handles small datasets gracefully."""
        result = ml_engine.detect_arbitrage_anomalies(sample_df)
        
        # Should add columns but mark nothing as anomaly
        assert 'is_anomaly' in result.columns
        assert result['is_anomaly'].sum() == 0  # No anomalies for small dataset
    
    def test_cluster_items(self, ml_engine, sample_df):
        """Test item clustering."""
        result = ml_engine.cluster_items(sample_df, n_clusters=3)
        
        assert 'cluster' in result.columns
        assert 'cluster_label' in result.columns
        assert 'cluster_confidence' in result.columns
        
        # Should have expected number of clusters or fewer
        assert result['cluster'].nunique() <= 3
    
    def test_cluster_items_small_df(self, ml_engine):
        """Test clustering with very small dataset."""
        small_df = pd.DataFrame({
            'item_id': ['item1', 'item2'],
            'profit': [1000, 2000],
            'flea_price': [10000, 20000]
        })
        result = ml_engine.cluster_items(small_df, n_clusters=5)
        
        # Should still work with fewer clusters
        assert 'cluster' in result.columns
    
    def test_find_similar_items(self, ml_engine, sample_df):
        """Test finding similar items."""
        result = ml_engine.find_similar_items(sample_df, 'item1', n_neighbors=3)
        
        # Should return up to 3 similar items (excluding the query item)
        assert len(result) <= 3
        
        if not result.empty:
            assert 'similarity_score' in result.columns
            # Query item should not be in results
            assert 'item1' not in result['item_id'].values
    
    def test_find_similar_items_not_found(self, ml_engine, sample_df):
        """Test finding similar items when query item doesn't exist."""
        result = ml_engine.find_similar_items(sample_df, 'nonexistent_item')
        assert result.empty
    
    def test_predict_profit_trend(self, ml_engine, history_df):
        """Test profit trend prediction."""
        result = ml_engine.predict_profit_trend(history_df, periods_ahead=6)
        
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'trend' in result
        assert 'recent_avg' in result
        assert 'volatility' in result
        
        # Should have 6 predictions
        assert len(result['predictions']) == 6
        
        # Confidence should be 0-100
        assert 0 <= result['confidence'] <= 100
        
        # Trend should be one of expected values
        assert result['trend'] in ['increasing', 'decreasing', 'stable']
    
    def test_predict_profit_trend_small_df(self, ml_engine):
        """Test trend prediction with insufficient data."""
        small_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='5min'),
            'profit': [1000, 1100, 1200, 1300, 1400]
        })
        result = ml_engine.predict_profit_trend(small_df)
        
        # Should return unknown trend for small datasets
        assert result['trend'] == 'unknown'
        assert result['confidence'] == 0
        # Check all expected keys are present
        assert 'predictions' in result
        assert 'recent_avg' in result
        assert 'volatility' in result
        assert len(result['predictions']) == 0
    
    def test_calculate_risk_score(self, ml_engine, sample_df):
        """Test risk score calculation."""
        result = ml_engine.calculate_risk_score(sample_df)
        
        assert 'risk_score' in result.columns
        assert 'risk_level' in result.columns
        
        # Risk scores should be 0-100
        assert result['risk_score'].min() >= 0
        assert result['risk_score'].max() <= 100
        
        # Risk levels should be valid categories
        valid_levels = ['Low', 'Medium', 'High']
        for level in result['risk_level'].dropna():
            assert level in valid_levels
    
    def test_generate_trading_recommendations(self, ml_engine, sample_df):
        """Test generating trading recommendations."""
        # Need at least 10 items for ML scoring
        large_df = pd.concat([sample_df] * 3, ignore_index=True)
        large_df['item_id'] = [f'item{i}' for i in range(len(large_df))]
        
        result = ml_engine.generate_trading_recommendations(
            large_df,
            player_level=15,
            capital=100000,
            risk_tolerance='medium'
        )
        
        # Should have recommendation columns
        assert 'rec_score' in result.columns
        assert 'accessible' in result.columns
        assert 'within_risk' in result.columns
        
        # All returned items should be accessible and within risk
        if not result.empty:
            assert result['accessible'].all()
            assert result['within_risk'].all()
    
    def test_singleton_get_ml_engine(self):
        """Test that get_ml_engine returns the same instance."""
        engine1 = get_ml_engine()
        engine2 = get_ml_engine()
        
        assert engine1 is engine2


class TestMLEngineEdgeCases:
    """Edge case tests for ML Engine."""
    
    def test_all_zero_profits(self):
        """Test handling of all-zero profit data."""
        engine = TarkovMLEngine()
        df = pd.DataFrame({
            'item_id': ['a', 'b', 'c', 'd', 'e'] * 4,
            'profit': [0] * 20,
            'flea_price': [10000] * 20,
            'trader_price': [10000] * 20,
            'width': [1] * 20,
            'height': [1] * 20,
            'weight': [1.0] * 20
        })
        
        result = engine.calculate_opportunity_score_ml(df)
        assert not result.empty
    
    def test_negative_prices(self):
        """Test handling of edge case with negative values."""
        engine = TarkovMLEngine()
        df = pd.DataFrame({
            'item_id': ['a', 'b', 'c', 'd', 'e'] * 4,
            'profit': [-1000, 1000, -500, 500, 0] * 4,
            'flea_price': [10000, 20000, 15000, 25000, 12000] * 4,
            'trader_price': [9000, 21000, 14500, 25500, 12000] * 4,
            'width': [1, 2, 1, 2, 1] * 4,
            'height': [1, 1, 2, 1, 2] * 4,
            'weight': [0.5, 1.0, 0.3, 0.8, 0.6] * 4
        })
        
        result = engine.calculate_risk_score(df)
        assert 'risk_score' in result.columns
        assert result['risk_score'].notna().all()
