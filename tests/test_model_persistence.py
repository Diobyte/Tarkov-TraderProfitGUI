"""
Tests for the Model Persistence module.
"""

import pytest
import os
import json
import tempfile
from datetime import datetime, timedelta
from model_persistence import (
    ModelPersistence, get_model_persistence,
    MODEL_STATE_FILE, MODEL_HISTORY_FILE
)


class TestModelPersistence:
    """Tests for the ModelPersistence class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def persistence(self, temp_dir, monkeypatch):
        """Create a ModelPersistence with temporary files."""
        state_file = os.path.join(temp_dir, 'state.pkl')
        history_file = os.path.join(temp_dir, 'history.json')
        
        monkeypatch.setattr('model_persistence.MODEL_STATE_FILE', state_file)
        monkeypatch.setattr('model_persistence.MODEL_HISTORY_FILE', history_file)
        
        return ModelPersistence()
    
    def test_init_creates_default_state(self, persistence):
        """Test that initialization creates default state."""
        assert 'version' in persistence._state
        assert 'item_statistics' in persistence._state
        assert 'category_weights' in persistence._state
        assert 'trader_reliability' in persistence._state
    
    def test_init_creates_default_history(self, persistence):
        """Test that initialization creates default history."""
        assert 'version' in persistence._history
        assert 'sessions' in persistence._history
        assert 'total_items_seen' in persistence._history
    
    def test_update_item_statistics_new_item(self, persistence):
        """Test updating statistics for a new item."""
        persistence.update_item_statistics(
            item_id='test_item',
            profit=5000,
            flea_price=20000,
            offers=50,
            category='Meds',
            trader='Therapist'
        )
        
        stats = persistence.get_item_learned_stats('test_item')
        assert stats is not None
        assert stats['count'] == 1
        assert stats['profit_mean'] == 5000
        assert stats['category'] == 'Meds'
    
    def test_update_item_statistics_multiple_times(self, persistence):
        """Test updating statistics multiple times."""
        for profit in [1000, 2000, 3000, 4000, 5000]:
            persistence.update_item_statistics(
                item_id='multi_item',
                profit=profit,
                flea_price=20000,
                offers=50,
                category='Weapons',
                trader='Mechanic'
            )
        
        stats = persistence.get_item_learned_stats('multi_item')
        assert stats['count'] == 5
        assert stats['profit_mean'] == 3000  # Mean of 1000-5000
        assert stats['profit_min'] == 1000
        assert stats['profit_max'] == 5000
    
    def test_update_category_performance(self, persistence):
        """Test updating category performance."""
        for i in range(10):
            persistence.update_category_performance(
                category='TestCategory',
                profit=1000 + i * 100,
                is_profitable=i < 8  # 80% profitable
            )
        
        weight = persistence.get_category_weight('TestCategory')
        assert weight > 1.0  # High profitability should increase weight
    
    def test_update_trader_reliability(self, persistence):
        """Test updating trader reliability."""
        for i in range(15):
            persistence.update_trader_reliability(
                trader='TestTrader',
                profit=5000
            )
        
        reliability = persistence.get_trader_reliability('TestTrader')
        assert reliability > 0.5  # Positive profit should increase reliability
    
    def test_record_training_session(self, persistence):
        """Test recording a training session."""
        persistence.record_training_session(
            items_processed=100,
            profitable_count=75,
            avg_profit=3000
        )
        
        assert len(persistence._history['sessions']) >= 1
        assert persistence._history['total_items_seen'] == 100
        assert persistence._state['total_training_sessions'] == 1
    
    def test_update_calibration(self, persistence):
        """Test updating calibration parameters."""
        persistence.update_calibration(
            profit_mean=5000,
            profit_std=2000,
            roi_mean=25,
            roi_std=10
        )
        
        cal = persistence.get_calibration()
        assert cal['profit_mean'] > 0
        assert cal['profit_std'] > 0
    
    def test_save_and_load_state(self, persistence):
        """Test saving and loading state."""
        persistence.update_item_statistics(
            item_id='save_test',
            profit=7500,
            flea_price=30000,
            offers=100,
            category='Gear',
            trader='Ragman'
        )
        
        success = persistence.save_state()
        assert success is True
        
        # Reload state
        persistence._load_state()
        
        stats = persistence.get_item_learned_stats('save_test')
        assert stats is not None
        assert stats['profit_mean'] == 7500
    
    def test_save_and_load_history(self, persistence):
        """Test saving and loading history."""
        persistence.record_training_session(
            items_processed=50,
            profitable_count=40,
            avg_profit=2000
        )
        
        success = persistence.save_history()
        assert success is True
        
        # Reload history
        persistence._load_history()
        
        assert persistence._history['total_items_seen'] == 50
    
    def test_get_training_stats(self, persistence):
        """Test getting training statistics."""
        # Add some data
        persistence.update_item_statistics(
            item_id='item1', profit=1000, flea_price=10000,
            offers=20, category='Cat1', trader='Trader1'
        )
        persistence.update_category_performance('Cat1', 1000, True)
        persistence.update_trader_reliability('Trader1', 1000)
        
        stats = persistence.get_training_stats()
        
        assert 'total_samples' in stats
        assert 'unique_items_learned' in stats
        assert stats['unique_items_learned'] >= 1
    
    def test_get_learning_progress(self, persistence):
        """Test getting learning progress."""
        # Add substantial data
        for i in range(100):
            persistence.update_item_statistics(
                item_id=f'item_{i}',
                profit=1000 * (i + 1),
                flea_price=20000,
                offers=50,
                category='TestCat',
                trader='TestTrader'
            )
        persistence.record_training_session(100, 80, 5000)
        
        progress = persistence.get_learning_progress()
        
        assert 'overall_quality' in progress
        assert 'items_quality' in progress
        assert progress['unique_items_learned'] == 100
    
    def test_get_item_trends(self, persistence):
        """Test getting top items by learned profitability."""
        # Add items with different profitability
        for i, profit in enumerate([1000, 5000, 3000, 8000, 2000]):
            for _ in range(10):  # Need enough data points
                persistence.update_item_statistics(
                    item_id=f'trend_item_{i}',
                    profit=profit,
                    flea_price=20000,
                    offers=50,
                    category='TrendCat',
                    trader='TrendTrader'
                )
        
        trends = persistence.get_item_trends(top_n=5)
        
        assert len(trends) == 5
        # Should be sorted by profit (highest first)
        assert trends[0]['profit_mean'] >= trends[-1]['profit_mean']
    
    def test_get_category_trends(self, persistence):
        """Test getting category performance trends."""
        # Add data for multiple categories
        for cat, profits in [('CatA', [5000, 6000, 7000]), ('CatB', [1000, 2000, 3000])]:
            for profit in profits:
                persistence.update_category_performance(cat, profit, profit > 0)
        
        # Need at least 5 items per category
        for _ in range(3):
            persistence.update_category_performance('CatA', 5000, True)
            persistence.update_category_performance('CatB', 2000, True)
        
        trends = persistence.get_category_trends()
        
        assert len(trends) >= 2
        # CatA should have higher avg_profit
        cat_a = next((t for t in trends if t['category'] == 'CatA'), None)
        cat_b = next((t for t in trends if t['category'] == 'CatB'), None)
        if cat_a and cat_b:
            assert cat_a['avg_profit'] > cat_b['avg_profit']
    
    def test_cleanup_old_items(self, persistence):
        """Test cleaning up old items."""
        # Add an item
        persistence.update_item_statistics(
            item_id='old_item',
            profit=1000,
            flea_price=10000,
            offers=20,
            category='Cat',
            trader='Trader'
        )
        
        # Manually set last_seen to old date
        persistence._state['item_statistics']['old_item']['last_seen'] = (
            datetime.now() - timedelta(days=60)
        ).isoformat()
        
        # Add a recent item
        persistence.update_item_statistics(
            item_id='new_item',
            profit=2000,
            flea_price=15000,
            offers=30,
            category='Cat',
            trader='Trader'
        )
        
        # Cleanup items older than 30 days
        removed = persistence.cleanup_old_items(days=30)
        
        assert removed == 1
        assert persistence.get_item_learned_stats('old_item') is None
        assert persistence.get_item_learned_stats('new_item') is not None
    
    def test_consistency_score_calculation(self, persistence):
        """Test that consistency score is calculated correctly."""
        # Add consistent item (same profit)
        for _ in range(10):
            persistence.update_item_statistics(
                item_id='consistent_item',
                profit=5000,
                flea_price=20000,
                offers=50,
                category='Cat',
                trader='Trader'
            )
        
        # Add inconsistent item (varying profit)
        for profit in [1000, 9000, 2000, 8000, 3000, 7000, 4000, 6000, 5000, 10000]:
            persistence.update_item_statistics(
                item_id='inconsistent_item',
                profit=profit,
                flea_price=20000,
                offers=50,
                category='Cat',
                trader='Trader'
            )
        
        consistent_stats = persistence.get_item_learned_stats('consistent_item')
        inconsistent_stats = persistence.get_item_learned_stats('inconsistent_item')
        
        # Consistent item should have higher consistency score
        assert consistent_stats['consistency_score'] > inconsistent_stats['consistency_score']


    def test_cleanup_old_items_invalid_days(self, persistence):
        """Test cleanup_old_items handles invalid days parameter."""
        # Test with 0 days
        result = persistence.cleanup_old_items(days=0)
        assert result == 0
        
        # Test with negative days
        result = persistence.cleanup_old_items(days=-5)
        assert result == 0


class TestModelPersistenceSingleton:
    """Test the singleton pattern."""
    
    def test_get_model_persistence(self):
        """Test getting the singleton instance."""
        persistence = get_model_persistence()
        assert isinstance(persistence, ModelPersistence)
    
    def test_get_model_persistence_same_instance(self):
        """Test that get_model_persistence returns same instance."""
        p1 = get_model_persistence()
        p2 = get_model_persistence()
        assert p1 is p2
