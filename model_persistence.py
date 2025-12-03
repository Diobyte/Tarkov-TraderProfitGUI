"""
Model Persistence Layer for Tarkov Trader Profit Analysis.

This module handles saving and loading of ML model state, learned parameters,
and historical statistics that persist across database cleanups.

The model state survives even when the database is cleared after 7 days,
allowing the ML engine to continuously improve its predictions over time.

Security Note:
    This module uses pickle for serialization of internal model state.
    The pickle files are stored locally and are not user-facing. Only load
    pickle files from trusted sources as pickle can execute arbitrary code.
    For maximum security in production, consider migrating to JSON where possible.
"""

import json
import math
import pickle
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

import config

__all__ = [
    'ModelPersistence', 'get_model_persistence',
    'MODEL_STATE_FILE', 'MODEL_HISTORY_FILE'
]

logger = logging.getLogger(__name__)

# Persistence files stored alongside the database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_STATE_FILE = os.path.join(BASE_DIR, 'ml_model_state.pkl')
MODEL_HISTORY_FILE = os.path.join(BASE_DIR, 'ml_learned_history.json')


class ModelPersistence:
    """
    Handles persistent storage of ML model state and learned parameters.
    
    This class ensures that learned patterns and statistics survive
    database cleanups, allowing the model to continuously improve.
    
    Stored data includes:
    - Item profit statistics (mean, variance, trends)
    - Category performance patterns
    - Trader reliability scores
    - Historical accuracy metrics
    - Model calibration parameters
    """
    
    def __init__(self) -> None:
        """Initialize the persistence layer."""
        self.state_file = MODEL_STATE_FILE
        self.history_file = MODEL_HISTORY_FILE
        self._state: Dict[str, Any] = {}
        self._history: Dict[str, Any] = {}
        self._load_state()
        self._load_history()
    
    def _load_state(self) -> None:
        """Load model state from pickle file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    loaded_state = pickle.load(f)
                    # Validate that loaded state has expected structure
                    if isinstance(loaded_state, dict) and 'version' in loaded_state:
                        self._state = loaded_state
                        logger.info("Loaded model state from %s", self.state_file)
                    else:
                        logger.warning("Invalid state structure, using defaults")
                        self._state = self._get_default_state()
            except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as e:
                logger.warning("Failed to load model state (corrupted?): %s", e)
                self._state = self._get_default_state()
            except Exception as e:
                logger.warning("Failed to load model state: %s", e)
                self._state = self._get_default_state()
        else:
            self._state = self._get_default_state()
    
    def _load_history(self) -> None:
        """Load learned history from JSON file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                    # Validate structure
                    if isinstance(loaded_history, dict) and 'version' in loaded_history:
                        self._history = loaded_history
                        logger.info("Loaded learned history from %s", self.history_file)
                    else:
                        logger.warning("Invalid history structure, using defaults")
                        self._history = self._get_default_history()
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse learned history: %s", e)
                self._history = self._get_default_history()
            except (OSError, IOError) as e:
                logger.warning("Failed to load learned history: %s", e)
                self._history = self._get_default_history()
        else:
            self._history = self._get_default_history()
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Get default model state structure."""
        return {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_training_samples': 0,
            'total_training_sessions': 0,
            'item_statistics': {},  # item_id -> stats
            'category_weights': {},  # category -> learned weight
            'trader_reliability': {},  # trader -> reliability score
            'feature_importance': {},  # feature -> importance
            'calibration': {
                'profit_mean': 0,
                'profit_std': 1,
                'roi_mean': 0,
                'roi_std': 1,
            }
        }
    
    def _get_default_history(self) -> Dict[str, Any]:
        """Get default history structure."""
        return {
            'version': '1.0',
            'sessions': [],  # List of training sessions
            'daily_summaries': [],  # Daily aggregated stats
            'category_trends': {},  # category -> trend data
            'accuracy_history': [],  # Historical accuracy metrics
            'total_items_seen': 0,
            'total_profitable_items': 0,
        }
    
    def save_state(self) -> bool:
        """Save model state to disk."""
        try:
            self._state['last_updated'] = datetime.now().isoformat()
            with open(self.state_file, 'wb') as f:
                pickle.dump(self._state, f)
            logger.info("Saved model state to %s", self.state_file)
            return True
        except Exception as e:
            logger.error("Failed to save model state: %s", e)
            return False
    
    def save_history(self) -> bool:
        """Save learned history to disk."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, default=str)
            logger.info("Saved learned history to %s", self.history_file)
            return True
        except Exception as e:
            logger.error("Failed to save learned history: %s", e)
            return False
    
    def update_item_statistics(self, item_id: str, profit: float, 
                               flea_price: float, offers: int,
                               category: str, trader: str) -> bool:
        """
        Update running statistics for an item.
        
        Uses Welford's online algorithm for numerically stable
        running mean and variance calculation. This algorithm allows
        incremental updates without storing all historical values.
        
        Args:
            item_id: Unique item identifier (non-empty string required).
            profit: Current profit value.
            flea_price: Current flea market price.
            offers: Number of offers on flea market.
            category: Item category name.
            trader: Best trader name for this item.
            
        Returns:
            True if statistics were updated, False if item_id was invalid.
            
        Note:
            NaN/inf values are automatically sanitized to prevent corruption.
        """
        # Skip empty or invalid item IDs
        if not item_id or not isinstance(item_id, str):
            return False
        item_id = str(item_id).strip()
        if not item_id:
            return False
        
        # Sanitize category and trader
        category = str(category).strip() if category else 'Unknown'
        trader = str(trader).strip() if trader else 'Unknown'
        if not category:
            category = 'Unknown'
        if not trader:
            trader = 'Unknown'
        
        # Sanitize inputs to avoid NaN/inf issues
        # Convert numpy/pandas types to Python native types
        if hasattr(profit, 'item'):
            profit = float(profit.item())
        elif hasattr(profit, 'iloc'):
            profit = float(profit.iloc[0]) if len(profit) > 0 else 0.0
        if hasattr(flea_price, 'item'):
            flea_price = float(flea_price.item())
        elif hasattr(flea_price, 'iloc'):
            flea_price = float(flea_price.iloc[0]) if len(flea_price) > 0 else 0.0
        if hasattr(offers, 'item'):
            offers = int(offers.item())
        elif hasattr(offers, 'iloc'):
            offers = int(offers.iloc[0]) if len(offers) > 0 else 0
            
        if not isinstance(profit, (int, float)) or math.isnan(profit) or math.isinf(profit):
            profit = 0.0
        if not isinstance(flea_price, (int, float)) or math.isnan(flea_price) or math.isinf(flea_price):
            flea_price = 0.0
        if not isinstance(offers, (int, float)) or (isinstance(offers, float) and (math.isnan(offers) or math.isinf(offers))):
            offers = 0
            
        if item_id not in self._state['item_statistics']:
            self._state['item_statistics'][item_id] = {
                'count': 0,
                'profit_mean': 0.0,
                'profit_m2': 0.0,  # For variance calculation
                'profit_min': profit,  # Initialize with first profit instead of inf
                'profit_max': profit,  # Initialize with first profit instead of -inf
                'flea_price_mean': 0.0,
                'offers_mean': 0.0,
                'category': category,
                'trader': trader,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'trend_direction': 'stable',
                'consistency_score': 50.0,
            }
        
        stats = self._state['item_statistics'][item_id]
        n = stats['count'] + 1
        
        # Welford's online algorithm for mean and variance
        delta = profit - stats['profit_mean']
        stats['profit_mean'] += delta / n
        delta2 = profit - stats['profit_mean']
        stats['profit_m2'] += delta * delta2
        
        # Update other stats
        stats['count'] = n
        # Use safe min/max to handle potential inf values from initialization
        current_min = stats['profit_min']
        current_max = stats['profit_max']
        if math.isinf(current_min) or math.isnan(current_min):
            stats['profit_min'] = profit
        else:
            stats['profit_min'] = min(current_min, profit)
        if math.isinf(current_max) or math.isnan(current_max):
            stats['profit_max'] = profit
        else:
            stats['profit_max'] = max(current_max, profit)
        stats['flea_price_mean'] = stats['flea_price_mean'] + (flea_price - stats['flea_price_mean']) / n
        stats['offers_mean'] = stats['offers_mean'] + (offers - stats['offers_mean']) / n
        stats['last_seen'] = datetime.now().isoformat()
        stats['category'] = category
        stats['trader'] = trader
        
        # Calculate consistency (low variance = high consistency)
        if n >= 2:
            variance = stats['profit_m2'] / (n - 1)
            std = np.sqrt(max(variance, 0))
            # Consistency: items with low std relative to mean are more consistent
            if abs(stats['profit_mean']) > 1:
                cv = std / abs(stats['profit_mean'])  # Coefficient of variation
                stats['consistency_score'] = max(0, min(100, 100 * (1 - cv)))
            else:
                stats['consistency_score'] = 50.0
        
        return True
    
    def update_category_performance(self, category: str, profit: float, 
                                    is_profitable: bool) -> None:
        """Update category performance tracking.
        
        Args:
            category: Category name to update.
            profit: Profit value to incorporate.
            is_profitable: Whether this item was profitable.
        """
        # Validate category name
        if not category or not isinstance(category, str):
            return
        category = category.strip()
        if not category:
            return
        
        # Sanitize profit
        if not isinstance(profit, (int, float)) or (isinstance(profit, float) and (math.isnan(profit) or math.isinf(profit))):
            profit = 0.0
            
        if category not in self._state['category_weights']:
            self._state['category_weights'][category] = {
                'total_items': 0,
                'profitable_items': 0,
                'avg_profit': 0.0,
                'weight': 1.0,
            }
        
        cat = self._state['category_weights'][category]
        n = cat['total_items'] + 1
        cat['total_items'] = n
        cat['profitable_items'] += 1 if is_profitable else 0
        cat['avg_profit'] = cat['avg_profit'] + (profit - cat['avg_profit']) / n
        
        # Category weight based on profitability rate
        if n >= 5:
            rate = cat['profitable_items'] / n
            cat['weight'] = 0.5 + rate  # Weight ranges from 0.5 to 1.5
    
    def update_trader_reliability(self, trader: str, profit: float) -> None:
        """Update trader reliability scores.
        
        Args:
            trader: Trader name to update.
            profit: Profit value to incorporate.
        """
        # Validate trader name
        if not trader or not isinstance(trader, str):
            return
        trader = trader.strip()
        if not trader:
            return
        
        # Sanitize profit
        if not isinstance(profit, (int, float)) or (isinstance(profit, float) and (math.isnan(profit) or math.isinf(profit))):
            profit = 0.0
            
        if trader not in self._state['trader_reliability']:
            self._state['trader_reliability'][trader] = {
                'count': 0,
                'avg_profit': 0.0,
                'reliability': 0.5,
            }
        
        t = self._state['trader_reliability'][trader]
        n = t['count'] + 1
        t['count'] = n
        t['avg_profit'] = t['avg_profit'] + (profit - t['avg_profit']) / n
        
        # Reliability based on average profit
        if n >= 10:
            # Normalize to 0-1 scale based on profit
            t['reliability'] = min(1.0, max(0.0, 0.5 + t['avg_profit'] / 20000))
    
    def record_training_session(self, items_processed: int, 
                               profitable_count: int,
                               avg_profit: float,
                               accuracy: Optional[float] = None) -> None:
        """
        Record a training session for historical tracking.
        
        Args:
            items_processed: Number of items in this session
            profitable_count: Number of profitable items
            avg_profit: Average profit of all items
            accuracy: Optional accuracy metric if available
        """
        session = {
            'timestamp': datetime.now().isoformat(),
            'items_processed': items_processed,
            'profitable_count': profitable_count,
            'profitable_rate': profitable_count / max(items_processed, 1),
            'avg_profit': avg_profit,
            'accuracy': accuracy,
        }
        
        self._history['sessions'].append(session)
        self._history['total_items_seen'] += items_processed
        self._history['total_profitable_items'] += profitable_count
        
        # Keep only last 1000 sessions
        if len(self._history['sessions']) > 1000:
            self._history['sessions'] = self._history['sessions'][-1000:]
        
        # Update state counters
        self._state['total_training_samples'] += items_processed
        self._state['total_training_sessions'] += 1
        
        # Record accuracy history
        if accuracy is not None:
            self._history['accuracy_history'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
            })
            if len(self._history['accuracy_history']) > 500:
                self._history['accuracy_history'] = self._history['accuracy_history'][-500:]
    
    def update_daily_summary(self, date: str, summary: Dict[str, Any]) -> None:
        """Update or add a daily summary."""
        # Find existing summary for this date
        for i, s in enumerate(self._history['daily_summaries']):
            if s.get('date') == date:
                self._history['daily_summaries'][i] = summary
                return
        
        self._history['daily_summaries'].append(summary)
        
        # Keep only last 90 days
        if len(self._history['daily_summaries']) > 90:
            self._history['daily_summaries'] = self._history['daily_summaries'][-90:]
    
    def get_item_learned_stats(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get learned statistics for a specific item.
        
        Args:
            item_id: The item ID to look up.
            
        Returns:
            Dict with item statistics or None if not found.
            Includes count, profit_mean, profit_min, profit_max, consistency_score,
            first_seen, last_seen, category, and trader.
        """
        if not item_id or not isinstance(item_id, str):
            return None
        item_id = item_id.strip()
        if not item_id:
            return None
        return self._state['item_statistics'].get(item_id)
    
    def get_category_weight(self, category: str) -> float:
        """Get learned weight for a category.
        
        Args:
            category: Category name to look up.
            
        Returns:
            Float weight (default 1.0 if category not found or invalid).
        """
        if not category or not isinstance(category, str):
            return 1.0
        category = category.strip()
        if not category:
            return 1.0
        cat = self._state['category_weights'].get(category, {})
        return cat.get('weight', 1.0)
    
    def get_trader_reliability(self, trader: str) -> float:
        """Get reliability score for a trader.
        
        Args:
            trader: Trader name to look up.
            
        Returns:
            Float reliability score (default 0.5 if trader not found or invalid).
        """
        if not trader or not isinstance(trader, str):
            return 0.5
        trader = trader.strip()
        if not trader:
            return 0.5
        t = self._state['trader_reliability'].get(trader, {})
        return t.get('reliability', 0.5)
    
    def get_calibration(self) -> Dict[str, float]:
        """Get calibration parameters.
        
        Returns:
            Dict with profit_mean, profit_std, roi_mean, roi_std.
            All values are guaranteed to be valid floats.
        """
        defaults: Dict[str, float] = {
            'profit_mean': 0.0,
            'profit_std': 1.0,
            'roi_mean': 0.0,
            'roi_std': 1.0,
        }
        cal = self._state.get('calibration', defaults)
        # Ensure all values are valid floats
        result: Dict[str, float] = {}
        for key, default in defaults.items():
            val = cal.get(key, default)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                result[key] = default
            else:
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    result[key] = default
        return result
    
    def update_calibration(self, profit_mean: float, profit_std: float,
                          roi_mean: float, roi_std: float) -> None:
        """Update calibration parameters with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        cal = self._state['calibration']
        
        cal['profit_mean'] = alpha * profit_mean + (1 - alpha) * cal['profit_mean']
        cal['profit_std'] = alpha * profit_std + (1 - alpha) * cal['profit_std']
        cal['roi_mean'] = alpha * roi_mean + (1 - alpha) * cal['roi_mean']
        cal['roi_std'] = alpha * roi_std + (1 - alpha) * cal['roi_std']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get overall training statistics."""
        return {
            'total_samples': self._state['total_training_samples'],
            'total_sessions': self._state['total_training_sessions'],
            'unique_items_learned': len(self._state['item_statistics']),
            'categories_tracked': len(self._state['category_weights']),
            'traders_tracked': len(self._state['trader_reliability']),
            'last_updated': self._state.get('last_updated'),
            'created_at': self._state.get('created_at'),
        }
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics for UI display."""
        stats = self.get_training_stats()
        
        # Calculate learning quality based on data volume
        items_quality = min(100, stats['unique_items_learned'] / 50 * 100)
        sessions_quality = min(100, stats['total_sessions'] / 100 * 100)
        samples_quality = min(100, stats['total_samples'] / 10000 * 100)
        
        overall_quality = (items_quality + sessions_quality + samples_quality) / 3
        
        # Get recent accuracy if available
        recent_accuracy = None
        if self._history['accuracy_history']:
            recent = self._history['accuracy_history'][-10:]
            valid_accuracies = [a['accuracy'] for a in recent if a['accuracy'] is not None]
            if valid_accuracies:
                recent_accuracy = float(np.mean(valid_accuracies))
        
        return {
            'overall_quality': overall_quality,
            'items_quality': items_quality,
            'sessions_quality': sessions_quality,
            'samples_quality': samples_quality,
            'recent_accuracy': recent_accuracy,
            **stats
        }
    
    def get_item_trends(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top items by learned profitability."""
        items = []
        for item_id, stats in self._state['item_statistics'].items():
            if stats['count'] >= config.TREND_MIN_DATA_POINTS:
                items.append({
                    'item_id': item_id,
                    'profit_mean': stats['profit_mean'],
                    'consistency_score': stats['consistency_score'],
                    'data_points': stats['count'],
                    'category': stats['category'],
                    'trader': stats['trader'],
                    'last_seen': stats['last_seen'],
                })
        
        # Sort by consistency-weighted profit
        items.sort(key=lambda x: x['profit_mean'] * (x['consistency_score'] / 100), reverse=True)
        return items[:top_n]
    
    def get_category_trends(self) -> List[Dict[str, Any]]:
        """Get category performance trends."""
        categories = []
        for cat, stats in self._state['category_weights'].items():
            if stats['total_items'] >= 5:
                categories.append({
                    'category': cat,
                    'avg_profit': stats['avg_profit'],
                    'profitable_rate': stats['profitable_items'] / stats['total_items'],
                    'weight': stats['weight'],
                    'total_items': stats['total_items'],
                })
        
        categories.sort(key=lambda x: x['avg_profit'], reverse=True)
        return categories
    
    def cleanup_old_items(self, days: int = 30) -> int:
        """
        Remove items not seen in the specified number of days.
        
        Args:
            days: Remove items not seen in this many days. Must be positive.
            
        Returns:
            Number of items removed. Returns 0 if days is invalid.
        """
        # Validate days parameter
        if days <= 0:
            logger.warning("cleanup_old_items called with invalid days: %d, skipping", days)
            return 0
        
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        to_remove = []
        for item_id, stats in self._state['item_statistics'].items():
            try:
                last_seen = datetime.fromisoformat(stats['last_seen'])
                if last_seen < cutoff:
                    to_remove.append(item_id)
            except (ValueError, KeyError):
                pass
        
        for item_id in to_remove:
            del self._state['item_statistics'][item_id]
        
        if to_remove:
            logger.info("Removed %d stale items from learned state", len(to_remove))
        
        return len(to_remove)


# Singleton instance with thread-safe initialization
import threading
_persistence: Optional[ModelPersistence] = None
_persistence_lock: threading.Lock = threading.Lock()


def get_model_persistence() -> ModelPersistence:
    """Get or create the model persistence singleton (thread-safe).
    
    Returns:
        The singleton ModelPersistence instance.
        
    Raises:
        RuntimeError: If model persistence initialization fails.
    """
    global _persistence
    if _persistence is None:
        with _persistence_lock:
            # Double-check locking pattern
            if _persistence is None:
                try:
                    _persistence = ModelPersistence()
                except Exception as e:
                    logger.error("Failed to initialize model persistence: %s", e)
                    raise RuntimeError(f"Model persistence initialization failed: {e}") from e
    return _persistence
