"""Alert System for Tarkov Trader Profit Analysis.

This module provides price alert functionality to notify users when
items reach profitable thresholds or unusual market conditions occur.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from tempfile import NamedTemporaryFile

import numpy as np
import math
import pandas as pd

import config

__all__ = [
    'AlertType', 'AlertPriority', 'Alert', 'AlertManager', 'get_alert_manager'
]

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_FILE = os.path.join(BASE_DIR, 'price_alerts.json')
ALERT_HISTORY_FILE = os.path.join(BASE_DIR, 'alert_history.json')


def _atomic_json_dump(file_path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Write JSON data atomically to avoid corrupting alert files.
    
    Uses a temporary file and atomic rename to ensure data integrity.
    Falls back to direct write if atomic operation fails.
    
    Args:
        file_path: Path to the JSON file to write.
        payload: Data to serialize as JSON (dict or list).
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    temp_name: Optional[str] = None
    try:
        with NamedTemporaryFile('w', delete=False, dir=directory or '.', encoding='utf-8', suffix='.tmp') as tmp:
            json.dump(payload, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, file_path)
        temp_name = None  # Clear so we don't try to delete on success
    except (OSError, IOError) as e:
        # Clean up temp file if it exists
        if temp_name:
            try:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except OSError:
                pass
            temp_name = None
        # Fall back to direct write if atomic write fails
        logger.warning("Atomic write failed, using direct write: %s", e)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
    finally:
        # Final cleanup attempt for temp file
        if temp_name:
            try:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except OSError:
                pass


class AlertType(Enum):
    """Types of alerts that can be triggered."""
    PROFIT_THRESHOLD = "profit_threshold"
    PRICE_DROP = "price_drop"
    PRICE_SPIKE = "price_spike"
    HIGH_ROI = "high_roi"
    VOLUME_SURGE = "volume_surge"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    ITEM_WATCHLIST = "item_watchlist"
    CATEGORY_ALERT = "category_alert"


class AlertPriority(Enum):
    """Priority levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a single alert configuration or triggered alert."""
    id: str
    alert_type: str
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    category: Optional[str] = None
    threshold_value: float = 0.0
    threshold_type: str = "above"  # "above" or "below"
    priority: str = "medium"
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    cooldown_minutes: int = 30
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create Alert from dictionary."""
        return cls(**data)
    
    def can_trigger(self) -> bool:
        """Check if alert can trigger based on cooldown."""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        try:
            last = datetime.fromisoformat(self.last_triggered)
            cooldown = timedelta(minutes=self.cooldown_minutes)
            return datetime.now() > last + cooldown
        except (ValueError, TypeError):
            return True


class AlertManager:
    """
    Manages price alerts and notifications.
    
    Thread-safe singleton manager for price alerts. Handles persistence,
    cooldown management, and callback notifications.
    
    Supports:
    - Custom profit threshold alerts per item
    - Category-wide alerts  
    - Automatic high-opportunity alerts
    - Alert history tracking with configurable limits
    - Cooldown management to prevent spam
    - Callback registration for custom alert handlers
    
    Example:
        >>> manager = get_alert_manager()
        >>> manager.add_item_watchlist('item123', 'GPU', 50000)
        >>> triggered = manager.check_alerts(market_data)
    """
    
    def __init__(self) -> None:
        """Initialize the alert manager."""
        self.alerts_file = ALERTS_FILE
        self.history_file = ALERT_HISTORY_FILE
        self._alerts: Dict[str, Alert] = {}
        self._history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[Alert, Dict[str, Any]], None]] = []
        self._load_alerts()
        self._load_history()
    
    def _load_alerts(self) -> None:
        """Load alerts from file."""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    data: Dict[str, Any] = json.load(f)
                    if not isinstance(data, dict):
                        raise json.JSONDecodeError("Expected dict", "", 0)
                    for alert_id, alert_data in data.items():
                        try:
                            if isinstance(alert_data, dict):
                                self._alerts[alert_id] = Alert.from_dict(alert_data)
                            else:
                                logger.warning("Skipping non-dict alert entry: %s", alert_id)
                        except (TypeError, KeyError) as e:
                            logger.warning("Skipping invalid alert entry %s: %s", alert_id, e)
                logger.info("Loaded %d alerts", len(self._alerts))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load alerts, recreating defaults: %s", e)
                self._alerts = {}
                self._create_default_alerts()
        else:
            self._create_default_alerts()
    
    def _load_history(self) -> None:
        """Load alert history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history: List[Dict[str, Any]] = json.load(f)
                if not isinstance(history, list):
                    raise json.JSONDecodeError("Expected list", "", 0)
                # Keep only last N entries based on config
                max_history = max(getattr(config, 'ALERT_MAX_HISTORY', 500), 1)
                self._history = history[-max_history:]
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load alert history, starting fresh: %s", e)
                self._history = []
    
    def _create_default_alerts(self) -> None:
        """Create default system alerts."""
        defaults = [
            Alert(
                id="high_profit_auto",
                alert_type=AlertType.PROFIT_THRESHOLD.value,
                threshold_value=float(config.ALERT_HIGH_PROFIT_THRESHOLD),
                threshold_type="above",
                priority=AlertPriority.HIGH.value,
                cooldown_minutes=config.ALERT_DEFAULT_COOLDOWN_MINUTES,
                message="Item has very high profit potential"
            ),
            Alert(
                id="high_roi_auto",
                alert_type=AlertType.HIGH_ROI.value,
                threshold_value=float(config.ALERT_HIGH_ROI_THRESHOLD),
                threshold_type="above",
                priority=AlertPriority.MEDIUM.value,
                cooldown_minutes=config.ALERT_DEFAULT_COOLDOWN_MINUTES,
                message="Item has exceptional ROI"
            ),
            Alert(
                id="arbitrage_auto",
                alert_type=AlertType.ARBITRAGE_OPPORTUNITY.value,
                threshold_value=0,
                priority=AlertPriority.CRITICAL.value,
                cooldown_minutes=config.ALERT_DEFAULT_COOLDOWN_MINUTES,
                message="Unusual arbitrage opportunity detected"
            ),
        ]
        for alert in defaults:
            self._alerts[alert.id] = alert
        self.save_alerts()
    
    def save_alerts(self) -> bool:
        """Save alerts to file."""
        try:
            data = {aid: alert.to_dict() for aid, alert in self._alerts.items()}
            _atomic_json_dump(self.alerts_file, data)
            return True
        except Exception as e:
            logger.error("Failed to save alerts: %s", e)
            return False
    
    def save_history(self) -> bool:
        """Save alert history to file."""
        try:
            max_history = max(config.ALERT_MAX_HISTORY, 1)
            _atomic_json_dump(self.history_file, self._history[-max_history:])
            return True
        except Exception as e:
            logger.error("Failed to save alert history: %s", e)
            return False
    
    def add_alert(self, alert: Alert) -> bool:
        """Add a new alert."""
        self._alerts[alert.id] = alert
        return self.save_alerts()
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert by ID."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return self.save_alerts()
        return False
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self._alerts.get(alert_id)
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all configured alerts."""
        return list(self._alerts.values())
    
    def enable_alert(self, alert_id: str, enabled: bool = True) -> bool:
        """Enable or disable an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].enabled = enabled
            return self.save_alerts()
        return False
    
    def add_item_watchlist(self, item_id: str, item_name: str, 
                           profit_threshold: float = 5000) -> str:
        """Add an item to the watchlist with profit threshold."""
        alert_id = f"watchlist_{item_id}"
        alert = Alert(
            id=alert_id,
            alert_type=AlertType.ITEM_WATCHLIST.value,
            item_id=item_id,
            item_name=item_name,
            threshold_value=profit_threshold,
            threshold_type="above",
            priority=AlertPriority.MEDIUM.value,
            message=f"Watchlist item {item_name} reached profit threshold"
        )
        self.add_alert(alert)
        return alert_id
    
    def register_callback(self, callback: Callable[[Alert, Dict[str, Any]], None]) -> None:
        """Register a callback for when alerts are triggered."""
        self._callbacks.append(callback)
    
    def check_alerts(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check all alerts against current market data.
        
        Args:
            market_data: List of item dicts with profit, roi, etc.
                        Each dict should contain at minimum 'item_id' and 'profit'.
            
        Returns:
            List of triggered alert notifications with alert details.
        """
        triggered: List[Dict[str, Any]] = []
        
        # Handle None or invalid input
        if not market_data or not isinstance(market_data, list):
            return triggered
        
        for item in market_data:
            # Skip invalid items
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('item_id', '')).strip() if item.get('item_id') else ''
            # Safely extract numeric values, handling None, NaN, and invalid types
            try:
                profit_raw = item.get('profit', 0)
                profit = float(profit_raw) if profit_raw is not None else 0.0
                if not np.isfinite(profit):
                    profit = 0.0
            except (ValueError, TypeError):
                profit = 0.0
            try:
                roi_raw = item.get('roi', 0)
                roi = float(roi_raw) if roi_raw is not None else 0.0
                if not np.isfinite(roi):
                    roi = 0.0
            except (ValueError, TypeError):
                roi = 0.0
            try:
                offers_raw = item.get('last_offer_count', 0)
                offers = int(offers_raw) if offers_raw is not None else 0
            except (ValueError, TypeError):
                offers = 0
            
            # Handle is_anomaly - can be bool, int, float (NaN), or missing
            is_anomaly_raw = item.get('is_anomaly', False)
            try:
                if is_anomaly_raw is None or (isinstance(is_anomaly_raw, float) and np.isnan(is_anomaly_raw)):
                    is_anomaly = False
                else:
                    is_anomaly = bool(is_anomaly_raw)
            except (ValueError, TypeError):
                is_anomaly = False
            
            for alert in self._alerts.values():
                if not alert.can_trigger():
                    continue
                
                should_trigger = False
                
                # Check different alert types
                if alert.alert_type == AlertType.PROFIT_THRESHOLD.value:
                    if alert.item_id and alert.item_id != item_id:
                        continue
                    if alert.threshold_type == "above" and profit >= alert.threshold_value:
                        should_trigger = True
                    elif alert.threshold_type == "below" and profit <= alert.threshold_value:
                        should_trigger = True
                
                elif alert.alert_type == AlertType.HIGH_ROI.value:
                    if roi >= alert.threshold_value:
                        should_trigger = True
                
                elif alert.alert_type == AlertType.ITEM_WATCHLIST.value:
                    if alert.item_id == item_id and profit >= alert.threshold_value:
                        should_trigger = True
                
                elif alert.alert_type == AlertType.ARBITRAGE_OPPORTUNITY.value:
                    if is_anomaly and profit > 5000:
                        should_trigger = True
                
                elif alert.alert_type == AlertType.CATEGORY_ALERT.value:
                    if alert.category == item.get('category') and profit >= alert.threshold_value:
                        should_trigger = True
                
                if should_trigger:
                    notification = self._trigger_alert(alert, item)
                    triggered.append(notification)
        
        if triggered:
            self.save_alerts()
            self.save_history()
        
        return triggered
    
    def _trigger_alert(self, alert: Alert, item: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger an alert and record it."""
        alert.last_triggered = datetime.now().isoformat()
        alert.trigger_count += 1
        
        notification = {
            'alert_id': alert.id,
            'alert_type': alert.alert_type,
            'priority': alert.priority,
            'item_id': item.get('item_id'),
            'item_name': item.get('name'),
            'profit': item.get('profit'),
            'roi': item.get('roi'),
            'message': alert.message or f"Alert triggered for {item.get('name')}",
            'triggered_at': alert.last_triggered,
        }
        
        self._history.append(notification)
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(alert, item)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)
        
        logger.info("Alert triggered: %s for %s", alert.id, item.get('name'))
        return notification
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts triggered in the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        for entry in reversed(self._history):
            try:
                triggered_at = datetime.fromisoformat(entry.get('triggered_at', ''))
                if triggered_at > cutoff:
                    recent.append(entry)
                else:
                    break
            except (ValueError, TypeError):
                continue
        return recent
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self._alerts)
        enabled_alerts = sum(1 for a in self._alerts.values() if a.enabled)
        total_triggers = sum(a.trigger_count for a in self._alerts.values())
        
        return {
            'total_alerts': total_alerts,
            'enabled_alerts': enabled_alerts,
            'disabled_alerts': total_alerts - enabled_alerts,
            'total_triggers': total_triggers,
            'history_count': len(self._history),
        }


# Singleton instance with thread-safe initialization
import threading
_alert_manager: Optional[AlertManager] = None
_alert_manager_lock: threading.Lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    """Get or create the alert manager singleton (thread-safe).
    
    Returns:
        The singleton AlertManager instance.
        
    Raises:
        RuntimeError: If alert manager initialization fails.
    """
    global _alert_manager
    if _alert_manager is None:
        with _alert_manager_lock:
            # Double-check locking pattern
            if _alert_manager is None:
                try:
                    _alert_manager = AlertManager()
                except Exception as e:
                    logger.error("Failed to initialize alert manager: %s", e)
                    raise RuntimeError(f"Alert manager initialization failed: {e}") from e
    return _alert_manager
