"""
Tests for the Alert System module.
"""

import pytest
import os
import tempfile
import json
from datetime import datetime, timedelta
from alerts import (
    Alert, AlertManager, AlertType, AlertPriority,
    get_alert_manager
)


class TestAlert:
    """Tests for the Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="test_alert",
            alert_type=AlertType.PROFIT_THRESHOLD.value,
            threshold_value=5000,
        )
        assert alert.id == "test_alert"
        assert alert.alert_type == AlertType.PROFIT_THRESHOLD.value
        assert alert.threshold_value == 5000
        assert alert.enabled is True
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            id="test_alert",
            alert_type=AlertType.HIGH_ROI.value,
            threshold_value=50.0,
            priority=AlertPriority.HIGH.value,
        )
        data = alert.to_dict()
        assert isinstance(data, dict)
        assert data['id'] == "test_alert"
        assert data['threshold_value'] == 50.0
    
    def test_alert_from_dict(self):
        """Test creating alert from dictionary."""
        data = {
            'id': 'from_dict',
            'alert_type': AlertType.ITEM_WATCHLIST.value,
            'threshold_value': 1000,
            'enabled': False,
        }
        alert = Alert.from_dict(data)
        assert alert.id == 'from_dict'
        assert alert.enabled is False
    
    def test_can_trigger_enabled(self):
        """Test can_trigger for enabled alert."""
        alert = Alert(id="test", alert_type="test", enabled=True)
        assert alert.can_trigger() is True
    
    def test_can_trigger_disabled(self):
        """Test can_trigger for disabled alert."""
        alert = Alert(id="test", alert_type="test", enabled=False)
        assert alert.can_trigger() is False
    
    def test_can_trigger_cooldown(self):
        """Test can_trigger respects cooldown."""
        alert = Alert(
            id="test", 
            alert_type="test",
            cooldown_minutes=60,
            last_triggered=datetime.now().isoformat()
        )
        assert alert.can_trigger() is False
        
        # After cooldown
        old_time = (datetime.now() - timedelta(minutes=61)).isoformat()
        alert.last_triggered = old_time
        assert alert.can_trigger() is True


class TestAlertManager:
    """Tests for the AlertManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def manager(self, temp_dir, monkeypatch):
        """Create an AlertManager with temporary files."""
        alerts_file = os.path.join(temp_dir, 'alerts.json')
        history_file = os.path.join(temp_dir, 'history.json')
        
        monkeypatch.setattr('alerts.ALERTS_FILE', alerts_file)
        monkeypatch.setattr('alerts.ALERT_HISTORY_FILE', history_file)
        
        return AlertManager()
    
    def test_add_alert(self, manager):
        """Test adding an alert."""
        alert = Alert(
            id="new_alert",
            alert_type=AlertType.PROFIT_THRESHOLD.value,
            threshold_value=10000,
        )
        result = manager.add_alert(alert)
        assert result is True
        assert manager.get_alert("new_alert") is not None
    
    def test_remove_alert(self, manager):
        """Test removing an alert."""
        alert = Alert(id="to_remove", alert_type="test")
        manager.add_alert(alert)
        
        result = manager.remove_alert("to_remove")
        assert result is True
        assert manager.get_alert("to_remove") is None
    
    def test_remove_nonexistent_alert(self, manager):
        """Test removing alert that doesn't exist."""
        result = manager.remove_alert("nonexistent")
        assert result is False
    
    def test_get_all_alerts(self, manager):
        """Test getting all alerts."""
        alerts = manager.get_all_alerts()
        assert isinstance(alerts, list)
        # Should have default alerts
        assert len(alerts) >= 1
    
    def test_add_item_watchlist(self, manager):
        """Test adding item to watchlist."""
        alert_id = manager.add_item_watchlist(
            item_id="test_item",
            item_name="Test Item",
            profit_threshold=5000
        )
        assert alert_id == "watchlist_test_item"
        
        alert = manager.get_alert(alert_id)
        assert alert is not None
        assert alert.item_id == "test_item"
        assert alert.threshold_value == 5000
    
    def test_enable_disable_alert(self, manager):
        """Test enabling and disabling alerts."""
        alert = Alert(id="toggle_test", alert_type="test", enabled=True)
        manager.add_alert(alert)
        
        manager.enable_alert("toggle_test", False)
        assert manager.get_alert("toggle_test").enabled is False
        
        manager.enable_alert("toggle_test", True)
        assert manager.get_alert("toggle_test").enabled is True
    
    def test_check_alerts_profit_threshold(self, manager):
        """Test checking profit threshold alerts."""
        alert = Alert(
            id="profit_test",
            alert_type=AlertType.PROFIT_THRESHOLD.value,
            threshold_value=5000,
            threshold_type="above",
        )
        manager.add_alert(alert)
        
        market_data = [
            {'item_id': 'item1', 'name': 'Item 1', 'profit': 10000, 'roi': 50},
            {'item_id': 'item2', 'name': 'Item 2', 'profit': 2000, 'roi': 10},
        ]
        
        triggered = manager.check_alerts(market_data)
        
        # Should trigger for item1 only
        assert len(triggered) >= 1
        triggered_items = [t['item_id'] for t in triggered if t['alert_id'] == 'profit_test']
        assert 'item1' in triggered_items
    
    def test_check_alerts_item_watchlist(self, manager):
        """Test checking watchlist alerts."""
        manager.add_item_watchlist("watched_item", "Watched Item", 3000)
        
        market_data = [
            {'item_id': 'watched_item', 'name': 'Watched Item', 'profit': 5000, 'roi': 25},
            {'item_id': 'other_item', 'name': 'Other Item', 'profit': 10000, 'roi': 50},
        ]
        
        triggered = manager.check_alerts(market_data)
        
        # Find watchlist alert
        watchlist_triggers = [t for t in triggered if 'watchlist' in t['alert_id']]
        assert len(watchlist_triggers) >= 1
    
    def test_get_recent_alerts(self, manager):
        """Test getting recent alerts."""
        # Trigger some alerts first
        alert = Alert(
            id="recent_test",
            alert_type=AlertType.PROFIT_THRESHOLD.value,
            threshold_value=1000,
        )
        manager.add_alert(alert)
        
        market_data = [{'item_id': 'test', 'name': 'Test', 'profit': 5000, 'roi': 25}]
        manager.check_alerts(market_data)
        
        recent = manager.get_recent_alerts(hours=1)
        assert len(recent) >= 1
    
    def test_get_alert_stats(self, manager):
        """Test getting alert statistics."""
        stats = manager.get_alert_stats()
        
        assert 'total_alerts' in stats
        assert 'enabled_alerts' in stats
        assert 'total_triggers' in stats
        assert 'history_count' in stats
    
    def test_callback_registration(self, manager):
        """Test registering and calling callbacks."""
        callback_data = []
        
        def test_callback(alert, item):
            callback_data.append({'alert': alert.id, 'item': item['item_id']})
        
        manager.register_callback(test_callback)
        
        alert = Alert(
            id="callback_test",
            alert_type=AlertType.PROFIT_THRESHOLD.value,
            threshold_value=1000,
        )
        manager.add_alert(alert)
        
        market_data = [{'item_id': 'cb_item', 'name': 'CB Item', 'profit': 5000, 'roi': 25}]
        manager.check_alerts(market_data)
        
        assert len(callback_data) >= 1


class TestAlertPersistence:
    """Tests for alert persistence."""
    
    def test_alerts_persist_across_restarts(self, tmp_path, monkeypatch):
        """Test that alerts persist across manager restarts."""
        alerts_file = tmp_path / 'alerts.json'
        history_file = tmp_path / 'history.json'
        
        monkeypatch.setattr('alerts.ALERTS_FILE', str(alerts_file))
        monkeypatch.setattr('alerts.ALERT_HISTORY_FILE', str(history_file))
        
        # Create manager and add alert
        manager1 = AlertManager()
        alert = Alert(id="persist_test", alert_type="test", threshold_value=999)
        manager1.add_alert(alert)
        
        # Create new manager instance
        manager2 = AlertManager()
        
        loaded = manager2.get_alert("persist_test")
        assert loaded is not None
        assert loaded.threshold_value == 999
