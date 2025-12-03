import unittest
import sqlite3
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database


class TestParseTimestamp(unittest.TestCase):
    """Tests for the parse_timestamp utility function."""
    
    def test_parse_iso_format(self):
        """Test parsing ISO format timestamps."""
        ts = database.parse_timestamp('2024-01-15T10:30:45.123456')
        self.assertIsNotNone(ts)
        assert ts is not None
        self.assertEqual(ts.year, 2024)
        self.assertEqual(ts.month, 1)
        self.assertEqual(ts.day, 15)
    
    def test_parse_legacy_format_with_space(self):
        """Test parsing legacy format with space separator."""
        ts = database.parse_timestamp('2024-01-15 10:30:45.123456')
        self.assertIsNotNone(ts)
        assert ts is not None
        self.assertEqual(ts.year, 2024)
    
    def test_parse_simple_format(self):
        """Test parsing simple datetime format."""
        ts = database.parse_timestamp('2024-01-15 10:30:45')
        self.assertIsNotNone(ts)
        assert ts is not None
        self.assertEqual(ts.hour, 10)
    
    def test_parse_none(self):
        """Test parsing None returns None."""
        self.assertIsNone(database.parse_timestamp(None))
    
    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        self.assertIsNone(database.parse_timestamp(''))
    
    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        self.assertIsNone(database.parse_timestamp('not-a-timestamp'))


class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Use a temporary file for DB testing instead of the real one
        self.test_db = 'test_tarkov_data.db'
        database.DB_NAME = self.test_db
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                pass # Windows file locking sometimes delays deletion

    def test_init_db(self):
        database.init_db()
        self.assertTrue(os.path.exists(self.test_db))
        
        conn = sqlite3.connect(self.test_db)
        c = conn.cursor()
        
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
        self.assertIsNotNone(c.fetchone())
        
        # Check columns
        c.execute("PRAGMA table_info(prices)")
        columns = {row[1] for row in c.fetchall()}
        expected_columns = {
            'id', 'item_id', 'name', 'timestamp', 'flea_price', 'trader_price', 
            'trader_name', 'profit', 'icon_link', 'width', 'height', 
            'avg_24h_price', 'low_24h_price', 'change_last_48h', 'weight', 'category'
        }
        self.assertTrue(expected_columns.issubset(columns))
        conn.close()

    def test_save_and_retrieve(self):
        database.init_db()
        
        # Mock data
        # (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
        now = datetime.now()
        items = [
            ('1', 'Test Item', now, 10000, 15000, 'Therapist', 5000, 'http://link', 1, 1, 11000, 9000, 5.0, 1.0, 'Medical')
        ]
        
        database.save_prices_batch(items)
        
        # Retrieve
        latest = database.get_latest_prices()
        self.assertIsNotNone(latest)
        self.assertIsInstance(latest, list)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0][1], 'Test Item') # Name
        self.assertEqual(latest[0][5], 5000) # Profit

    def test_save_and_retrieve_enhanced_format(self):
        """Test saving and retrieving items with the enhanced 25-column format."""
        database.init_db()
        
        now = datetime.now()
        # Enhanced format: (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, 
        #                   icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, 
        #                   weight, category, base_price, high_24h_price, last_offer_count, short_name,
        #                   wiki_link, trader_level_required, trader_task_unlock, price_velocity, 
        #                   liquidity_score, api_updated)
        items = [
            ('2', 'Enhanced Test Item', now, 20000, 25000, 'Peacekeeper', 5000, 
             'http://icon.link', 2, 2, 21000, 18000, -3.5, 
             2.5, 'Weapons', 15000, 22000, 100, 'ETI',
             'http://wiki.link', 2, 'Test Task', 5.0, 
             80.0, '2024-01-01T00:00:00')
        ]
        
        database.save_prices_batch(items)
        
        # Retrieve
        latest = database.get_latest_prices()
        self.assertIsNotNone(latest)
        self.assertIsInstance(latest, list)
        self.assertGreaterEqual(len(latest), 1)
        
        # Find our item
        enhanced_item = next((r for r in latest if r[0] == '2'), None)
        self.assertIsNotNone(enhanced_item)
        self.assertIsInstance(enhanced_item, tuple)
        self.assertEqual(enhanced_item[1], 'Enhanced Test Item')  # Name
        self.assertEqual(enhanced_item[5], 5000)  # Profit
        self.assertEqual(enhanced_item[18], 'ETI')  # Short name

if __name__ == '__main__':
    unittest.main()
