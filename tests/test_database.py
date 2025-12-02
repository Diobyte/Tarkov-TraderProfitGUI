import unittest
import sqlite3
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database

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
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0][1], 'Test Item') # Name
        self.assertEqual(latest[0][5], 5000) # Profit

if __name__ == '__main__':
    unittest.main()
