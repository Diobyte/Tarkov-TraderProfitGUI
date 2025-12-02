import unittest
import os
import sys
from datetime import datetime, timedelta
import sqlite3

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database

class TestCleanupAndTrends(unittest.TestCase):
    def setUp(self):
        self.test_db = 'test_tarkov_data_cleanup_trends.db'
        database.DB_NAME = self.test_db
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        database.init_db()

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                pass

    def _insert_item(self, item_id: str, ts: datetime, profit: int = 1000):
        # Minimal tuple matching schema positions
        # (item_id, name, timestamp, flea_price, trader_price, trader_name, profit, icon_link, width, height, avg_24h_price, low_24h_price, change_last_48h, weight, category)
        database.save_prices_batch([
            (item_id, 'Item ' + item_id, ts, 10000, 11000, 'Trader', profit, '', 1, 1, 10000, 9000, 0.0, 1.0, 'Category')
        ])

    def test_cleanup_old_data_uses_iso_timestamp(self):
        now = datetime.now()
        old_ts = now - timedelta(days=10)
        recent_ts = now - timedelta(days=1)
        self._insert_item('a', old_ts)
        self._insert_item('b', recent_ts)

        deleted = database.cleanup_old_data(days=7, vacuum=False)
        self.assertIsNotNone(deleted)
        self.assertGreaterEqual(deleted, 1)  # type: ignore[arg-type]

        # Ensure recent remains
        conn = sqlite3.connect(self.test_db)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM prices')
        count = c.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_get_market_trends_time_window(self):
        now = datetime.now()
        inside = now - timedelta(hours=2)
        outside = now - timedelta(hours=10)
        # Insert older then newer to ensure MAX(timestamp) anchor works
        self._insert_item('x', outside, profit=500)
        self._insert_item('x', inside, profit=1500)

        rows = database.get_market_trends(hours=6)
        self.assertIsNotNone(rows)
        # Expect one row for item 'x' and avg computed over the window (> cutoff)
        self.assertTrue(any(r[0] == 'x' for r in rows))  # type: ignore[union-attr]
        for r in rows:  # type: ignore[union-attr]
            if r[0] == 'x':
                avg_profit = r[1]
                # Should reflect only the 'inside' record (1500)
                self.assertAlmostEqual(avg_profit, 1500.0, places=2)

if __name__ == '__main__':
    unittest.main()
