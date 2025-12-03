import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import config

class TestCalculations(unittest.TestCase):
    def test_calculations(self):
        # Mock data with edge cases (0 price, 0 weight, 0 slots)
        data = {
            'item_id': ['1', '2', '3'],
            'profit': [1000, 500, 0],
            'flea_price': [10000, 0, 5000], # Item 2 has 0 flea price
            'width': [1, 2, 1],
            'height': [1, 0, 1], # Item 2 has 0 height (invalid but testing robustness)
            'weight': [1.0, 0.0, 0.5], # Item 2 has 0 weight
            'avg_24h_price': [11000, 0, 5000]
        }
        
        df = pd.DataFrame(data)
        
        # Logic from utils
        df = utils.calculate_metrics(df)
        
        # Assertions
        # Item 1: Normal
        self.assertEqual(df.loc[0, 'roi'], 10.0)
        self.assertEqual(df.loc[0, 'profit_per_slot'], 1000.0)
        self.assertEqual(df.loc[0, 'profit_per_kg'], 1000.0)
        
        # Item 2: Zeros
        self.assertEqual(df.loc[1, 'roi'], 0)
        self.assertEqual(df.loc[1, 'profit_per_slot'], 0)
        self.assertEqual(df.loc[1, 'profit_per_kg'], 0)
        self.assertEqual(df.loc[1, 'discount_percent'], 0)


class TestFleaLevelRequirement(unittest.TestCase):
    """Test the get_flea_level_requirement function for Patch 1.0 flea market restrictions."""
    
    def test_none_inputs(self):
        """Should handle None inputs gracefully."""
        level = utils.get_flea_level_requirement(None, None)  # type: ignore[arg-type]
        self.assertEqual(level, config.FLEA_MARKET_UNLOCK_LEVEL)
    
    def test_empty_string_inputs(self):
        """Should handle empty strings gracefully."""
        level = utils.get_flea_level_requirement("", "")
        self.assertEqual(level, config.FLEA_MARKET_UNLOCK_LEVEL)
    
    def test_specific_item_lock_graphics_card(self):
        """Graphics card should require level 40."""
        level = utils.get_flea_level_requirement("Graphics card", "Electronics")
        self.assertEqual(level, 40)
    
    def test_specific_item_lock_ledx(self):
        """LEDX should require level 35."""
        level = utils.get_flea_level_requirement("LEDX Skin Transilluminator", "Medical supplies")
        self.assertEqual(level, 35)
    
    def test_specific_item_lock_partial_match(self):
        """Should match partial item names."""
        level = utils.get_flea_level_requirement("Military circuit board", "Electronics")
        self.assertEqual(level, 35)
    
    def test_category_lock_assault_rifle(self):
        """Assault rifles should require level 25."""
        level = utils.get_flea_level_requirement("AK-74N", "Assault rifle")
        self.assertEqual(level, 25)
    
    def test_category_lock_suppressor(self):
        """Suppressors should require level 25."""
        level = utils.get_flea_level_requirement("PBS-4 5.45x39 sound suppressor", "Suppressor")
        self.assertEqual(level, 25)
    
    def test_category_lock_electronics(self):
        """Electronics category should require level 20."""
        level = utils.get_flea_level_requirement("USB Adapter", "Electronics")
        self.assertEqual(level, 20)
    
    def test_category_lock_valuables(self):
        """Valuables category should require level 30."""
        level = utils.get_flea_level_requirement("Bronze lion", "Valuables")
        self.assertEqual(level, 30)
    
    def test_default_level_for_unlocked_category(self):
        """Items in unlocked categories should return default level 15."""
        level = utils.get_flea_level_requirement("Random Item", "Food")
        self.assertEqual(level, config.FLEA_MARKET_UNLOCK_LEVEL)
    
    def test_item_lock_overrides_category_lock(self):
        """Item-specific lock should take priority over category lock."""
        # Graphics card is in Electronics (level 20) but has item lock (level 40)
        level = utils.get_flea_level_requirement("Graphics card", "Electronics")
        self.assertEqual(level, 40)
    
    def test_marked_keys(self):
        """All marked keys should require level 35."""
        level = utils.get_flea_level_requirement("Dorm room 314 marked key", "Mechanical key")
        self.assertEqual(level, 35)
    
    def test_ammo_pack_category(self):
        """Ammo packs should require level 25."""
        level = utils.get_flea_level_requirement("7.62x39mm BP ammo pack", "Ammo pack")
        self.assertEqual(level, 25)


if __name__ == '__main__':
    unittest.main()
