import unittest
import pandas as pd
import numpy as np

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
        
        # Logic from app.py
        
        # ROI
        df['roi'] = df.apply(lambda x: (x['profit'] / x['flea_price'] * 100) if x['flea_price'] > 0 else 0, axis=1)
        
        # Slots
        df['slots'] = df['width'] * df['height']
        df['profit_per_slot'] = df.apply(lambda x: x['profit'] / x['slots'] if x['slots'] > 0 else 0, axis=1)
        
        # Discount
        df['discount_from_avg'] = df['avg_24h_price'] - df['flea_price']
        df['discount_percent'] = df.apply(
            lambda x: (x['discount_from_avg'] / x['avg_24h_price'] * 100) if x['avg_24h_price'] > 0 else 0, 
            axis=1
        )
        
        # Profit per Kg
        df['profit_per_kg'] = df.apply(lambda x: x['profit'] / x['weight'] if x['weight'] > 0 else 0, axis=1)
        
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

if __name__ == '__main__':
    unittest.main()
