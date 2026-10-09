import unittest
import pandas as pd
import numpy as np
from eda_stats import calculate_summary_stats

class TestEdaStats(unittest.TestCase):
    def test_calculate_summary_stats_basic(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        stats = calculate_summary_stats(df)
        self.assertIn('A', stats)
        self.assertIn('B', stats)
        self.assertEqual(stats['A']['mean'], 3.0)
        self.assertEqual(stats['B']['mean'], 30.0)

if __name__ == '__main__':
    unittest.main()
