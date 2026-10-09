import unittest
import pandas as pd
import numpy as np
from eda_stats import (
    calculate_summary_stats,
    iqr_bounds,
    iqr_outlier_mask,
    detect_outliers_zscore,
    detect_outliers_modified_zscore,
    classify_correlation_strength,
    classify_distribution_shape,
    bucket_top_n
)

class TestEdaStatsComplete(unittest.TestCase):
    def test_iqr_bounds(self):
        s = pd.Series([1, 2, 3, 4, 100]) # 100 is outlier
        lower, upper = iqr_bounds(s)
        self.assertLess(lower, upper)

    def test_iqr_outlier_mask(self):
        s = pd.Series([1, 2, 3, 4, 100])
        mask = iqr_outlier_mask(s)
        self.assertTrue(mask.iloc[4])
        self.assertFalse(mask.iloc[0])

    def test_detect_outliers_zscore(self):
        s = pd.Series([10, 12, 11, 10, 11, 100])
        mask = detect_outliers_zscore(s, threshold=2.0)
        self.assertTrue(mask.iloc[5])

    def test_detect_outliers_modified_zscore(self):
        s = pd.Series([10, 11, 10, 12, 11, 100])
        mask = detect_outliers_modified_zscore(s, threshold=3.5)
        self.assertTrue(mask.iloc[5])

    def test_classify_correlation_strength(self):
        self.assertEqual(classify_correlation_strength(0.95), "Very Strong")
        self.assertEqual(classify_correlation_strength(0.8), "Strong")
        self.assertEqual(classify_correlation_strength(0.6), "Moderate")
        self.assertEqual(classify_correlation_strength(0.4), "Weak")
        self.assertEqual(classify_correlation_strength(0.1), "Very Weak")

    def test_classify_distribution_shape(self):
        self.assertEqual(classify_distribution_shape(0.1, 1.0, True), "Normal")
        self.assertEqual(classify_distribution_shape(1.5, 1.0, False), "Right-skewed (Positive)")
        self.assertEqual(classify_distribution_shape(-1.5, 1.0, False), "Left-skewed (Negative)")

    def test_bucket_top_n(self):
        vc = pd.Series([10, 8, 5, 3, 2, 1], index=['A', 'B', 'C', 'D', 'E', 'F'])
        bucketed, other_n, coverage = bucket_top_n(vc, n=2)
        self.assertIn('A', bucketed.index)
        self.assertIn('B', bucketed.index)
        self.assertIn('Other (4 categories)', bucketed.index)

if __name__ == '__main__':
    unittest.main()
