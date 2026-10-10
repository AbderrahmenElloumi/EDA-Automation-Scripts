"""
Test suite for Headless Plot Persistence and Metadata JSON Logging (Ticket #8)
"""

import unittest
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import shutil

from eda_common import finalize_plot


class TestHeadlessPlotPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_finalize_plot_saves_file_and_metadata(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        plot_path = str(Path(self.test_dir) / "test_plot.png")
        metadata = {"script": "test_script", "metric": "value"}

        finalize_plot(fig, plot_path, show=False, metadata=metadata)

        # Verify plot file exists
        self.assertTrue(Path(plot_path).exists())

        # Verify metadata JSON exists and has correct content
        meta_path = Path(plot_path).with_suffix(".json")
        self.assertTrue(meta_path.exists())

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["script"], "test_script")
        self.assertEqual(data["metric"], "value")
        self.assertIn("timestamp", data)
        self.assertEqual(data["plot_path"], str(Path(plot_path).resolve()))


if __name__ == "__main__":
    unittest.main()
