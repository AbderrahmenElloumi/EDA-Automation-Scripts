"""
eda_common.py

Shared utilities used across the EDA script suite (data_profiler,
distribution_analyzer, correlation_explorer, outlier_suite,
missing_data_analyzer). Centralizing this logic fixes a cross-cutting
issue in the original scripts: the IQR-outlier calculation and the
"show-only" plotting pattern were copy-pasted (and slightly
inconsistent) in three different files.

Provides:
- logging setup (replaces bare `print` statements)
- a shared `iqr_bounds` / `iqr_outlier_mask` implementation
- `finalize_plot` helper so every script can save-to-file instead of
  only calling plt.show() (needed for headless / batch use)
- a common CLI argument parser + CSV loader so every script can be run
  directly against a real dataset instead of only synthetic demo data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Shared statistics
# --------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Tuple[float, float]:
    """Return (lower_bound, upper_bound) for the IQR outlier rule.

    Single source of truth for IQR logic — previously reimplemented
    (with copy-pasted code) in data_profiler.py, distribution_analyzer.py,
    and outlier_suite.py.
    """
    clean = series.dropna()
    if clean.empty:
        return (np.nan, np.nan)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    return (q1 - multiplier * iqr, q3 + multiplier * iqr)


def iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Boolean mask (aligned to series.index) flagging IQR outliers.

    Correctly returns False (not NaN-propagated) for missing values,
    and preserves the original index — a subtle bug in one of the
    original z-score implementations was fixed by following this same
    pattern everywhere.
    """
    lower, upper = iqr_bounds(series, multiplier)
    mask = pd.Series(False, index=series.index)
    if pd.isna(lower):
        return mask
    valid = series.notna()
    mask.loc[valid] = (series[valid] < lower) | (series[valid] > upper)
    return mask


# --------------------------------------------------------------------------
# Plotting helper — fixes the "only plt.show(), no headless/batch support"
# gap present in every original script.
# --------------------------------------------------------------------------
def finalize_plot(fig, out_path: Optional[str], show: bool, dpi: int = 150) -> None:
    """Either save a figure to disk, show it interactively, or both.

    If out_path is given, the figure is saved there (parent dirs created
    as needed) regardless of `show`. If show is True (default when no
    out_path is given and not running headless), the figure is displayed.
    Always closes the figure afterward to avoid memory leaks when a
    script generates many plots in a loop.
    """
    import matplotlib.pyplot as plt

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        get_logger("eda_common").info("Saved plot -> %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def use_headless_backend_if_needed() -> None:
    """Switch matplotlib to a non-interactive backend when there is no
    display available (e.g. CI, servers, containers). Prevents scripts
    from crashing with 'no display name and no $DISPLAY' errors.
    """
    import os

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")


# --------------------------------------------------------------------------
# Common CLI / data loading — fixes "no CLI, must edit code to load a real
# CSV" gap present in every original script.
# --------------------------------------------------------------------------
def build_base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Path to a CSV file to analyze. If omitted, a synthetic "
        "demo dataset is generated so the script still runs out of the box.",
    )
    parser.add_argument(
        "--output-dir",
        default="eda_output",
        help="Directory to write reports/plots to (default: ./eda_output)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows (still saves to --output-dir).",
    )
    return parser


def load_dataframe(input_csv: Optional[str], demo_factory) -> pd.DataFrame:
    """Load a user-supplied CSV, or fall back to a synthetic demo dataset.

    demo_factory: a zero-arg callable returning a demo DataFrame, so each
    script can keep its own illustrative example data.
    """
    log = get_logger("eda_common")
    if input_csv is None:
        log.info("No input CSV given — using synthetic demo dataset.")
        return demo_factory()

    path = Path(input_csv)
    if not path.exists():
        log.error("File not found: %s", path)
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() != ".csv":
        log.warning("Expected a .csv file, got '%s' — attempting to read anyway.", path.suffix)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Failed to read '%s': %s", path, exc)
        raise

    if df.empty:
        raise ValueError(f"'{path}' loaded but contains 0 rows.")
    log.info("Loaded '%s' -> shape=%s", path, df.shape)
    return df
