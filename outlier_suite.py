"""
Outlier Detection and Analysis Suite (fixed)

Detects outliers using multiple methods and provides comprehensive
analysis of their impact and patterns.

Fixes applied vs. the original version (see also eda_common.py):
- Added `treat_outliers()` (cap/winsorize, remove, impute) — the
  original script only ever detected outliers with no way to act.
- Replaced a bare `except:` in Mahalanobis detection with a narrowed,
  logged exception handler.
- `fillna(mean)` before Isolation Forest / Mahalanobis now logs how
  many values were imputed this way (previously silent).
- Consensus threshold ("flagged by >= 2 methods") is now configurable
  instead of hardcoded.
- `plot_multivariate_outliers` now explicitly logs when extra columns
  are used for detection but not shown in the 2D scatter.
- IQR logic delegates to the shared `eda_common.iqr_outlier_mask`.
- BUG FIX (visual): plots use `constrained_layout=True` via matplotlib
  directly (fixed 2x2/1x2 grids here, so a fixed figsize is fine, but
  layout negotiation is now automatic rather than a single
  `tight_layout()` call at the end).
- Multi-format input and split plots/ vs reports/ output directories.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from eda_common import (
    build_base_arg_parser,
    finalize_plot,
    get_logger,
    get_output_dirs,
    iqr_bounds,
    iqr_outlier_mask,
    load_dataframe,
    use_headless_backend_if_needed,
)

use_headless_backend_if_needed()
import matplotlib.pyplot as plt  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

log = get_logger("outlier_suite")


class OutlierSuite:
    def __init__(self, df: pd.DataFrame, consensus_min_methods: int = 2):
        if df is None or df.empty:
            raise ValueError("OutlierSuite requires a non-empty DataFrame.")
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.consensus_min_methods = consensus_min_methods

    # ------------------------------------------------------------------
    def detect_iqr_outliers(self, column: str, multiplier: float = 1.5) -> pd.Series:
        return iqr_outlier_mask(self.df[column], multiplier)

    def detect_zscore_outliers(self, column: str, threshold: float = 3) -> pd.Series:
        series = self.df[column]
        non_null = series.dropna()
        mask = pd.Series(False, index=series.index)
        if non_null.empty or non_null.std(ddof=0) == 0:
            return mask
        z_scores = np.abs(stats.zscore(non_null))
        mask.loc[non_null.index] = z_scores > threshold
        return mask

    def detect_modified_zscore_outliers(self, column: str, threshold: float = 3.5) -> pd.Series:
        series = self.df[column]
        median = series.median()
        mad = np.median(np.abs(series.dropna() - median))
        if mad == 0:
            return pd.Series(False, index=self.df.index)
        modified_z = 0.6745 * (series - median) / mad
        return (np.abs(modified_z) > threshold).fillna(False)

    def _fill_for_multivariate(self, cols: List[str]) -> pd.DataFrame:
        X = self.df[cols]
        n_missing = int(X.isna().sum().sum())
        if n_missing:
            log.warning(
                "Imputing %d missing values with column means before multivariate "
                "outlier detection on %s — this can under-flag outliers near the mean.",
                n_missing, cols,
            )
        return X.fillna(X.mean())

    def detect_isolation_forest_outliers(self, columns: Optional[List[str]] = None,
                                          contamination: float = 0.1) -> pd.Series:
        cols = columns or self.numeric_cols
        if not cols:
            return pd.Series(False, index=self.df.index)
        X = self._fill_for_multivariate(cols)
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = iso_forest.fit_predict(X)
        return pd.Series(predictions == -1, index=self.df.index)

    def detect_mahalanobis_outliers(self, columns: Optional[List[str]] = None,
                                     contamination: float = 0.1) -> pd.Series:
        cols = columns or self.numeric_cols
        if len(cols) < 2:
            return pd.Series(False, index=self.df.index)
        X = self._fill_for_multivariate(cols)
        try:
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X)
            return pd.Series(predictions == -1, index=self.df.index)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.error("Mahalanobis/EllipticEnvelope fit failed for columns %s: %s", cols, exc)
            return pd.Series(False, index=self.df.index)

    # ------------------------------------------------------------------
    def analyze_all_methods(self, column: str, iqr_mult: float = 1.5, z_threshold: float = 3,
                             modified_z_threshold: float = 3.5) -> pd.DataFrame:
        results = pd.DataFrame(index=self.df.index)
        results["iqr"] = self.detect_iqr_outliers(column, iqr_mult)
        results["zscore"] = self.detect_zscore_outliers(column, z_threshold)
        results["modified_zscore"] = self.detect_modified_zscore_outliers(column, modified_z_threshold)
        results["consensus"] = results.sum(axis=1) >= self.consensus_min_methods
        return results

    def get_outlier_summary(self, column: str) -> Dict:
        analysis = self.analyze_all_methods(column)
        return {
            "column": column,
            "total_outliers_iqr": int(analysis["iqr"].sum()),
            "total_outliers_zscore": int(analysis["zscore"].sum()),
            "total_outliers_modified_zscore": int(analysis["modified_zscore"].sum()),
            "consensus_outliers": int(analysis["consensus"].sum()),
            "percentage_consensus": round(analysis["consensus"].mean() * 100, 2),
        }

    def compare_methods_all_columns(self) -> pd.DataFrame:
        return pd.DataFrame([self.get_outlier_summary(c) for c in self.numeric_cols])

    def analyze_outlier_impact(self, column: str) -> Dict:
        analysis = self.analyze_all_methods(column)
        consensus_outliers = analysis["consensus"]
        data_with = self.df[column]
        data_without = self.df.loc[~consensus_outliers, column]
        return {
            "column": column,
            "mean_with_outliers": round(data_with.mean(), 4),
            "mean_without_outliers": round(data_without.mean(), 4),
            "mean_difference": round(data_with.mean() - data_without.mean(), 4),
            "median_with_outliers": round(data_with.median(), 4),
            "median_without_outliers": round(data_without.median(), 4),
            "std_with_outliers": round(data_with.std(), 4),
            "std_without_outliers": round(data_without.std(), 4),
        }

    # ------------------------------------------------------------------
    def treat_outliers(self, column: str, strategy: str = "cap", method: str = "iqr",
                        iqr_mult: float = 1.5) -> pd.Series:
        """Return a *new* Series with outliers treated.

        strategy: 'cap' (winsorize to the bounds), 'remove' (set to NaN),
                  or 'impute' (replace with the column median).
        method: which detector's mask to use ('iqr', 'zscore', 'modified_zscore', 'consensus').
        """
        if strategy not in {"cap", "remove", "impute"}:
            raise ValueError("strategy must be one of: cap, remove, impute")

        series = self.df[column].copy()
        if method == "iqr":
            mask = self.detect_iqr_outliers(column, iqr_mult)
        else:
            analysis = self.analyze_all_methods(column, iqr_mult=iqr_mult)
            if method not in analysis.columns:
                raise ValueError(f"Unknown method '{method}'")
            mask = analysis[method]

        n_flagged = int(mask.sum())
        log.info("treat_outliers(%s, strategy=%s, method=%s): %d/%d rows flagged",
                  column, strategy, method, n_flagged, len(series))

        if strategy == "cap":
            lower, upper = iqr_bounds(series, iqr_mult)
            series = series.clip(lower=lower, upper=upper)
        elif strategy == "remove":
            series.loc[mask] = np.nan
        elif strategy == "impute":
            median = series.median()
            series.loc[mask] = median

        return series

    # ------------------------------------------------------------------
    def plot_outlier_comparison(self, column: str, out_path: Optional[str] = None, show: bool = True):
        analysis = self.analyze_all_methods(column)
        data = self.df[column].copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        methods = ["iqr", "zscore", "modified_zscore", "consensus"]
        titles = ["IQR Method", "Z-Score Method", "Modified Z-Score",
                  f"Consensus (>= {self.consensus_min_methods} methods)"]

        for ax, method, title in zip(axes.flatten(), methods, titles):
            outliers = analysis[method]
            ax.scatter(data.index[~outliers], data[~outliers], c="blue", alpha=0.5, s=20, label="Normal")
            ax.scatter(data.index[outliers], data[outliers], c="red", alpha=0.7, s=50, label="Outlier")
            ax.set_title(f"{title}\n{outliers.sum()} outliers ({outliers.mean() * 100:.1f}%)")
            ax.set_xlabel("Index")
            ax.set_ylabel(column)
            ax.legend()
            ax.grid(True, alpha=0.3)

        finalize_plot(fig, out_path, show)

    def plot_multivariate_outliers(self, columns: Optional[List[str]] = None,
                                    out_path: Optional[str] = None, show: bool = True):
        cols = columns or self.numeric_cols[:3]
        if len(cols) < 2:
            log.warning("Need at least 2 columns for multivariate outlier detection")
            return
        if len(cols) > 2:
            log.info("plot_multivariate_outliers: detection uses all %d columns %s, "
                     "but the 2D scatter only visualizes '%s' vs '%s'.",
                     len(cols), cols, cols[0], cols[1])

        iso_outliers = self.detect_isolation_forest_outliers(cols)
        maha_outliers = self.detect_mahalanobis_outliers(cols)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        for ax, outliers, title in zip(
            axes, [iso_outliers, maha_outliers], ["Isolation Forest", "Mahalanobis Distance"]
        ):
            ax.scatter(self.df.loc[~outliers, cols[0]], self.df.loc[~outliers, cols[1]],
                       c="blue", alpha=0.5, s=20, label="Normal")
            ax.scatter(self.df.loc[outliers, cols[0]], self.df.loc[outliers, cols[1]],
                       c="red", alpha=0.7, s=50, label="Outlier")
            ax.set_title(f"{title}\n{outliers.sum()} outliers")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.legend()
            ax.grid(True, alpha=0.3)

        finalize_plot(fig, out_path, show)


# --------------------------------------------------------------------------
def _demo_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    normal_data = np.random.normal(50, 10, n - 30)
    outlier_data = np.random.uniform(120, 150, 30)
    df = pd.DataFrame({
        "feature1": np.concatenate([normal_data, outlier_data]),
        "feature2": np.random.normal(100, 15, n),
        "feature3": np.random.exponential(20, n),
        "feature4": np.random.uniform(0, 100, n),
    })
    df.loc[np.random.choice(df.index, 10, replace=False), "feature2"] = np.random.uniform(200, 250, 10)
    return df


def main(argv=None) -> int:
    parser = build_base_arg_parser("Multi-method outlier detection & treatment for any dataset.")
    parser.add_argument("--consensus-min-methods", type=int, default=2)
    parser.add_argument("--treat-column", default=None, help="Column to demonstrate outlier treatment on.")
    parser.add_argument("--treat-strategy", default="cap", choices=["cap", "remove", "impute"])
    args = parser.parse_args(argv)
    show = not args.no_show

    df = load_dataframe(args, _demo_dataframe)
    suite = OutlierSuite(df, consensus_min_methods=args.consensus_min_methods)

    plots_dir, reports_dir = get_output_dirs(args.output_dir, "outlier_suite")

    summary = suite.compare_methods_all_columns()
    summary.to_csv(reports_dir / "outlier_summary.csv", index=False)
    print("Outlier Detection Summary:")
    print(summary.to_string(index=False))

    first_col = suite.numeric_cols[0]
    impact = suite.analyze_outlier_impact(first_col)
    print(f"\nOutlier Impact Analysis for '{first_col}':")
    for k, v in impact.items():
        print(f"{k}: {v}")

    treat_col = args.treat_column or first_col
    treated = suite.treat_outliers(treat_col, strategy=args.treat_strategy)
    pd.DataFrame({treat_col: df[treat_col], f"{treat_col}_treated": treated}).to_csv(
        reports_dir / f"{treat_col}_treated.csv", index=False
    )

    suite.plot_outlier_comparison(first_col, out_path=str(plots_dir / "outlier_comparison.png"), show=show)
    if len(suite.numeric_cols) >= 2:
        suite.plot_multivariate_outliers(suite.numeric_cols[:2],
                                          out_path=str(plots_dir / "multivariate_outliers.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
