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
    create_grid,
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
        mean_with, mean_without = data_with.mean(), data_without.mean()
        # Relative shift lets columns on very different scales (e.g. age vs.
        # income) be compared and ranked fairly — see analyze_outlier_impact_all_columns.
        relative_shift_pct = (
            abs(mean_with - mean_without) / abs(mean_without) * 100 if mean_without else np.nan
        )
        return {
            "column": column,
            "consensus_outlier_count": int(consensus_outliers.sum()),
            "mean_with_outliers": round(mean_with, 4),
            "mean_without_outliers": round(mean_without, 4),
            "mean_difference": round(mean_with - mean_without, 4),
            "relative_mean_shift_pct": round(relative_shift_pct, 2) if pd.notna(relative_shift_pct) else None,
            "median_with_outliers": round(data_with.median(), 4),
            "median_without_outliers": round(data_without.median(), 4),
            "std_with_outliers": round(data_with.std(), 4),
            "std_without_outliers": round(data_without.std(), 4),
        }

    def analyze_outlier_impact_all_columns(self) -> pd.DataFrame:
        """Impact analysis for every numeric column, ranked by relative
        mean shift so the columns most distorted by outliers surface
        first. Previously only the first numeric column was analyzed.
        """
        rows = [self.analyze_outlier_impact(c) for c in self.numeric_cols]
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        return result.sort_values("relative_mean_shift_pct", ascending=False, na_position="last").reset_index(drop=True)

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

    def treat_outliers_all_columns(self, strategy: str = "cap", method: str = "iqr",
                                    iqr_mult: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply treat_outliers() to every numeric column at once.

        Returns (treated_df, summary_df). treated_df is a copy of the full
        DataFrame with every numeric column treated; summary_df reports how
        many rows were flagged per column, ordered by the largest share
        flagged first (the columns treatment affects most).
        """
        treated_df = self.df.copy()
        summary_rows = []
        for col in self.numeric_cols:
            treated_series = self.treat_outliers(col, strategy=strategy, method=method, iqr_mult=iqr_mult)
            treated_df[col] = treated_series
            if method == "iqr":
                mask = self.detect_iqr_outliers(col, iqr_mult)
            else:
                mask = self.analyze_all_methods(col, iqr_mult=iqr_mult)[method]
            summary_rows.append({
                "column": col,
                "rows_flagged": int(mask.sum()),
                "pct_flagged": round(mask.mean() * 100, 2),
                "strategy": strategy,
                "method": method,
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("pct_flagged", ascending=False).reset_index(drop=True)
        return treated_df, summary_df

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

    def plot_consensus_outliers_all_columns(self, out_path: Optional[str] = None, show: bool = True):
        """One scatter subplot per numeric column, showing consensus
        outliers. Companion to `plot_outlier_comparison` (which drills
        into all 4 detection methods for a *single* column): this gives
        the "all columns" overview that was previously missing — the
        plots used to always show only the first column regardless of
        how many columns were actually analyzed/treated.
        """
        cols = self.numeric_cols
        if not cols:
            log.warning("No numeric columns to plot")
            return
        n_cols = min(3, len(cols))
        fig, axes, _ = create_grid(len(cols), n_cols, per_row_height=3.4)

        for idx, col in enumerate(cols):
            ax = axes[idx]
            outliers = self.analyze_all_methods(col)["consensus"]
            data = self.df[col]
            ax.scatter(data.index[~outliers], data[~outliers], c="blue", alpha=0.5, s=15, label="Normal")
            ax.scatter(data.index[outliers], data[outliers], c="red", alpha=0.7, s=35, label="Outlier")
            ax.set_title(f"{col}\n{outliers.sum()} outliers ({outliers.mean() * 100:.1f}%)", fontsize=10)
            ax.set_xlabel("Index", fontsize=9)
            ax.set_ylabel(col, fontsize=9)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(cols), len(axes)):
            axes[idx].axis("off")
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
    parser.add_argument("--treat-column", default="all",
                         help="Column to treat, or 'all' (default) to treat every numeric column at once.")
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

    # FIX: impact analysis now covers every numeric column (previously only
    # the first one), ranked by relative mean shift so the most-affected
    # columns are easy to spot at the top.
    impact_all = suite.analyze_outlier_impact_all_columns()
    if not impact_all.empty:
        impact_all.to_csv(reports_dir / "outlier_impact_all_columns.csv", index=False)
        print("\nOutlier Impact Analysis (all numeric columns, ranked by relative mean shift):")
        print(impact_all.to_string(index=False))

    if args.treat_column == "all":
        treated_df, treat_summary = suite.treat_outliers_all_columns(strategy=args.treat_strategy)
        treat_summary.to_csv(reports_dir / "treatment_summary_all_columns.csv", index=False)
        treated_df.to_csv(reports_dir / "dataset_treated_all_columns.csv", index=False)
        print(f"\nTreatment summary ({args.treat_strategy}, all columns):")
        print(treat_summary.to_string(index=False))
    else:
        treat_col = args.treat_column
        if treat_col not in suite.numeric_cols:
            log.error("'--treat-column %s' is not a numeric column. Available: %s",
                      treat_col, suite.numeric_cols)
            return 1
        treated = suite.treat_outliers(treat_col, strategy=args.treat_strategy)
        pd.DataFrame({treat_col: df[treat_col], f"{treat_col}_treated": treated}).to_csv(
            reports_dir / f"{treat_col}_treated.csv", index=False
        )

    if args.treat_column == "all":
        # FIX: previously plot_outlier_comparison() always plotted only
        # `first_col`, even in "all columns" mode. Now the plot set matches
        # the report set: one consensus-outlier subplot per numeric column.
        suite.plot_consensus_outliers_all_columns(
            out_path=str(plots_dir / "consensus_outliers_all_columns.png"), show=show)
    else:
        suite.plot_outlier_comparison(treat_col, out_path=str(plots_dir / f"outlier_comparison_{treat_col}.png"),
                                       show=show)

    if len(suite.numeric_cols) >= 2:
        suite.plot_multivariate_outliers(suite.numeric_cols[:2],
                                          out_path=str(plots_dir / "multivariate_outliers.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
