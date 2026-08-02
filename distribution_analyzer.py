"""
Distribution Analyzer and Visualizer (fixed)

Automatically generates comprehensive distribution visualizations
and statistical tests for all features in a dataset.

Fixes applied vs. the original version (see also eda_common.py):
- BUG FIX (visual): the original `plot_categorical_distributions` used a
  fixed figsize regardless of how many columns/rows were being plotted,
  and lumped date-like string columns in with genuine categorical
  columns. With a column like "Date recrutement" (1500+ unique values),
  this produced a near-unreadable bar chart with hundreds of thin bars,
  and — because the figure height never grew with the number of rows —
  titles overlapped the axes above them and rotated tick labels collided
  between subplots. Fixed by:
    1. Date-like string columns are now auto-detected and cast to real
       datetime64 columns at load time (see `eda_common.infer_and_cast_datetime_columns`),
       so they're routed to their own `plot_datetime_distributions` instead
       of the categorical bar-chart grid.
    2. Remaining high-cardinality categorical columns are bucketed into
       "top N + Other" (`eda_common.bucket_top_n`) instead of plotting
       every category.
    3. Grid figures are built with `eda_common.create_grid`, which scales
       figure height with the number of rows and uses
       `constrained_layout=True` so titles/labels never overlap.
- `Dict[str, any]` used the Python builtin `any` instead of `typing.Any`.
- Shapiro-Wilk normality test now uses a fixed random_state (reproducible).
- All `plot_*` methods can save to file (headless-safe), not just plt.show().
- IQR outlier counting delegates to the shared `eda_common.iqr_outlier_mask`.
- Added a categorical distribution *report* (previously only a plot existed).
- Multi-format input (CSV/TSV/TXT/Excel/Parquet/JSON/XML/DB) and split
  plots/ vs reports/ output directories.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from eda_common import (
    bucket_top_n,
    build_base_arg_parser,
    create_grid,
    finalize_plot,
    get_logger,
    get_output_dirs,
    iqr_outlier_mask,
    load_dataframe,
    truncate_label,
    use_headless_backend_if_needed,
)

use_headless_backend_if_needed()
import matplotlib.pyplot as plt  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

log = get_logger("distribution_analyzer")

RANDOM_STATE = 42  # fixed seed so normality tests are reproducible
MAX_CATEGORIES_PLOTTED = 15  # top-N cap before bucketing the rest into "Other"


class DistributionAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DistributionAnalyzer requires a non-empty DataFrame.")
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # FIX: datetime columns get their own analysis/plot path instead of
        # being lumped into "categorical" (which is what produced the
        # 1500-bar unreadable chart in the original script).
        self.datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    # ------------------------------------------------------------------
    def analyze_numeric_distribution(self, col: str) -> Dict[str, Any]:
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"column": col, "error": "No non-null values"}

        mean, median, std = series.mean(), series.median(), series.std()
        mode = series.mode()[0] if len(series.mode()) > 0 else np.nan
        skewness, kurtosis = series.skew(), series.kurtosis()

        if len(series) < 5000:
            sample = series.sample(min(5000, len(series)), random_state=RANDOM_STATE)
            _, p_value = stats.shapiro(sample)
            is_normal = p_value > 0.05
        else:
            sample = series.sample(5000, random_state=RANDOM_STATE)
            result = stats.anderson(sample)
            is_normal = result.statistic < result.critical_values[2]
            p_value = None

        dist_type = self._classify_distribution(skewness, kurtosis, is_normal)
        outlier_mask = iqr_outlier_mask(series)
        outliers = int(outlier_mask.sum())

        return {
            "column": col,
            "count": len(series),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "mode": round(mode, 4) if not pd.isna(mode) else None,
            "std": round(std, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "is_normal": is_normal,
            "normality_p_value": round(p_value, 4) if p_value else None,
            "distribution_type": dist_type,
            "outlier_count": outliers,
            "outlier_percentage": round(outliers / len(series) * 100, 2),
        }

    def _classify_distribution(self, skew: float, kurt: float, is_normal: bool) -> str:
        if is_normal and abs(skew) < 0.5:
            return "Normal"
        elif skew > 1:
            return "Right-skewed (Positive)"
        elif skew < -1:
            return "Left-skewed (Negative)"
        elif abs(skew) < 0.5 and kurt > 3:
            return "Leptokurtic (Heavy-tailed)"
        elif abs(skew) < 0.5 and kurt < 0:
            return "Platykurtic (Light-tailed)"
        return "Approximately symmetric"

    def analyze_categorical_distribution(self, col: str) -> Dict[str, Any]:
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"column": col, "error": "No non-null values"}
        value_counts = series.value_counts()
        top_value = value_counts.index[0]
        top_pct = round(value_counts.iloc[0] / len(series) * 100, 2)
        return {
            "column": col,
            "count": len(series),
            "unique_values": series.nunique(),
            "top_value": top_value,
            "top_value_percentage": top_pct,
            "is_imbalanced": top_pct > 80,
            "is_high_cardinality": series.nunique() > MAX_CATEGORIES_PLOTTED,
        }

    def analyze_datetime_distribution(self, col: str) -> Dict[str, Any]:
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"column": col, "error": "No non-null values"}
        return {
            "column": col,
            "count": len(series),
            "min_date": series.min(),
            "max_date": series.max(),
            "range_days": (series.max() - series.min()).days,
            "most_common_year": int(series.dt.year.mode()[0]),
            "most_common_month": int(series.dt.month.mode()[0]),
        }

    def generate_distribution_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.analyze_numeric_distribution(c) for c in self.numeric_cols])

    def generate_categorical_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.analyze_categorical_distribution(c) for c in self.categorical_cols])

    def generate_datetime_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.analyze_datetime_distribution(c) for c in self.datetime_cols])

    # ------------------------------------------------------------------
    def plot_numeric_distributions(self, columns: Optional[List[str]] = None, max_cols: int = 10,
                                    out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        n_cols = min(3, len(cols_to_plot))
        fig, axes, _ = create_grid(len(cols_to_plot), n_cols)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                continue
            ax.hist(data, bins=30, alpha=0.6, color="skyblue", edgecolor="black", density=True)
            try:
                data_range = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                ax.plot(data_range, kde(data_range), "r-", linewidth=2, label="KDE")
                ax.legend(fontsize=8)
            except Exception as exc:
                log.debug("KDE failed for %s: %s", col, exc)
            ax.set_title(f"{col}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}", fontsize=10)
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.tick_params(labelsize=8)

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        finalize_plot(fig, out_path, show)

    def plot_boxplots(self, columns: Optional[List[str]] = None, max_cols: int = 10,
                       out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        n_cols = min(4, len(cols_to_plot))
        fig, axes, _ = create_grid(len(cols_to_plot), n_cols, per_row_height=3.2)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                continue
            bp = ax.boxplot(data, vert=True, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue")
            outliers = int(iqr_outlier_mask(data).sum())
            ax.set_title(f"{col}\n{outliers} outliers ({outliers / len(data) * 100:.1f}%)", fontsize=10)
            ax.set_ylabel("Value", fontsize=9)
            ax.tick_params(labelsize=8)

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        finalize_plot(fig, out_path, show)

    def plot_categorical_distributions(self, columns: Optional[List[str]] = None,
                                        out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = columns or self.categorical_cols
        if not cols_to_plot:
            log.warning("No categorical columns to plot")
            return
        n_cols = min(2, len(cols_to_plot))
        fig, axes, _ = create_grid(len(cols_to_plot), n_cols, per_row_height=3.6)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            value_counts = self.df[col].value_counts()
            if len(value_counts) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                continue

            # FIX: cap at top-N + "Other" instead of plotting every category
            # (previously produced hundreds of unreadable bars for
            # high-cardinality columns).
            bucketed, other_n, coverage = bucket_top_n(value_counts, n=MAX_CATEGORIES_PLOTTED)
            labels = [truncate_label(i) for i in bucketed.index]
            ax.bar(labels, bucketed.values, color="steelblue", edgecolor="black")

            subtitle = f"{col}\n{self.df[col].nunique()} unique values"
            if other_n:
                subtitle += f" (top {MAX_CATEGORIES_PLOTTED} shown, {coverage * 100:.0f}% of data)"
            ax.set_title(subtitle, fontsize=10)
            ax.set_xlabel("Category", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        finalize_plot(fig, out_path, show)

    def plot_datetime_distributions(self, columns: Optional[List[str]] = None,
                                     out_path: Optional[str] = None, show: bool = True):
        """New: datetime columns are plotted as a timeline histogram
        (counts per month) instead of one bar per unique timestamp.
        """
        cols_to_plot = columns or self.datetime_cols
        if not cols_to_plot:
            log.warning("No datetime columns to plot")
            return
        n_cols = min(2, len(cols_to_plot))
        fig, axes, _ = create_grid(len(cols_to_plot), n_cols, per_row_height=3.4)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                continue
            ax.hist(data, bins=min(30, data.nunique()), color="mediumseagreen", edgecolor="black")
            ax.set_title(f"{col}\n{data.min().date()} to {data.max().date()}", fontsize=10)
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        finalize_plot(fig, out_path, show)

    def plot_qq_plots(self, columns: Optional[List[str]] = None, max_cols: int = 9,
                       out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        fig, axes, _ = create_grid(len(cols_to_plot), 3, per_row_height=3.2)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            if len(data) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
                ax.set_title(col)
                continue
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"{col}\nQ-Q Plot", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        finalize_plot(fig, out_path, show)


# --------------------------------------------------------------------------
def _demo_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "normal": np.random.normal(50, 10, n),
        "right_skewed": np.random.exponential(5, n),
        "left_skewed": 100 - np.random.exponential(5, n),
        "uniform": np.random.uniform(0, 100, n),
        "bimodal": np.concatenate([np.random.normal(30, 5, n // 2), np.random.normal(70, 5, n // 2)]),
        "with_outliers": np.concatenate([np.random.normal(50, 10, n - 50), np.random.uniform(150, 200, 50)]),
        "category": np.random.choice(["A", "B", "C", "D", "E"], n),
        "segment": np.random.choice(["High", "Medium", "Low"], n, p=[0.2, 0.5, 0.3]),
        # High-cardinality date-like column (mirrors the "Date recrutement"
        # example that used to break the categorical plot).
        "signup_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.random.randint(0, 1800, n), unit="D"),
        "employee_id": [f"EMP-{i:05d}" for i in range(n)],
    })


def main(argv=None) -> int:
    parser = build_base_arg_parser("Distribution analysis + visualization for any dataset.")
    args = parser.parse_args(argv)
    show = not args.no_show

    df = load_dataframe(args, _demo_dataframe)
    analyzer = DistributionAnalyzer(df)

    plots_dir, reports_dir = get_output_dirs(args.output_dir, "distribution_analyzer")

    report = analyzer.generate_distribution_report()
    if not report.empty:
        report.to_csv(reports_dir / "numeric_distribution_report.csv", index=False)
        print("Numeric Distribution Report:")
        print(report.to_string(index=False))

    cat_report = analyzer.generate_categorical_report()
    if not cat_report.empty:
        cat_report.to_csv(reports_dir / "categorical_distribution_report.csv", index=False)
        print("\nCategorical Distribution Report:")
        print(cat_report.to_string(index=False))

    dt_report = analyzer.generate_datetime_report()
    if not dt_report.empty:
        dt_report.to_csv(reports_dir / "datetime_distribution_report.csv", index=False)
        print("\nDatetime Distribution Report:")
        print(dt_report.to_string(index=False))

    analyzer.plot_numeric_distributions(out_path=str(plots_dir / "numeric_distributions.png"), show=show)
    analyzer.plot_boxplots(out_path=str(plots_dir / "boxplots.png"), show=show)
    analyzer.plot_qq_plots(out_path=str(plots_dir / "qq_plots.png"), show=show)
    analyzer.plot_categorical_distributions(out_path=str(plots_dir / "categorical_distributions.png"), show=show)
    analyzer.plot_datetime_distributions(out_path=str(plots_dir / "datetime_distributions.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
