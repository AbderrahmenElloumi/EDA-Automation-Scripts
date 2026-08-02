"""
Distribution Analyzer and Visualizer (fixed)

Automatically generates comprehensive distribution visualizations
and statistical tests for all features in a dataset.

Fixes applied vs. the original version:
- `Dict[str, any]` used the Python builtin `any` instead of `typing.Any`
  (a real typing bug, would fail static type checking / mypy).
- The Shapiro-Wilk normality test sampled the data with
  `series.sample(...)` without a fixed random_state, so results were
  not reproducible between runs on the same data. Now seeded.
- All `plot_*` methods only ever called `plt.show()`, making the script
  unusable headless/in batch pipelines. They now accept `out_path` and
  use the shared `finalize_plot` helper (save-to-file and/or show).
- IQR outlier counting (duplicated logic vs. outlier_suite.py) now
  delegates to `eda_common.iqr_outlier_mask`.
- Added a categorical distribution *report* (previously only a plot
  existed for categoricals — no tabular summary was ever produced).
- Added CLI support to run against a real CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eda_common import (
    build_base_arg_parser,
    finalize_plot,
    get_logger,
    iqr_outlier_mask,
    load_dataframe,
    use_headless_backend_if_needed,
)

use_headless_backend_if_needed()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

log = get_logger("distribution_analyzer")

RANDOM_STATE = 42  # FIX: fixed seed so normality tests are reproducible


class DistributionAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DistributionAnalyzer requires a non-empty DataFrame.")
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ------------------------------------------------------------------
    def analyze_numeric_distribution(self, col: str) -> Dict[str, Any]:
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"column": col, "error": "No non-null values"}

        mean, median, std = series.mean(), series.median(), series.std()
        mode = series.mode()[0] if len(series.mode()) > 0 else np.nan
        skewness, kurtosis = series.skew(), series.kurtosis()

        # FIX: seeded sampling for reproducibility
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

        # FIX: delegate to the shared IQR implementation instead of a
        # locally-duplicated calculation.
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

    # FIX: new — categorical columns previously had a plot but no report table.
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
        }

    def generate_distribution_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.analyze_numeric_distribution(c) for c in self.numeric_cols])

    def generate_categorical_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.analyze_categorical_distribution(c) for c in self.categorical_cols])

    # ------------------------------------------------------------------
    def _grid(self, n_items: int, n_cols: int, figsize: Tuple[int, int]):
        n_rows = (n_items + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])
        return fig, axes

    def plot_numeric_distributions(self, columns: Optional[List[str]] = None, max_cols: int = 10,
                                    figsize: Tuple[int, int] = (15, 12), out_path: Optional[str] = None,
                                    show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        fig, axes = self._grid(len(cols_to_plot), min(3, len(cols_to_plot)), figsize)

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
            except Exception as exc:
                log.debug("KDE failed for %s: %s", col, exc)
            ax.set_title(f"{col}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}", fontsize=10)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    def plot_boxplots(self, columns: Optional[List[str]] = None, max_cols: int = 10,
                       figsize: Tuple[int, int] = (15, 8), out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        fig, axes = self._grid(len(cols_to_plot), min(4, len(cols_to_plot)), figsize)

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
            ax.set_title(f"{col}\n{outliers} outliers ({outliers / len(data) * 100:.1f}%)")
            ax.set_ylabel("Value")

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    def plot_categorical_distributions(self, columns: Optional[List[str]] = None, max_categories: int = 20,
                                        figsize: Tuple[int, int] = (15, 10), out_path: Optional[str] = None,
                                        show: bool = True):
        cols_to_plot = columns or self.categorical_cols
        if not cols_to_plot:
            log.warning("No categorical columns to plot")
            return
        fig, axes = self._grid(len(cols_to_plot), min(2, len(cols_to_plot)), figsize)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            value_counts = self.df[col].value_counts().head(max_categories)
            if len(value_counts) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                continue
            value_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title(f"{col}\n{self.df[col].nunique()} unique values")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    def plot_qq_plots(self, columns: Optional[List[str]] = None, max_cols: int = 9,
                       figsize: Tuple[int, int] = (12, 10), out_path: Optional[str] = None, show: bool = True):
        cols_to_plot = (columns or self.numeric_cols)[:max_cols]
        if not cols_to_plot:
            log.warning("No numeric columns to plot")
            return
        fig, axes = self._grid(len(cols_to_plot), 3, figsize)

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = self.df[col].dropna()
            if len(data) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
                ax.set_title(col)
                continue
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"{col}\nQ-Q Plot")
            ax.grid(True, alpha=0.3)

        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
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
    })


def main(argv=None) -> int:
    parser = build_base_arg_parser("Distribution analysis + visualization for any CSV dataset.")
    args = parser.parse_args(argv)
    show = not args.no_show

    df = load_dataframe(args.input_csv, _demo_dataframe)
    analyzer = DistributionAnalyzer(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = analyzer.generate_distribution_report()
    if not report.empty:
        report.to_csv(out_dir / "numeric_distribution_report.csv", index=False)
        print("Numeric Distribution Report:")
        print(report.to_string(index=False))

    cat_report = analyzer.generate_categorical_report()
    if not cat_report.empty:
        cat_report.to_csv(out_dir / "categorical_distribution_report.csv", index=False)
        print("\nCategorical Distribution Report:")
        print(cat_report.to_string(index=False))

    analyzer.plot_numeric_distributions(out_path=str(out_dir / "numeric_distributions.png"), show=show)
    analyzer.plot_boxplots(out_path=str(out_dir / "boxplots.png"), show=show)
    analyzer.plot_qq_plots(out_path=str(out_dir / "qq_plots.png"), show=show)
    analyzer.plot_categorical_distributions(out_path=str(out_dir / "categorical_distributions.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
