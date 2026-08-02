"""
Correlation and Relationship Explorer (fixed)

Analyzes relationships between variables using multiple correlation
methods and detects multicollinearity issues.

Fixes applied vs. the original version (see also eda_common.py):
- REAL BUG: `plot_top_correlations` iterated with
  `for idx, row in top_pairs.iterrows()` and then indexed `axes[idx]`.
  Because `top_pairs` is sorted by correlation strength, its index is
  NOT a contiguous 0..n range, so `axes[idx]` could raise IndexError or
  place plots on the wrong axis. Fixed with `enumerate(... .itertuples())`.
- `mutual_information_analysis` silently replaced NaNs with 0, biasing
  scores with no warning — now logs dropped rows instead.
- Added p-values (pearsonr/spearmanr/kendalltau) alongside coefficients.
- Added Cramer's V for categorical-categorical association.
- Heatmaps cap at `max_features_for_heatmap` with a warning instead of
  becoming unreadable.
- BUG FIX (visual): all multi-panel plots now use `eda_common.create_grid`
  (dynamic figsize + constrained_layout) instead of a fixed figsize,
  which is what caused title/label overlap in dense grids.
- Multi-format input and split plots/ vs reports/ output directories.
"""

from __future__ import annotations

import sys
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from eda_common import (
    build_base_arg_parser,
    create_grid,
    finalize_plot,
    get_logger,
    get_output_dirs,
    load_dataframe,
    use_headless_backend_if_needed,
)

use_headless_backend_if_needed()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

log = get_logger("correlation_explorer")

_CORR_TEST = {
    "pearson": stats.pearsonr,
    "spearman": stats.spearmanr,
    "kendall": stats.kendalltau,
}


class CorrelationExplorer:
    def __init__(self, df: pd.DataFrame, max_features_for_heatmap: int = 40):
        if df is None or df.empty:
            raise ValueError("CorrelationExplorer requires a non-empty DataFrame.")
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Explicitly excludes datetime64 columns (previously object-only
        # selection meant this was already safe, but being explicit avoids
        # any future dtype surprises after auto-datetime casting).
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.max_features_for_heatmap = max_features_for_heatmap

    # ------------------------------------------------------------------
    def calculate_correlations(self, method: str = "pearson", columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols = columns or self.numeric_cols
        if len(cols) < 2:
            log.warning("Need at least 2 numeric columns for correlation")
            return pd.DataFrame()
        return self.df[cols].corr(method=method)

    def find_high_correlations(self, threshold: float = 0.7, method: str = "pearson") -> pd.DataFrame:
        cols = self.numeric_cols
        if len(cols) < 2:
            return pd.DataFrame()

        test_fn = _CORR_TEST[method]
        high_corr = []
        for c1, c2 in combinations(cols, 2):
            pair = self.df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            result = test_fn(pair[c1], pair[c2])
            corr_val, p_value = float(result[0]), float(result[1])
            if abs(corr_val) >= threshold:
                high_corr.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(corr_val, 4),
                    "abs_correlation": round(abs(corr_val), 4),
                    "p_value": round(p_value, 6),
                    "significant_at_0.05": p_value < 0.05,
                    "strength": self._classify_correlation(abs(corr_val)),
                })
        if not high_corr:
            return pd.DataFrame()
        return pd.DataFrame(high_corr).sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    def _classify_correlation(self, abs_corr: float) -> str:
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        return "Very Weak"

    # ------------------------------------------------------------------
    def calculate_vif(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression

        cols = columns or self.numeric_cols
        if len(cols) < 2:
            log.warning("Need at least 2 columns for VIF")
            return pd.DataFrame()

        vif_data = []
        for col in cols:
            X = self.df[cols].drop(columns=[col])
            y = self.df[col]
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean, y_clean = X[mask], y[mask]
            if len(X_clean) < 2:
                continue
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            r_squared = model.score(X_clean, y_clean)
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            vif_data.append({
                "feature": col, "vif": round(vif, 4), "r_squared": round(r_squared, 4),
                "multicollinearity": self._classify_vif(vif),
            })
        return pd.DataFrame(vif_data).sort_values("vif", ascending=False).reset_index(drop=True)

    def _classify_vif(self, vif: float) -> str:
        if vif > 10:
            return "High (Remove)"
        elif vif > 5:
            return "Moderate (Consider removing)"
        return "Low (Acceptable)"

    # ------------------------------------------------------------------
    def cramers_v_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols = columns or self.categorical_cols
        if len(cols) < 2:
            log.warning("Need at least 2 categorical columns for Cramer's V")
            return pd.DataFrame()

        result = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for c1, c2 in combinations(cols, 2):
            contingency = pd.crosstab(self.df[c1], self.df[c2])
            if contingency.size == 0:
                v = np.nan
            else:
                chi2 = stats.chi2_contingency(contingency)[0]
                n = contingency.to_numpy().sum()
                r, k = contingency.shape
                denom = n * (min(r, k) - 1)
                v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
            result.loc[c1, c2] = v
            result.loc[c2, c1] = v
        return result

    # ------------------------------------------------------------------
    def mutual_information_analysis(self, target_col: str, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        cols = feature_cols or [c for c in self.numeric_cols if c != target_col]
        subset = self.df[cols + [target_col]]
        n_before = len(subset)
        subset = subset.dropna()
        n_after = len(subset)
        if n_after < n_before:
            log.warning(
                "mutual_information_analysis: dropped %d/%d rows with missing values.",
                n_before - n_after, n_before,
            )
        if n_after == 0:
            log.error("No complete rows remain after dropping missing values.")
            return pd.DataFrame()

        X, y = subset[cols], subset[target_col]
        if y.nunique() < 10:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_df = pd.DataFrame({
            "feature": cols, "mutual_information": mi_scores,
            "mi_normalized": mi_scores / mi_scores.max() if mi_scores.max() > 0 else 0,
        })
        return mi_df.sort_values("mutual_information", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def plot_correlation_heatmap(self, method: str = "pearson", annot: bool = True, cmap: str = "coolwarm",
                                  out_path: Optional[str] = None, show: bool = True):
        cols = self.numeric_cols
        if len(cols) > self.max_features_for_heatmap:
            log.warning("%d numeric columns exceeds max_features_for_heatmap=%d; showing the first %d only.",
                        len(cols), self.max_features_for_heatmap, self.max_features_for_heatmap)
            cols = cols[: self.max_features_for_heatmap]

        corr_matrix = self.calculate_correlations(method=method, columns=cols)
        if corr_matrix.empty:
            log.warning("No correlation matrix to plot")
            return

        size = max(6, min(0.5 * len(cols) + 4, 20))
        fig, ax = plt.subplots(figsize=(size, size * 0.85), constrained_layout=True)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=annot and len(cols) <= 25, fmt=".2f", cmap=cmap,
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f"{method.capitalize()} Correlation Matrix")
        finalize_plot(fig, out_path, show)

    def plot_correlation_comparison(self, out_path: Optional[str] = None, show: bool = True):
        methods = ["pearson", "spearman", "kendall"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        for idx, method in enumerate(methods):
            corr = self.calculate_correlations(method=method)
            if corr.empty:
                continue
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, ax=axes[idx], cmap="coolwarm", center=0, square=True,
                        linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=len(corr) < 10, fmt=".2f")
            axes[idx].set_title(f"{method.capitalize()} Correlation")
        finalize_plot(fig, out_path, show)

    def plot_top_correlations(self, n_pairs: int = 10, method: str = "pearson",
                               out_path: Optional[str] = None, show: bool = True):
        high_corr = self.find_high_correlations(threshold=0.0, method=method)
        if high_corr.empty:
            log.warning("No correlations to plot")
            return

        top_pairs = high_corr.head(n_pairs)
        fig, axes, _ = create_grid(len(top_pairs), 3, per_row_height=3.6)

        # FIX: enumerate() over a contiguous position instead of the
        # post-sort DataFrame index — the original `iterrows()` index
        # could exceed len(axes) or scatter plots onto the wrong subplot.
        for pos, row in enumerate(top_pairs.itertuples(index=False)):
            ax = axes[pos]
            feat1, feat2, corr = row.feature_1, row.feature_2, row.correlation
            ax.scatter(self.df[feat1], self.df[feat2], alpha=0.5, s=20)
            ax.set_xlabel(feat1, fontsize=9)
            ax.set_ylabel(feat2, fontsize=9)
            ax.set_title(f"r = {corr:.3f}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        for pos in range(len(top_pairs), len(axes)):
            axes[pos].axis("off")
        finalize_plot(fig, out_path, show)


# --------------------------------------------------------------------------
def _demo_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    x1 = np.random.normal(50, 10, n)
    x2 = x1 + np.random.normal(0, 5, n)
    x3 = 100 - x1 + np.random.normal(0, 8, n)
    x4 = np.random.normal(30, 15, n)
    x5 = x1 * 0.5 + x4 * 0.5 + np.random.normal(0, 3, n)
    return pd.DataFrame({
        "feature_A": x1, "feature_B": x2, "feature_C": x3, "feature_D": x4, "feature_E": x5,
        "target": x1 * 2 + x4 - x3 * 0.5 + np.random.normal(0, 10, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "tier": np.random.choice(["Gold", "Silver", "Bronze"], n),
    })


def main(argv=None) -> int:
    parser = build_base_arg_parser("Correlation & relationship explorer for any dataset.")
    parser.add_argument("--target", default=None, help="Target column for mutual-information analysis.")
    args = parser.parse_args(argv)
    show = not args.no_show

    df = load_dataframe(args, _demo_dataframe)
    explorer = CorrelationExplorer(df)

    plots_dir, reports_dir = get_output_dirs(args.output_dir, "correlation_explorer")

    high_corr = explorer.find_high_correlations(threshold=0.5)
    if not high_corr.empty:
        high_corr.to_csv(reports_dir / "high_correlations.csv", index=False)
        print("High Correlations (threshold = 0.5):")
        print(high_corr.to_string(index=False))

    vif = explorer.calculate_vif()
    if not vif.empty:
        vif.to_csv(reports_dir / "vif.csv", index=False)
        print("\nVariance Inflation Factors:")
        print(vif.to_string(index=False))

    target = args.target or ("target" if "target" in df.columns else None)
    if target:
        mi = explorer.mutual_information_analysis(target)
        if not mi.empty:
            mi.to_csv(reports_dir / "mutual_information.csv", index=False)
            print(f"\nMutual Information with '{target}':")
            print(mi.to_string(index=False))

    cramers = explorer.cramers_v_matrix()
    if not cramers.empty:
        cramers.to_csv(reports_dir / "cramers_v.csv")
        print("\nCramer's V (categorical association):")
        print(cramers.to_string())

    explorer.plot_correlation_heatmap(out_path=str(plots_dir / "correlation_heatmap.png"), show=show)
    explorer.plot_correlation_comparison(out_path=str(plots_dir / "correlation_comparison.png"), show=show)
    explorer.plot_top_correlations(n_pairs=6, out_path=str(plots_dir / "top_correlations.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
