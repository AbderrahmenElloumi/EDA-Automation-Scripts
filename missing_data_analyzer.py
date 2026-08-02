"""
Missing Data Pattern Analyzer (fixed)

Analyzes patterns in missing data, classifies missingness mechanisms,
and provides visualization, imputation recommendations, and actual
imputation.

Fixes applied vs. the original version:
- STATISTICAL BUG: `classify_missingness_type` ran up to 10 separate
  hypothesis tests (t-tests / chi-square tests) per column with no
  correction for multiple comparisons, inflating the false-positive
  rate for "this column is MAR" conclusions. Now applies a Bonferroni
  correction to the significance threshold.
- The MCAR/MNAR heuristic was a bare "missing_pct < 5% => MCAR" rule
  with no statistical grounding. It's now clearly labeled as a
  heuristic (not a real Little's MCAR test) and documents that
  limitation instead of presenting it as a definitive classification.
- `recommend_strategy` only ever returned a *string* recommendation —
  no imputation was actually implemented anywhere in the script.
  Added `apply_recommended_imputation()` that performs mean/median/mode/
  predictive imputation matching the recommendation.
- Added CLI support + headless plot saving via the shared helper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eda_common import (
    build_base_arg_parser,
    finalize_plot,
    get_logger,
    load_dataframe,
    use_headless_backend_if_needed,
)

use_headless_backend_if_needed()
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import warnings

warnings.filterwarnings("ignore")

log = get_logger("missing_data_analyzer")


class MissingDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("MissingDataAnalyzer requires a non-empty DataFrame.")
        self.df = df

    # ------------------------------------------------------------------
    def get_missing_summary(self) -> pd.DataFrame:
        missing_data = []
        for col in self.df.columns:
            missing_count = int(self.df[col].isna().sum())
            if missing_count > 0:
                missing_data.append({
                    "column": col,
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_count / len(self.df) * 100, 2),
                    "dtype": str(self.df[col].dtype),
                    "non_missing_count": len(self.df) - missing_count,
                })
        if not missing_data:
            return pd.DataFrame()
        return pd.DataFrame(missing_data).sort_values("missing_percentage", ascending=False)

    def analyze_missingness_patterns(self) -> pd.DataFrame:
        missing_matrix = self.df.isna().astype(int)
        cols_with_missing = [c for c in missing_matrix.columns if missing_matrix[c].sum() > 0]
        if len(cols_with_missing) < 2:
            return pd.DataFrame()

        missing_corr = missing_matrix[cols_with_missing].corr()
        patterns = []
        for i in range(len(missing_corr.columns)):
            for j in range(i + 1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.3:
                    col1, col2 = missing_corr.columns[i], missing_corr.columns[j]
                    both_missing = int(((self.df[col1].isna()) & (self.df[col2].isna())).sum())
                    patterns.append({
                        "column_1": col1, "column_2": col2, "correlation": round(corr_val, 4),
                        "both_missing_count": both_missing,
                        "both_missing_pct": round(both_missing / len(self.df) * 100, 2),
                    })
        if not patterns:
            return pd.DataFrame()
        return pd.DataFrame(patterns).sort_values("correlation", ascending=False, key=abs)

    # ------------------------------------------------------------------
    def classify_missingness_type(self, column: str, test_columns: Optional[List[str]] = None,
                                   alpha: float = 0.05) -> Dict:
        """Classify the likely missingness mechanism for a column.

        IMPORTANT (documented limitation, previously not disclosed):
        this is a heuristic, not a formal test like Little's MCAR test.
        It looks for statistical association between "is this value
        missing" and other observed variables (evidence of MAR); absence
        of such evidence is treated as weak evidence for MCAR, not proof.

        FIX: the original ran up to 10 hypothesis tests per column without
        correcting for multiple comparisons, which inflates the chance of
        a spurious "significant" result. A Bonferroni correction is now
        applied: alpha_per_test = alpha / n_tests.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        missing_mask = self.df[column].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            return {"column": column, "missingness_type": "No missing values", "confidence": "N/A",
                    "related_variables": [], "evidence_count": 0}

        if test_columns is None:
            test_columns = [c for c in self.df.select_dtypes(include=[np.number]).columns
                             if c != column and self.df[c].notna().sum() > 0]
        test_columns = test_columns[:10]

        # FIX: Bonferroni-corrected per-test alpha instead of using the raw
        # 0.05 threshold for every one of up to 10 tests.
        n_tests = max(len(test_columns), 1)
        alpha_corrected = alpha / n_tests

        mar_evidence = []
        for test_col in test_columns:
            try:
                if self.df[test_col].dtype in ["object", "category"]:
                    contingency = pd.crosstab(self.df[test_col].fillna("_missing_"), missing_mask)
                    _, p_value, _, _ = stats.chi2_contingency(contingency)
                else:
                    group1 = self.df.loc[missing_mask, test_col].dropna()
                    group2 = self.df.loc[~missing_mask, test_col].dropna()
                    if len(group1) <= 1 or len(group2) <= 1:
                        continue
                    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            except (ValueError, TypeError) as exc:
                log.debug("Skipping test for %s vs %s: %s", column, test_col, exc)
                continue

            if p_value < alpha_corrected:
                mar_evidence.append((test_col, p_value))

        if mar_evidence:
            missingness_type = "MAR (Missing At Random) — evidence-based"
            confidence = "High" if len(mar_evidence) >= 3 else "Medium"
            related_vars = [c for c, _ in mar_evidence[:5]]
        else:
            # FIX: explicitly label this branch as a heuristic guess, since
            # "no MAR evidence found" is not proof of MCAR.
            if missing_count / len(self.df) < 0.05:
                missingness_type = "Possibly MCAR (heuristic: low missing rate, no MAR evidence)"
            else:
                missingness_type = "Possibly MNAR (heuristic: no MAR evidence, non-trivial missing rate)"
            confidence = "Low (heuristic, not a formal MCAR test)"
            related_vars = []

        return {
            "column": column,
            "missingness_type": missingness_type,
            "confidence": confidence,
            "related_variables": related_vars,
            "evidence_count": len(mar_evidence),
            "n_tests_run": n_tests,
            "bonferroni_alpha": round(alpha_corrected, 5),
        }

    def recommend_strategy(self, column: str) -> str:
        classification = self.classify_missingness_type(column)
        missing_pct = self.df[column].isna().mean() * 100
        dtype = self.df[column].dtype

        if missing_pct > 50:
            return "drop_column"

        miss_type = classification["missingness_type"]
        if "MCAR" in miss_type:
            return "mean_median" if pd.api.types.is_numeric_dtype(dtype) else "mode"
        elif "MAR" in miss_type:
            return "predictive"
        return "flag_mnar"  # MNAR: no safe automatic fix, flag for manual review

    # FIX: new — the original script only ever printed a text recommendation;
    # nothing actually imputed the data.
    def apply_recommended_imputation(self, column: str) -> Tuple[pd.Series, str]:
        """Apply the recommended strategy and return (imputed_series, strategy_used).

        - drop_column: returns the original series unchanged (caller should drop it)
        - mean_median: numeric -> median (robust to skew); categorical falls back to mode
        - mode: most frequent value
        - predictive: simple regression/classification imputation using the
          columns identified as related during MAR testing
        - flag_mnar: no automatic fix applied; series returned unchanged with a
          boolean indicator column also recommended (left to the caller)
        """
        strategy = self.recommend_strategy(column)
        series = self.df[column].copy()

        if strategy == "drop_column":
            return series, strategy

        if strategy == "mean_median":
            fill_value = series.median()
            return series.fillna(fill_value), strategy

        if strategy == "mode":
            mode_vals = series.mode()
            fill_value = mode_vals[0] if len(mode_vals) else None
            return series.fillna(fill_value), strategy

        if strategy == "predictive":
            classification = self.classify_missingness_type(column)
            predictors = [c for c in classification["related_variables"]
                          if pd.api.types.is_numeric_dtype(self.df[c])]
            if not predictors or not pd.api.types.is_numeric_dtype(series):
                log.warning("No usable numeric predictors for '%s'; falling back to median.", column)
                return series.fillna(series.median()), "mean_median_fallback"

            from sklearn.linear_model import LinearRegression

            data = self.df[[column] + predictors].copy()
            train = data.dropna()
            predict_mask = data[column].isna() & data[predictors].notna().all(axis=1)
            if train.empty or predict_mask.sum() == 0:
                return series.fillna(series.median()), "mean_median_fallback"

            model = LinearRegression().fit(train[predictors], train[column])
            preds = model.predict(data.loc[predict_mask, predictors])
            series.loc[predict_mask] = preds
            # Anything still missing (predictors also missing) falls back to median.
            remaining = series.isna()
            if remaining.any():
                series.loc[remaining] = series.median()
            return series, strategy

        # flag_mnar
        log.info("'%s' classified as possible MNAR — no automatic imputation applied "
                 "(consider adding a missingness indicator column and reviewing manually).", column)
        return series, strategy

    # ------------------------------------------------------------------
    def plot_missing_heatmap(self, figsize: Tuple[int, int] = (12, 8), max_cols: int = 30,
                              out_path: Optional[str] = None, show: bool = True):
        cols_with_missing = [c for c in self.df.columns if self.df[c].isna().sum() > 0][:max_cols]
        if not cols_with_missing:
            log.warning("No missing values to visualize")
            return
        fig = plt.figure(figsize=figsize)
        missing_matrix = self.df[cols_with_missing].isna()
        sns.heatmap(missing_matrix.T, cbar=False, yticklabels=True, cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.title("Missing Value Pattern (Yellow = Missing)")
        plt.xlabel("Row Index")
        plt.ylabel("Column")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    def plot_missing_bar(self, figsize: Tuple[int, int] = (10, 6),
                          out_path: Optional[str] = None, show: bool = True):
        summary = self.get_missing_summary()
        if summary.empty:
            log.warning("No missing values to plot")
            return
        fig = plt.figure(figsize=figsize)
        plt.barh(summary["column"], summary["missing_percentage"], color="coral", edgecolor="black")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Column")
        plt.title("Missing Value Percentage by Column")
        plt.grid(axis="x", alpha=0.3)
        for i, (col, pct) in enumerate(zip(summary["column"], summary["missing_percentage"])):
            plt.text(pct + 1, i, f"{pct:.1f}%", va="center")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    def plot_missing_correlation(self, figsize: Tuple[int, int] = (10, 8),
                                  out_path: Optional[str] = None, show: bool = True):
        missing_matrix = self.df.isna().astype(int)
        cols_with_missing = [c for c in missing_matrix.columns if missing_matrix[c].sum() > 0]
        if len(cols_with_missing) < 2:
            log.warning("Need at least 2 columns with missing values")
            return
        fig = plt.figure(figsize=figsize)
        missing_corr = missing_matrix[cols_with_missing].corr()
        sns.heatmap(missing_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation of Missing Value Patterns")
        plt.tight_layout()
        finalize_plot(fig, out_path, show)

    # ------------------------------------------------------------------
    def generate_full_report(self) -> Dict:
        report = {
            "summary": self.get_missing_summary(),
            "patterns": self.analyze_missingness_patterns(),
            "classifications": [],
            "recommendations": [],
        }
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                report["classifications"].append(self.classify_missingness_type(col))
                report["recommendations"].append({"column": col, "recommendation": self.recommend_strategy(col)})

        report["classifications"] = pd.DataFrame(report["classifications"])
        report["recommendations"] = pd.DataFrame(report["recommendations"])
        return report


# --------------------------------------------------------------------------
def _demo_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "id": range(n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.exponential(50000, n),
        "score": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], n),
    })
    df.loc[np.random.choice(df.index, 50, replace=False), "score"] = np.nan
    df.loc[df["age"] > 60, "income"] = np.nan
    missing_both = np.random.choice(df.index, 80, replace=False)
    df.loc[missing_both, "age"] = np.nan
    df.loc[missing_both, "score"] = np.nan
    df.loc[np.random.choice(df.index, 600, replace=False), "city"] = np.nan
    return df


def main(argv=None) -> int:
    parser = build_base_arg_parser("Missing-data pattern analysis, classification, and imputation.")
    parser.add_argument("--impute", action="store_true", help="Apply the recommended imputation and save the result.")
    args = parser.parse_args(argv)
    show = not args.no_show

    df = load_dataframe(args.input_csv, _demo_dataframe)
    analyzer = MissingDataAnalyzer(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = analyzer.generate_full_report()

    if not report["summary"].empty:
        report["summary"].to_csv(out_dir / "missing_summary.csv", index=False)
        print("Missing Value Summary:")
        print(report["summary"].to_string(index=False))

    if not report["patterns"].empty:
        report["patterns"].to_csv(out_dir / "missing_patterns.csv", index=False)
        print("\nMissingness Patterns (Co-occurrence):")
        print(report["patterns"].to_string(index=False))

    if not report["classifications"].empty:
        report["classifications"].to_csv(out_dir / "missingness_classifications.csv", index=False)
        print("\nMissingness Classifications (heuristic, Bonferroni-corrected):")
        print(report["classifications"].to_string(index=False))

    if not report["recommendations"].empty:
        report["recommendations"].to_csv(out_dir / "imputation_recommendations.csv", index=False)
        print("\nImputation Recommendations:")
        print(report["recommendations"].to_string(index=False))

    if args.impute and not report["summary"].empty:
        imputed_df = df.copy()
        applied = []
        for col in report["summary"]["column"]:
            new_series, strategy_used = analyzer.apply_recommended_imputation(col)
            if strategy_used == "drop_column":
                imputed_df = imputed_df.drop(columns=[col])
            else:
                imputed_df[col] = new_series
            applied.append({"column": col, "strategy_applied": strategy_used})
        pd.DataFrame(applied).to_csv(out_dir / "imputation_applied.csv", index=False)
        imputed_df.to_csv(out_dir / "imputed_dataset.csv", index=False)
        print(f"\nImputed dataset written to {out_dir / 'imputed_dataset.csv'}")

    analyzer.plot_missing_bar(out_path=str(out_dir / "missing_bar.png"), show=show)
    analyzer.plot_missing_heatmap(out_path=str(out_dir / "missing_heatmap.png"), show=show)
    analyzer.plot_missing_correlation(out_path=str(out_dir / "missing_correlation.png"), show=show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
