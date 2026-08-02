"""
Comprehensive Data Profiler (fixed)

Automatically generates complete profiles of datasets including
data types, statistics, cardinality, and data quality indicators.

Fixes applied vs. the original version:
- BUG: generate_full_profile() used to *append* to self.numeric_profiles /
  self.categorical_profiles without clearing them first. Calling it twice
  (as the original __main__ block did, once via print_summary() and once
  directly) silently duplicated every row in the report. Profiles are now
  rebuilt fresh on every call.
- Datetime and boolean columns were counted in the overview but never
  actually profiled. They now get their own profile classes.
- Type-mismatch detection only checked the first 100 non-null values;
  now samples up to 1000 (still bounded, but far less likely to miss it).
- Division-by-zero / empty-DataFrame guards added.
- Added CLI support and CSV export of every report table (previously
  only available as commented-out example lines).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda_common import build_base_arg_parser, get_logger, load_dataframe

import warnings

warnings.filterwarnings("ignore")

log = get_logger("data_profiler")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    memory_bytes: int


@dataclass
class NumericProfile(ColumnProfile):
    min_value: Optional[float]
    max_value: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    zeros_count: int
    zeros_percentage: float
    negative_count: int


@dataclass
class CategoricalProfile(ColumnProfile):
    top_value: Any
    top_frequency: int
    top_percentage: float
    top_5_values: Dict[Any, int]
    is_high_cardinality: bool


@dataclass
class DatetimeProfile(ColumnProfile):
    min_date: Any
    max_date: Any
    date_range_days: Optional[float]
    most_common_year: Any


@dataclass
class BooleanProfile(ColumnProfile):
    true_count: int
    false_count: int
    true_percentage: float


class DataProfiler:
    def __init__(self, df: pd.DataFrame, high_cardinality_threshold: float = 0.5):
        if df is None or df.empty:
            raise ValueError("DataProfiler requires a non-empty DataFrame.")
        self.df = df
        self.high_cardinality_threshold = high_cardinality_threshold

    # ------------------------------------------------------------------
    def generate_overview(self) -> Dict[str, Any]:
        n_rows = len(self.df)
        overview = {
            "total_rows": n_rows,
            "total_columns": len(self.df.columns),
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "total_memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024**2), 2),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "duplicate_percentage": round(self.df.duplicated().sum() / n_rows * 100, 2) if n_rows else 0.0,
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(self.df.select_dtypes(include=["object", "category"]).columns),
            "datetime_columns": len(self.df.select_dtypes(include=["datetime64"]).columns),
            "boolean_columns": len(self.df.select_dtypes(include=["bool"]).columns),
        }
        return overview

    # ------------------------------------------------------------------
    def _base_stats(self, col: str) -> Dict[str, Any]:
        series = self.df[col]
        n = len(self.df)
        null_count = int(series.isna().sum())
        return dict(
            name=col,
            dtype=str(series.dtype),
            non_null_count=n - null_count,
            null_count=null_count,
            null_percentage=round(null_count / n * 100, 2) if n else 0.0,
            unique_count=int(series.nunique()),
            unique_percentage=round(series.nunique() / n * 100, 2) if n else 0.0,
            memory_bytes=int(series.memory_usage(deep=True)),
        )

    def profile_numeric_column(self, col: str) -> NumericProfile:
        series = self.df[col]
        non_null = series.dropna()
        n = len(self.df)

        def r(x):
            return round(x, 4) if x is not None and not pd.isna(x) else None

        stats = dict(
            min_value=non_null.min() if len(non_null) else np.nan,
            max_value=non_null.max() if len(non_null) else np.nan,
            mean=non_null.mean() if len(non_null) else np.nan,
            median=non_null.median() if len(non_null) else np.nan,
            std=non_null.std() if len(non_null) else np.nan,
            q25=non_null.quantile(0.25) if len(non_null) else np.nan,
            q75=non_null.quantile(0.75) if len(non_null) else np.nan,
            skewness=non_null.skew() if len(non_null) > 2 else np.nan,
            kurtosis=non_null.kurtosis() if len(non_null) > 3 else np.nan,
        )
        zeros = int((series == 0).sum())
        negatives = int((series < 0).sum())

        return NumericProfile(
            **self._base_stats(col),
            min_value=r(stats["min_value"]),
            max_value=r(stats["max_value"]),
            mean=r(stats["mean"]),
            median=r(stats["median"]),
            std=r(stats["std"]),
            q25=r(stats["q25"]),
            q75=r(stats["q75"]),
            skewness=r(stats["skewness"]),
            kurtosis=r(stats["kurtosis"]),
            zeros_count=zeros,
            zeros_percentage=round(zeros / n * 100, 2) if n else 0.0,
            negative_count=negatives,
        )

    def profile_categorical_column(self, col: str) -> CategoricalProfile:
        series = self.df[col]
        n = len(self.df)
        value_counts = series.value_counts()
        top_value = value_counts.index[0] if len(value_counts) else None
        top_freq = int(value_counts.iloc[0]) if len(value_counts) else 0
        unique_count = int(series.nunique())
        is_high_card = (unique_count / n) > self.high_cardinality_threshold if n else False

        return CategoricalProfile(
            **self._base_stats(col),
            top_value=top_value,
            top_frequency=top_freq,
            top_percentage=round(top_freq / n * 100, 2) if n else 0.0,
            top_5_values=value_counts.head(5).to_dict(),
            is_high_cardinality=is_high_card,
        )

    def profile_datetime_column(self, col: str) -> DatetimeProfile:
        series = pd.to_datetime(self.df[col], errors="coerce")
        non_null = series.dropna()
        min_d = non_null.min() if len(non_null) else None
        max_d = non_null.max() if len(non_null) else None
        range_days = (max_d - min_d).days if (min_d is not None and max_d is not None) else None
        most_common_year = non_null.dt.year.mode()[0] if len(non_null) else None

        return DatetimeProfile(
            **self._base_stats(col),
            min_date=min_d,
            max_date=max_d,
            date_range_days=range_days,
            most_common_year=most_common_year,
        )

    def profile_boolean_column(self, col: str) -> BooleanProfile:
        series = self.df[col]
        n = len(self.df)
        true_count = int(series.sum(skipna=True))
        non_null = int(series.notna().sum())
        false_count = non_null - true_count

        return BooleanProfile(
            **self._base_stats(col),
            true_count=true_count,
            false_count=false_count,
            true_percentage=round(true_count / n * 100, 2) if n else 0.0,
        )

    # ------------------------------------------------------------------
    def detect_issues(self) -> List[Dict[str, Any]]:
        issues = []
        n = len(self.df)
        for col in self.df.columns:
            series = self.df[col]
            missing_pct = series.isna().mean() * 100

            if missing_pct > 50:
                issues.append({"column": col, "issue": "High Missing Rate", "severity": "High",
                                "detail": f"{missing_pct:.1f}% missing"})

            if series.nunique(dropna=True) <= 1:
                issues.append({"column": col, "issue": "Zero Variance", "severity": "High",
                                "detail": "Column has only one unique value"})

            if series.dtype in ["object", "category"]:
                unique_ratio = series.nunique() / n if n else 0
                if unique_ratio > self.high_cardinality_threshold:
                    issues.append({"column": col, "issue": "High Cardinality", "severity": "Medium",
                                    "detail": f"{series.nunique()} unique values ({unique_ratio * 100:.1f}%)"})

            if series.dtype == "object":
                # FIX: sample up to 1000 rows (was 100) to reduce the chance
                # of missing a type mismatch further down the column.
                sample = series.dropna().head(1000)
                if len(sample):
                    try:
                        pd.to_numeric(sample, errors="raise")
                        issues.append({"column": col, "issue": "Potential Type Mismatch", "severity": "Low",
                                        "detail": "Stored as object but appears numeric"})
                    except (ValueError, TypeError):
                        pass

        return issues

    # ------------------------------------------------------------------
    def generate_full_profile(self) -> Dict[str, Any]:
        """Generate complete data profile.

        FIX: previously this method appended to self.numeric_profiles /
        self.categorical_profiles, so calling it more than once produced
        duplicate rows. Profiles are now local to this call.
        """
        log.info("Generating data profile for shape=%s ...", self.df.shape)

        overview = self.generate_overview()

        numeric_profiles = [self.profile_numeric_column(c) for c in self.df.select_dtypes(include=[np.number]).columns]
        categorical_profiles = [self.profile_categorical_column(c) for c in self.df.select_dtypes(include=["object", "category"]).columns]
        datetime_profiles = [self.profile_datetime_column(c) for c in self.df.select_dtypes(include=["datetime64"]).columns]
        boolean_profiles = [self.profile_boolean_column(c) for c in self.df.select_dtypes(include=["bool"]).columns]

        issues = self.detect_issues()

        report = {
            "overview": overview,
            "numeric_profiles": pd.DataFrame([vars(p) for p in numeric_profiles]),
            "categorical_profiles": pd.DataFrame([vars(p) for p in categorical_profiles]),
            "datetime_profiles": pd.DataFrame([vars(p) for p in datetime_profiles]),
            "boolean_profiles": pd.DataFrame([vars(p) for p in boolean_profiles]),
            "data_quality_issues": pd.DataFrame(issues) if issues else pd.DataFrame(),
        }
        log.info("Profile generation complete.")
        return report

    def print_summary(self, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Print a human-readable summary.

        FIX: now accepts an optional pre-built report so callers don't
        trigger a second (redundant, and previously duplicating) profile
        generation pass.
        """
        if report is None:
            report = self.generate_full_profile()

        print("\n" + "=" * 70)
        print("DATASET OVERVIEW")
        print("=" * 70)
        for key, value in report["overview"].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        for label, key in [
            ("NUMERIC COLUMNS SUMMARY", "numeric_profiles"),
            ("CATEGORICAL COLUMNS SUMMARY", "categorical_profiles"),
            ("DATETIME COLUMNS SUMMARY", "datetime_profiles"),
            ("BOOLEAN COLUMNS SUMMARY", "boolean_profiles"),
        ]:
            print("\n" + "=" * 70)
            print(label)
            print("=" * 70)
            df_ = report[key]
            if not df_.empty:
                print(df_.to_string(index=False))
            else:
                print("None found")

        print("\n" + "=" * 70)
        print("DATA QUALITY ISSUES")
        print("=" * 70)
        if not report["data_quality_issues"].empty:
            print(report["data_quality_issues"].to_string(index=False))
        else:
            print("No major issues detected")
        print("=" * 70 + "\n")
        return report


# --------------------------------------------------------------------------
def _demo_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(1000),
        "age": np.random.normal(35, 10, 1000),
        "income": np.random.exponential(50000, 1000),
        "score": np.random.uniform(0, 100, 1000),
        "signup_date": pd.date_range("2020-01-01", periods=1000, freq="D"),
        "is_active": np.random.choice([True, False], 1000),
        "category": np.random.choice(["A", "B", "C", "D"], 1000),
        "segment": np.random.choice(["Premium", "Standard", "Basic"], 1000),
        "name": [f"User_{i}" for i in range(1000)],
        "constant": ["same_value"] * 1000,
        "numeric_string": [str(i) for i in range(1000)],
    })
    df.loc[np.random.choice(df.index, 600, replace=False), "score"] = np.nan
    df.loc[np.random.choice(df.index, 50, replace=False), "age"] = np.nan
    return df


def main(argv=None) -> int:
    parser = build_base_arg_parser("Comprehensive data profiler for any CSV dataset.")
    parser.add_argument("--high-cardinality-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    df = load_dataframe(args.input_csv, _demo_dataframe)

    profiler = DataProfiler(df, high_cardinality_threshold=args.high_cardinality_threshold)
    report = profiler.generate_full_profile()
    profiler.print_summary(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ["numeric_profiles", "categorical_profiles", "datetime_profiles", "boolean_profiles", "data_quality_issues"]:
        df_ = report[key]
        if not df_.empty:
            out_path = out_dir / f"{key}.csv"
            df_.to_csv(out_path, index=False)
            log.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
