"""
eda_common.py

Shared utilities used across the EDA script suite (data_profiler,
distribution_analyzer, correlation_explorer, outlier_suite,
missing_data_analyzer).

v2 additions:
- Multi-format data loading: CSV, TSV/TXT, Excel (.xlsx/.xls, with
  --sheet-name), Parquet, JSON, XML, and SQL databases (--db-uri +
  --db-table or --db-query).
- Output files are now split into `plots/<script>/` and
  `reports/<script>/` subfolders under --output-dir instead of one
  flat folder.
- Datetime auto-detection: object/string columns that are actually
  dates (e.g. "Date recrutement", "2022-03-15") are now parsed and
  cast to real datetime64 columns at load time. Previously these were
  left as generic strings, so every script's `select_dtypes(object)`
  picked them up as "categorical" columns — which is exactly what
  produced the unreadable multi-hundred-bar charts with overlapping
  tick labels / titles seen in `plot_categorical_distributions`.
- `create_grid`: a replacement for the old fixed-figsize subplot grids.
  Figure height now scales with the number of rows and uses
  `constrained_layout=True`, which fixes the title/tick-label overlap
  bug visible in the original multi-panel plots (titles were being
  pushed into the axes above because a fixed figsize was crammed with
  a growing number of rows).
- `bucket_top_n`: high-cardinality categorical columns (e.g. IDs,
  free-text, or anything with 50+ unique values) are now shown as
  "top N + Other" instead of dumping every category into an unreadable
  bar chart.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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


log = get_logger("eda_common")


# --------------------------------------------------------------------------
# Shared statistics
# --------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Tuple[float, float]:
    """Return (lower_bound, upper_bound) for the IQR outlier rule."""
    clean = series.dropna()
    if clean.empty:
        return (np.nan, np.nan)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    return (q1 - multiplier * iqr, q3 + multiplier * iqr)


def iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Boolean mask (aligned to series.index) flagging IQR outliers."""
    lower, upper = iqr_bounds(series, multiplier)
    mask = pd.Series(False, index=series.index)
    if pd.isna(lower):
        return mask
    valid = series.notna()
    mask.loc[valid] = (series[valid] < lower) | (series[valid] > upper)
    return mask


def bucket_top_n(value_counts: pd.Series, n: int = 15) -> Tuple[pd.Series, int, float]:
    """Collapse a value_counts() Series into its top-n entries + 'Other'.

    Returns (bucketed_series, n_categories_in_other, coverage_fraction)
    where coverage_fraction is the share of total observations shown
    explicitly (i.e. not folded into 'Other').

    This replaces the previous behavior of plotting every category
    (sometimes 1000+) as its own bar, which was unreadable and, for
    near-uniform high-cardinality columns (like IDs or raw date
    strings), visually meaningless.
    """
    total = value_counts.sum()
    if len(value_counts) <= n or total == 0:
        return value_counts, 0, 1.0
    top = value_counts.iloc[:n]
    other_count = value_counts.iloc[n:].sum()
    other_n = len(value_counts) - n
    bucketed = pd.concat([top, pd.Series({f"Other ({other_n} categories)": other_count})])
    coverage = top.sum() / total
    return bucketed, other_n, coverage


def truncate_label(label: Any, max_len: int = 22) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# --------------------------------------------------------------------------
# Datetime auto-detection — fixes date-like string columns being treated
# as high-cardinality "categorical" columns everywhere downstream.
# --------------------------------------------------------------------------
_DATE_NAME_HINTS = (
    "date", "_at", "_dt", "naissance", "recrutement", "embauche", "depart",
    "birth", "hired", "start", "end", "created", "updated", "timestamp",
)


def infer_and_cast_datetime_columns(df: pd.DataFrame, min_success_rate: float = 0.9,
                                     sample_size: int = 500) -> pd.DataFrame:
    """Detect object/string columns that are really dates and cast them to
    datetime64. Returns a new DataFrame.

    A column is cast when EITHER:
    - its name matches a common date-related hint (e.g. contains "date",
      "_at", "recrutement", ...), and at least `min_success_rate` of a
      sample parses as a date, OR
    - it has no date-hint in its name but a very high fraction of a
      sample still parses AND the column isn't purely numeric-looking
      (avoids turning numeric-string ID columns into dates).

    Intentionally conservative — only touches object-dtype columns,
    never already-numeric or already-boolean columns.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        sample = series.sample(min(sample_size, len(series)), random_state=42) if len(series) > sample_size else series

        name_hint = any(hint in col.lower() for hint in _DATE_NAME_HINTS)
        looks_numeric = pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9
        if looks_numeric and not name_hint:
            continue

        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            success_rate = parsed.notna().mean()
        except Exception:
            success_rate = 0.0

        threshold = min_success_rate if name_hint else max(min_success_rate, 0.98)
        if success_rate >= threshold:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            log.info("Auto-detected '%s' as a datetime column (parse rate %.0f%%) — "
                     "cast from object to datetime64.", col, success_rate * 100)
    return df


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def use_headless_backend_if_needed() -> None:
    import os

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")


def finalize_plot(fig, out_path: Optional[str], show: bool, dpi: int = 150) -> None:
    """Save a figure to disk and/or show it, then close it."""
    import matplotlib.pyplot as plt

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved plot -> %s", out_path)
    if show:
        plt.show()
    plt.close(fig)


def create_grid(n_items: int, n_cols: int, per_row_height: float = 3.4, per_col_width: float = 6.0,
                 min_height: float = 3.5, max_height: float = 60.0):
    """Create a subplot grid sized to the number of items, using
    constrained_layout so titles/labels never overlap neighboring axes.

    FIX: the original scripts used a hardcoded `figsize` regardless of
    how many rows were needed (e.g. always (15, 10) even for 8 rows),
    which is exactly what caused titles to overlap the axes above them
    and tick labels to collide between subplots. Height now scales with
    `n_rows`, and `constrained_layout=True` lets matplotlib negotiate
    spacing instead of a single global `tight_layout()` call.
    """
    import matplotlib.pyplot as plt

    n_items = max(n_items, 1)
    n_cols = max(min(n_cols, n_items), 1)
    n_rows = (n_items + n_cols - 1) // n_cols
    height = min(max(per_row_height * n_rows, min_height), max_height)
    width = per_col_width * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    return fig, axes, n_rows


# --------------------------------------------------------------------------
# Output directory layout — split into plots/<script>/ and reports/<script>/
# --------------------------------------------------------------------------
def get_output_dirs(base_dir: str, script_name: str) -> Tuple[Path, Path]:
    """Return (plots_dir, reports_dir), creating both if needed.

    FIX: every script used to dump plots and CSV/report files together
    into one flat --output-dir. Now: <output-dir>/plots/<script>/ and
    <output-dir>/reports/<script>/.
    """
    base = Path(base_dir)
    plots_dir = base / "plots" / script_name
    reports_dir = base / "reports" / script_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, reports_dir


# --------------------------------------------------------------------------
# Common CLI / multi-format data loading
# --------------------------------------------------------------------------
def build_base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to a data file (.csv, .tsv, .txt, .xlsx, .xls, .parquet, "
        ".json, .xml). Omit to use a synthetic demo dataset, or omit and pass "
        "--db-uri to read from a database instead.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "txt", "excel", "parquet", "json", "xml", "db"],
        default=None,
        help="Force the input format instead of guessing from the file extension.",
    )
    parser.add_argument("--sheet-name", default=0,
                         help="Excel sheet name or 0-based index (default: first sheet).")
    parser.add_argument("--delimiter", default=None,
                         help="Field delimiter for csv/tsv/txt (default: auto — comma for .csv, tab for .tsv/.txt).")
    parser.add_argument("--json-orient", default=None,
                         help="Orient passed to pandas.read_json (e.g. 'records', 'columns').")
    parser.add_argument("--xml-xpath", default=None,
                         help="XPath passed to pandas.read_xml to select the repeating record element.")
    parser.add_argument("--db-uri", default=None,
                         help="SQLAlchemy database URI, e.g. 'sqlite:///mydata.db' or "
                         "'postgresql://user:pass@host/db'. Requires --db-table or --db-query.")
    parser.add_argument("--db-table", default=None, help="Table name to read entirely from --db-uri.")
    parser.add_argument("--db-query", default=None, help="Custom SQL query to run against --db-uri.")
    parser.add_argument("--no-auto-datetime", action="store_true",
                         help="Disable automatic detection/casting of date-like string columns.")
    parser.add_argument(
        "--output-dir",
        default="eda_output",
        help="Base directory for outputs; plots go to <output-dir>/plots/<script>/ "
        "and reports go to <output-dir>/reports/<script>/.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows (still saves to --output-dir).",
    )
    return parser


def _read_by_format(path: Path, fmt: str, args) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path, delimiter=args.delimiter or ",")
    if fmt in ("tsv", "txt"):
        return pd.read_csv(path, delimiter=args.delimiter or "\t")
    if fmt == "excel":
        return pd.read_excel(path, sheet_name=args.sheet_name)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        return pd.read_json(path, orient=args.json_orient)
    if fmt == "xml":
        kwargs = {"xpath": args.xml_xpath} if args.xml_xpath else {}
        return pd.read_xml(path, **kwargs)
    raise ValueError(f"Unsupported format: {fmt}")


_EXTENSION_MAP = {
    ".csv": "csv", ".tsv": "tsv", ".txt": "txt",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".parquet": "parquet", ".pq": "parquet",
    ".json": "json",
    ".xml": "xml",
}


def _load_from_database(args) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    if not args.db_table and not args.db_query:
        raise ValueError("--db-uri requires either --db-table or --db-query.")
    engine = create_engine(args.db_uri)
    with engine.connect() as conn:
        if args.db_query:
            log.info("Running SQL query against %s", args.db_uri)
            return pd.read_sql_query(text(args.db_query), conn)
        log.info("Reading table '%s' from %s", args.db_table, args.db_uri)
        return pd.read_sql_table(args.db_table, conn)


def load_dataframe(args, demo_factory: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load data from any supported source based on parsed CLI args.

    Supports (in order of precedence):
    1. --db-uri (+ --db-table or --db-query) — any SQLAlchemy-supported database
    2. input_path with format auto-detected from its extension, or forced
       via --format: csv, tsv, txt, excel (with --sheet-name), parquet,
       json, xml
    3. no input given -> falls back to the script's synthetic demo dataset

    Date-like string columns are auto-detected and cast to real datetime64
    columns unless --no-auto-datetime is passed — this is what prevents
    columns like "Date recrutement" from being treated as high-cardinality
    categorical columns downstream.
    """
    if getattr(args, "db_uri", None):
        df = _load_from_database(args)
    elif getattr(args, "input_path", None) is None:
        log.info("No input file given — using synthetic demo dataset.")
        df = demo_factory()
    else:
        path = Path(args.input_path)
        if not path.exists():
            log.error("File not found: %s", path)
            raise FileNotFoundError(f"No such file: {path}")

        fmt = args.format or _EXTENSION_MAP.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Could not infer format from extension '{path.suffix}'. "
                f"Pass --format explicitly (csv/tsv/txt/excel/parquet/json/xml)."
            )
        try:
            df = _read_by_format(path, fmt, args)
        except Exception as exc:
            log.error("Failed to read '%s' as %s: %s", path, fmt, exc)
            raise
        log.info("Loaded '%s' (%s) -> shape=%s", path, fmt, df.shape)

    if df.empty:
        raise ValueError("Loaded data has 0 rows.")

    if not getattr(args, "no_auto_datetime", False):
        df = infer_and_cast_datetime_columns(df)

    return df
