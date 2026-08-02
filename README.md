# Original work credit goes to [balapriyac](https://github.com/balapriyac)
## This is an attempt at enhancing the scripts from [balapriyac/data-science-tuotrials/useful-python-scripts-eda](https://github.com/balapriyac/data-science-tutorials/tree/main/useful-python-scripts-eda) after running into some limitations while using them in my own EDA workflow.

# EDA scripts — fixed versions (with the help of Claude Sonnet 5)

Corrected versions of the 5 scripts from `useful-python-scripts-eda`
(originally accompanying the KDnuggets article *"5 Useful Python
Scripts to Automate Exploratory Data Analysis"*).

## Install dependencies

```bash
pip install -r requirements.txt
```

`openpyxl`/`xlrd` (Excel), `pyarrow` (Parquet), `lxml` (XML), and
`SQLAlchemy` (databases) are included so every supported format works
out of the box. Connecting to a database other than SQLite (Postgres,
MySQL, etc.) additionally needs that database's driver — e.g.
`pip install psycopg2-binary` for Postgres, `pip install pymysql` for
MySQL — SQLAlchemy loads the driver by name from the connection URI.

## Output layout

Every script now writes into two subfolders under `--output-dir`
(default `./eda_output`), grouped by script:

```
eda_output/
├── plots/
│   ├── data_profiler/            (none — profiler is report-only)
│   ├── distribution_analyzer/
│   ├── correlation_explorer/
│   ├── outlier_suite/
│   └── missing_data_analyzer/
└── reports/
    ├── data_profiler/
    ├── distribution_analyzer/
    ├── correlation_explorer/
    ├── outlier_suite/
    └── missing_data_analyzer/
```

Previously everything landed in one flat folder; this makes it easy to
grab "all the plots" or "all the CSV reports" for a given script
without sifting through a mixed pile of files.

## Supported input formats

Every script accepts the same input options (via the shared
`eda_common.py`):

| Format | How to pass it | Notes |
|---|---|---|
| CSV | `script.py data.csv` | delimiter auto = `,`; override with `--delimiter` |
| TSV / TXT | `script.py data.tsv` | delimiter auto = tab; override with `--delimiter` |
| Excel | `script.py data.xlsx --sheet-name Sheet1` | `--sheet-name` accepts a name or a 0-based index |
| Parquet | `script.py data.parquet` | — |
| JSON | `script.py data.json --json-orient records` | `--json-orient` optional, forwarded to `pandas.read_json` |
| XML | `script.py data.xml --xml-xpath ".//row"` | `--xml-xpath` optional, forwarded to `pandas.read_xml` |
| Database | `script.py --db-uri sqlite:///data.db --db-table employees` | or `--db-query "SELECT ..."` instead of `--db-table`; any SQLAlchemy URI |

If the extension is ambiguous or missing, force it with `--format
{csv,tsv,txt,excel,parquet,json,xml}`.

You can try to run any script with **no** input argument to try it against its
built-in synthetic demo dataset.

### Automatic date detection

String columns that are actually dates (e.g. `"2022-03-15"`) are auto-detected and cast to real datetime columns
before analysis, regardless of source format. Disable this with
`--no-auto-datetime` if you need the raw string behavior.

## What was fixed in this version

- **Visual bug:** multi-panel plots (categorical
  distributions, numeric distributions, box plots, Q-Q plots, top
  correlations) used a **fixed figsize regardless of how many rows**
  were being drawn, and dumped every category of high-cardinality
  columns — including raw date strings — into one bar chart. With 8
  rows crammed into a short, fixed-height figure, subplot titles
  overlapped the axes above them and rotated tick labels collided
  between panels (the artifact you shared). Fixed by:
  - `eda_common.create_grid()`: figure height now scales with the
    number of rows, and every grid uses `constrained_layout=True` so
    matplotlib negotiates spacing instead of a single global
    `tight_layout()` call.
  - Date-like string columns are auto-detected and cast to datetime64
    at load time, so they're plotted as timelines, not bar charts.
  - Remaining high-cardinality categorical columns are bucketed into
    "top 15 + Other" (`eda_common.bucket_top_n`) with the coverage
    percentage shown in the subtitle, instead of one bar per category.
  - Long category labels are truncated with an ellipsis and rotated
    with proper right-alignment so they no longer run into each other.
- **Multi-format input**: CSV/TSV/TXT/Excel/Parquet/JSON/XML/databases,
  all through the same `--input-path` / `--db-uri` CLI options.
- **Output organization**: `plots/<script>/` and `reports/<script>/`
  instead of one flat folder.
- *(carried over from the previous round)* the `data_profiler.py`
  duplicate-report bug, the `correlation_explorer.py`
  `plot_top_correlations` indexing bug, the bare `except:` in
  `outlier_suite.py`, the uncorrected multiple-hypothesis-testing issue
  in `missing_data_analyzer.py`, reproducible normality testing, actual
  outlier treatment and imputation implementations, and p-values /
  Cramer's V in `correlation_explorer.py`.

## Usage — commands, arguments, and flags

All five scripts share the base options below (from
`eda_common.build_base_arg_parser`), plus script-specific flags listed
under each one.

**Shared options:**

| Flag | Default | Meaning |
|---|---|---|
| `input_path` (positional) | none | Path to the data file. Omit to use synthetic demo data. |
| `--format` | auto-detected | Force `csv`/`tsv`/`txt`/`excel`/`parquet`/`json`/`xml` |
| `--sheet-name` | `0` | Excel sheet name or index |
| `--delimiter` | auto | Field delimiter for csv/tsv/txt |
| `--json-orient` | none | Orient for `pandas.read_json` |
| `--xml-xpath` | none | XPath for `pandas.read_xml` |
| `--db-uri` | none | SQLAlchemy database URI |
| `--db-table` | none | Table to read entirely (needs `--db-uri`) |
| `--db-query` | none | Custom SQL query (needs `--db-uri`) |
| `--no-auto-datetime` | off | Disable automatic date-column detection |
| `--output-dir` | `eda_output` | Base folder; writes to `<dir>/plots/<script>/` and `<dir>/reports/<script>/` |
| `--no-show` | off | Don't open interactive plot windows (still saves them) |

### `data_profiler.py`

```bash
python data_profiler.py [input_path] [--high-cardinality-threshold 0.5] [shared options]
```
- `--high-cardinality-threshold` (default `0.5`): fraction of unique
  values above which a categorical column is flagged high-cardinality.
- Output: `reports/data_profiler/` only (numeric/categorical/datetime/
  boolean profiles + data quality issues, each as CSV). No plots.

```bash
python data_profiler.py employees.xlsx --sheet-name RH --output-dir out
python data_profiler.py --db-uri sqlite:///hr.db --db-table employees --output-dir out
```

### `distribution_analyzer.py`

```bash
python distribution_analyzer.py [input_path] [shared options]
```
- No script-specific flags beyond the shared set.
- Output: `reports/distribution_analyzer/` (numeric, categorical, and
  datetime distribution reports as CSV) and
  `plots/distribution_analyzer/` (`numeric_distributions.png`,
  `boxplots.png`, `qq_plots.png`, `categorical_distributions.png`,
  `datetime_distributions.png`).

```bash
python distribution_analyzer.py employees.csv --no-show --output-dir out
```

### `correlation_explorer.py`

```bash
python correlation_explorer.py [input_path] [--target COLUMN] [shared options]
```
- `--target` (optional): numeric column to run mutual-information
  analysis against. If omitted, tries a column literally named
  `target`; otherwise mutual information is skipped.
- Output: `reports/correlation_explorer/` (`high_correlations.csv`
  with p-values, `vif.csv`, `mutual_information.csv` if a target was
  found, `cramers_v.csv` for categorical associations) and
  `plots/correlation_explorer/` (`correlation_heatmap.png`,
  `correlation_comparison.png`, `top_correlations.png`).

```bash
python correlation_explorer.py sales.parquet --target revenue --output-dir out
```

### `outlier_suite.py`

```bash
python outlier_suite.py [input_path] [--consensus-min-methods 2] \
    [--treat-column COLUMN] [--treat-strategy {cap,remove,impute}] [shared options]
```
- `--consensus-min-methods` (default `2`): how many of the 3 detectors
  (IQR, z-score, modified z-score) must flag a row for it to count as
  a "consensus" outlier.
- `--treat-column` (default: first numeric column): column to
  demonstrate outlier treatment on.
- `--treat-strategy` (default `cap`): `cap` (winsorize to IQR bounds),
  `remove` (set to NaN), or `impute` (replace with the median).
- Output: `reports/outlier_suite/` (`outlier_summary.csv`,
  `<column>_treated.csv`) and `plots/outlier_suite/`
  (`outlier_comparison.png`, `multivariate_outliers.png`).

```bash
python outlier_suite.py transactions.json --treat-column amount --treat-strategy remove --output-dir out
```

### `missing_data_analyzer.py`

```bash
python missing_data_analyzer.py [input_path] [--impute] [shared options]
```
- `--impute` (flag, off by default): actually apply the recommended
  imputation strategy per column and write the resulting dataset.
- Output: `reports/missing_data_analyzer/` (`missing_summary.csv`,
  `missing_patterns.csv`, `missingness_classifications.csv`,
  `imputation_recommendations.csv`, and — only with `--impute` —
  `imputation_applied.csv` + `imputed_dataset.csv`) and
  `plots/missing_data_analyzer/` (`missing_bar.png`,
  `missing_heatmap.png`, `missing_correlation.png`).

```bash
python missing_data_analyzer.py survey.xml --impute --output-dir out
```

## Known remaining limitations

- The MCAR/MNAR classification is still a heuristic (association
  testing + missing-rate threshold), not Little's formal MCAR test.
- Predictive imputation in `missing_data_analyzer.py` uses simple
  linear regression on numeric predictors only.
- Datetime auto-detection is heuristic (name hints + parse success
  rate on a sample); very unusual date formats or locales may not be
  picked up — use `--no-auto-datetime` and pre-parse the column
  yourself if needed.
- No automated test suite yet — each script was manually verified
  end-to-end against its demo dataset, a headless run, and all
  supported file formats (CSV/TSV/Excel/Parquet/JSON/XML/SQLite).