# Original work credit goes to [balapriyac](https://github.com/balapriyac)
## This is an attempt at enhancing the scripts from [balapriyac/data-science-tuotrials/useful-python-scripts-eda](https://github.com/balapriyac/data-science-tutorials/tree/main/useful-python-scripts-eda) after running into some limitations while using them in my own EDA workflow.


# EDA scripts — fixed versions (with the help of Claude Sonnet 5)

Corrected versions of the 5 scripts from `useful-python-scripts-eda`
(originally accompanying the KDnuggets article *"5 Useful Python
Scripts to Automate Exploratory Data Analysis"*).

## Install

```bash
pip install -r requirements.txt
```

## Run against your own data

Every script now accepts a CSV path and writes reports/plots to disk
instead of only printing / calling `plt.show()`:

```bash
python data_profiler.py your_data.csv --output-dir out/

python distribution_analyzer.py your_data.csv --output-dir out/ --no-show

python correlation_explorer.py your_data.csv --output-dir out/ --target price

python outlier_suite.py your_data.csv --output-dir out/ --treat-column revenue --treat-strategy cap

python missing_data_analyzer.py your_data.csv --output-dir out/ --impute
```

Run with no CSV argument to try any script against its built-in
synthetic demo dataset.

## What changed vs. the original scripts

- **`eda_common.py`** (new file added): shared IQR logic, plot-saving helper,
  logging, and a common CLI/CSV loader — removes the duplicated (and
  slightly inconsistent) IQR code that used to live in three different
  files, and gives every script a real command-line interface.
- **`data_profiler.py`**: fixed a bug where calling
  `generate_full_profile()` twice silently duplicated every row in the
  report; added datetime/boolean column profiling; CSV export.
- **`distribution_analyzer.py`**: fixed a `typing.Any` typo
  (`Dict[str, any]`); made the normality test reproducible (seeded
  sampling); added a categorical distribution report; plots can now be
  saved to disk.
- **`correlation_explorer.py`**: fixed a bug in
  `plot_top_correlations` where indexing `axes[idx]` with a
  post-sort, non-contiguous DataFrame index could raise `IndexError` or
  misplace plots; added correlation p-values and Cramer's V for
  categorical associations.
- **`outlier_suite.py`**: replaced a bare `except:` with narrowed
  exception handling that logs failures; added actual outlier
  *treatment* (cap/remove/impute) — previously detection-only; made the
  consensus threshold configurable instead of hardcoded.
- **`missing_data_analyzer.py`**: applied a Bonferroni correction for
  the multiple hypothesis tests run per column (previously inflated the
  false-positive rate for "MAR" classifications); the MCAR/MNAR
  heuristic is now clearly labeled as a heuristic, not a formal test;
  added real imputation (`apply_recommended_imputation`), not just a
  text recommendation.

## Known remaining limitations

- The MCAR/MNAR classification is still a heuristic (association
  testing + missing-rate threshold), not Little's formal MCAR test.
- Predictive imputation in `missing_data_analyzer.py` uses simple
  linear regression on numeric predictors only.
- No automated test suite yet — each script was manually verified
  end-to-end against its demo dataset and a headless run.
