# Findings: ticket#5

## Review Results
- Test coverage complete for all 7 functions, but missing edge cases (exact threshold boundaries, zero-MAD/zero-IQR, NaN propagation, remaining distribution shapes).
- `classify_correlation_strength` expects absolute correlation but doesn't enforce `abs()`.
- `detect_outliers_modified_zscore` returns unaligned boolean series on early return.

## Action Taken
- Tests expanded in `tests/test_eda_stats.py`.
- Improvements noted for future refactoring.
