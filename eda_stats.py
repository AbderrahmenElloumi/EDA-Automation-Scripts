import pandas as pd
import numpy as np
from scipy import stats

def calculate_summary_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for numeric columns in a DataFrame."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats_dict = {}
    for col in numeric_df.columns:
        series = numeric_df[col]
        stats_dict[col] = {
            'mean': float(series.mean()),
            'std': float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            'min': float(series.min()),
            'max': float(series.max()),
            'count': int(series.count())
        }
    return stats_dict

def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Return (lower_bound, upper_bound) for the IQR outlier rule."""
    clean = series.dropna()
    if clean.empty:
        return (float('nan'), float('nan'))
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    return (float(q1 - multiplier * iqr), float(q3 + multiplier * iqr))

def iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Boolean mask (aligned to series.index) flagging IQR outliers."""
    lower, upper = iqr_bounds(series, multiplier)
    mask = pd.Series(False, index=series.index)
    if pd.isna(lower):
        return mask
    valid = series.notna()
    mask.loc[valid] = (series[valid] < lower) | (series[valid] > upper)
    return mask

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Boolean mask flagging Z-score outliers.

    Args:
        series (pd.Series): Input numeric series.
        threshold (float): Z-score threshold for outlier detection. Defaults to 3.0.

    Returns:
        pd.Series: Boolean mask aligned to series.index flagging outliers.
    """
    non_null = series.dropna()
    mask = pd.Series(False, index=series.index)
    if non_null.empty or non_null.std(ddof=0) == 0:
        return mask
    z_scores = np.abs(stats.zscore(non_null))
    mask.loc[non_null.index] = z_scores > threshold
    return mask

def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Boolean mask flagging Modified Z-score (MAD) outliers.

    Args:
        series (pd.Series): Input numeric series.
        threshold (float): Modified Z-score threshold for outlier detection. Defaults to 3.5.

    Returns:
        pd.Series: Boolean mask aligned to series.index flagging outliers.
    """
    median = series.median()
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    mad = np.median(np.abs(clean - median))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return (np.abs(modified_z) > threshold).fillna(False)

def classify_correlation_strength(abs_corr: float) -> str:
    """Classify correlation strength given absolute correlation value."""
    if abs_corr >= 0.9:
        return "Very Strong"
    elif abs_corr >= 0.7:
        return "Strong"
    elif abs_corr >= 0.5:
        return "Moderate"
    elif abs_corr >= 0.3:
        return "Weak"
    return "Very Weak"

def classify_distribution_shape(skew: float, kurt: float, is_normal: bool) -> str:
    """Classify distribution shape based on skewness, kurtosis, and normality."""
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

def bucket_top_n(value_counts: pd.Series, n: int = 15) -> tuple[pd.Series, int, float]:
    """Collapse a value_counts() Series into its top-n entries + 'Other'."""
    total = value_counts.sum()
    if len(value_counts) <= n or total == 0:
        return value_counts, 0, 1.0
    top = value_counts.iloc[:n]
    other_count = value_counts.iloc[n:].sum()
    other_n = len(value_counts) - n
    bucketed = pd.concat([top, pd.Series({f"Other ({other_n} categories)": other_count})])
    coverage = top.sum() / total
    return bucketed, other_n, coverage
