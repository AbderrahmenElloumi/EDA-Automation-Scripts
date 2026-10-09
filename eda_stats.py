import pandas as pd
import numpy as np

def calculate_summary_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for numeric columns in a DataFrame."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col]
        stats[col] = {
            'mean': float(series.mean()),
            'std': float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            'min': float(series.min()),
            'max': float(series.max()),
            'count': int(series.count())
        }
    return stats
