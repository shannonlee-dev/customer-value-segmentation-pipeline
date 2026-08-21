"""Pure descriptive-statistics helpers shared by pipeline and reporting layers."""

import numpy as np


def summarize_numeric(values) -> dict[str, float | int]:
    """Return full-input descriptive statistics with sample standard deviation."""
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        raise ValueError("Numeric summary requires at least one value")
    q1, median, q3 = np.quantile(array, [0.25, 0.50, 0.75])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(median),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
