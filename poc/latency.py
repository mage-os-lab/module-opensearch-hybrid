from __future__ import annotations

from statistics import fmean

import numpy as np


def summarize_latency_ms(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(sample < 0 for sample in samples):
        raise ValueError("latency samples must be non-empty and non-negative")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": len(samples),
        "mean_ms": fmean(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": max(samples),
    }
