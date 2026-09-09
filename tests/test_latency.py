from __future__ import annotations

from poc.latency import summarize_latency_ms


def test_latency_summary_reports_registered_percentiles() -> None:
    summary = summarize_latency_ms([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "samples": 5,
        "mean_ms": 3.0,
        "p50_ms": 3.0,
        "p95_ms": 4.8,
        "p99_ms": 4.96,
        "maximum_ms": 5.0,
    }
