from __future__ import annotations

from typing import Any, Iterable


_FALLBACK_METRICS = {
    "cuda_time_total": "device_time_total",
    "self_cuda_time_total": "self_device_time_total",
}


def _metric(event: Any, name: str) -> float:
    if hasattr(event, name):
        return float(getattr(event, name))
    fallback_name = _FALLBACK_METRICS.get(name)
    if fallback_name and hasattr(event, fallback_name):
        return float(getattr(event, fallback_name))
    return 0.0


def extract_top_ops(events: Iterable[Any], sort_key: str, row_limit: int = 20) -> list[dict[str, float | int | str]]:
    sorted_events = sorted(events, key=lambda event: _metric(event, sort_key), reverse=True)
    top_events = []
    total = sum(_metric(event, sort_key) for event in sorted_events)
    for event in sorted_events[:row_limit]:
        metric_value = _metric(event, sort_key)
        top_events.append(
            {
                "name": event.key,
                "count": int(getattr(event, "count", 0)),
                sort_key: round(metric_value, 3),
                "cpu_time_total_us": round(float(getattr(event, "cpu_time_total", 0.0)), 3),
                "self_cpu_time_total_us": round(float(getattr(event, "self_cpu_time_total", 0.0)), 3),
                "cuda_time_total_us": round(_metric(event, "cuda_time_total"), 3),
                "self_cuda_time_total_us": round(_metric(event, "self_cuda_time_total"), 3),
                "share_pct": round((metric_value / total) * 100.0, 3) if total else 0.0,
            }
        )
    return top_events
