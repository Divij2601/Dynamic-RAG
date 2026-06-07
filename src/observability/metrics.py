"""
Operational metrics aggregated from persisted
request traces (Mongo `traces` collection).

Backs the GET /system/metrics endpoint. All
queries are best-effort: if Mongo is unavailable
the endpoint still returns a well-formed payload.
"""

from typing import Dict, Any

from src.database.mongo_client import mongo_client
from src.observability.logger import app_logger


def _percentile(values, pct: float) -> float:
    """
    Simple percentile without numpy.
    """

    if not values:
        return 0.0

    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)

    if lower == upper:
        return round(ordered[lower], 2)

    weight = k - lower

    return round(
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight,
        2
    )


def get_system_metrics() -> Dict[str, Any]:
    """
    Aggregate operational metrics from traces.
    """

    try:
        db = mongo_client.get_database()
        traces = list(
            db["traces"].find(
                {},
                {"_id": 0}
            )
        )

    except Exception as exc:
        app_logger.error(
            f"Metrics aggregation failed: {exc!r}"
        )
        return {
            "status": "unavailable",
            "total_requests": 0,
            "error": repr(exc)
        }

    total = len(traces)

    if total == 0:
        return {
            "status": "active",
            "total_requests": 0,
            "message":
            "No requests traced yet."
        }

    def _latency(t):
        return (
            (t.get("retrieval_latency_ms") or 0)
            + (t.get("generation_latency_ms") or 0)
        )

    latencies = [_latency(t) for t in traces]

    confidences = [
        t.get("confidence", 0.0)
        for t in traces
    ]

    abstentions = sum(
        1 for t in traces
        if t.get("status") == "abstained"
    )

    retries = sum(
        t.get("retry_count", 0) for t in traces
    )

    # Route distribution
    route_counts: Dict[str, int] = {}
    for t in traces:
        route = t.get("route", "unknown")
        route_counts[route] = (
            route_counts.get(route, 0) + 1
        )

    total_cost = round(
        sum(
            t.get("total_cost_usd", 0.0)
            for t in traces
        ),
        6
    )

    return {
        "status": "active",
        "total_requests": total,
        "route_distribution": route_counts,
        "mean_latency_ms": round(
            sum(latencies) / total, 2
        ),
        "p95_latency_ms": _percentile(
            latencies, 95
        ),
        "p99_latency_ms": _percentile(
            latencies, 99
        ),
        "mean_confidence": round(
            sum(confidences) / total, 3
        ),
        "abstention_rate": round(
            abstentions / total, 3
        ),
        "total_retries": retries,
        "total_cost_usd": total_cost,
        "mean_cost_per_query_usd": round(
            total_cost / total, 6
        )
    }
