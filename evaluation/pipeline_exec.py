"""
Shared pipeline execution for evaluation.

Runs each dataset query once through the full
Dynamic-RAG graph and returns the FinalResponse
plus timing. Plane 2 (generation) and Plane 3
(system) evaluations are both computed from this
single pass, so the expensive pipeline is not run
multiple times per query.
"""

import json
import time
from typing import List, Dict, Any

from src.graph.graph_builder import run_query
from src.observability.logger import app_logger


def execute_dataset(
    dataset_path: str
) -> List[Dict[str, Any]]:
    """
    Execute every query in the dataset through the
    graph. Returns one record per example:

        {
          "example": <dataset item>,
          "response": <FinalResponse or None>,
          "latency_ms": float,
          "error": <str or None>
        }
    """

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:
        dataset = json.load(f)

    results = []

    for i, example in enumerate(dataset, start=1):

        query = example["query"]

        start = time.perf_counter()

        response = None
        error = None

        try:
            response = run_query(query)

        except Exception as exc:
            error = repr(exc)
            app_logger.error(
                f"Eval pipeline failed on query "
                f"{i} ({query[:40]!r}): {error}"
            )

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        results.append({
            "example": example,
            "response": response,
            "latency_ms": latency_ms,
            "error": error
        })

        app_logger.info(
            f"Eval {i}/{len(dataset)}: "
            f"route="
            f"{getattr(response, 'route', None)} "
            f"status="
            f"{getattr(response, 'status', 'error')}"
        )

    return results
