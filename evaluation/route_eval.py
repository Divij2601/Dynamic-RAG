"""
Route-accuracy evaluation (Gate C).

Measures how well the planner assigns queries to
routes. The expected route is taken from an explicit
`expected_route` field if present, otherwise derived
from the `answerable` flag:

    answerable == True  -> internal_rag
                           (answerable from the corpus)
    answerable == False -> web_research
                           (out-of-domain, web-answerable)

This derivation is non-destructive: it does not modify
the dataset file.
"""

import json
import time
from collections import defaultdict

INTER_QUERY_DELAY_SECONDS = 2.0

from src.planner.planner import query_planner
from src.observability.logger import app_logger


def _expected_route(example) -> str:
    """
    Return the expected route, or None if this query
    has no correct route (truly unanswerable — the
    system should abstain regardless of route).
    """

    if example.get("expected_route"):
        return example["expected_route"]

    # Truly unanswerable queries (fictional entities,
    # private documents, classified info) have no
    # correct route — exclude them from route accuracy.
    category = example.get("metadata", {}).get(
        "category", ""
    )
    if category == "unanswerable_true":
        return None

    return (
        "internal_rag"
        if example.get("answerable", True)
        else "web_research"
    )


class RouteEvaluator:
    """
    Gate C: planner route accuracy + confusion.
    """

    def evaluate(
        self,
        dataset_path: str
    ):

        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as f:
            dataset = json.load(f)

        total = 0
        correct = 0

        # confusion[expected][predicted] = count
        confusion = defaultdict(
            lambda: defaultdict(int)
        )

        per_class_total = defaultdict(int)
        per_class_correct = defaultdict(int)

        for example in dataset:

            query = example["query"]
            expected = _expected_route(example)

            # Skip truly unanswerable queries from
            # route accuracy — they have no correct route.
            if expected is None:
                continue

            predicted = query_planner.plan(
                query
            ).route

            total += 1
            per_class_total[expected] += 1

            confusion[expected][predicted] += 1

            # "hybrid" is acceptable for internal_rag
            # queries: the system fetches corpus + web,
            # which is a strict superset of internal_rag.
            # Penalising it as wrong would punish the
            # planner for being overly thorough.
            is_correct = (predicted == expected) or (
                expected == "internal_rag"
                and predicted == "hybrid"
            )

            if is_correct:
                correct += 1
                per_class_correct[expected] += 1

            time.sleep(INTER_QUERY_DELAY_SECONDS)

        per_class_accuracy = {
            route: round(
                per_class_correct[route]
                / per_class_total[route],
                4
            )
            for route in per_class_total
        }

        metrics = {
            "Route Accuracy": round(
                correct / total, 4
            ) if total else 0.0,

            "Per-Class Accuracy":
            per_class_accuracy,

            "Confusion Matrix": {
                expected: dict(preds)
                for expected, preds
                in confusion.items()
            },

            "Total Queries": total
        }

        app_logger.success(
            "Route (Gate C) evaluation complete"
        )

        return metrics


route_evaluator = RouteEvaluator()
