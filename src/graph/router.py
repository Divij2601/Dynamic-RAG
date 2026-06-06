"""
Conditional edge functions for the Dynamic-RAG
LangGraph. These functions are pure: they inspect
the state and return the name of the next node.
They never mutate state.
"""

from src.graph.state import QueryState
from src.config import settings


# Map planner route -> evidence-gathering node name.
_ROUTE_TO_NODE = {
    "internal_rag": "internal_retrieval",
    "web_research": "web_research",
    "hybrid": "hybrid",
    "memory": "memory",
    "direct_generation": "direct"
}

EVIDENCE_ROUTES = {
    "internal_rag",
    "web_research",
    "hybrid"
}


def route_after_planner(
    state: QueryState
) -> str:
    """
    Dispatch to the evidence node for the planned
    route. Falls back to internal retrieval.
    """

    route = (
        state.planner_output.route
        if state.planner_output
        else "internal_rag"
    )

    return _ROUTE_TO_NODE.get(
        route,
        "internal_retrieval"
    )


def route_after_verify(
    state: QueryState
) -> str:
    """
    Decide whether to retry generation or format
    the response. Retries are only for grounded
    routes that failed verification and are bounded
    by MAX_RETRIES.
    """

    route = state.selected_route or ""

    # Non-evidence routes never retry on grounding.
    if route not in EVIDENCE_ROUTES:
        return "format"

    verdict = state.verification_result or {}

    grounded = verdict.get("grounded", True)

    if (
        grounded is False
        and state.retry_count < settings.MAX_RETRIES
    ):
        return "retry"

    return "format"
