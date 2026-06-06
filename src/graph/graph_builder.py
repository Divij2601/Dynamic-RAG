"""
Assemble and compile the Dynamic-RAG LangGraph.

Flow:

    START
      -> context_loader
      -> planner
      -> (route) internal_retrieval | web_research
                 | hybrid | memory | direct
      -> generate
      -> verify
      -> (retry?) retry -> generate   [bounded]
      -> format
      -> persist
      -> END
"""

from typing import Optional

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from src.graph.state import (
    QueryState,
    FinalResponse
)

from src.graph import nodes
from src.graph.router import (
    route_after_planner,
    route_after_verify
)

from src.observability.tracing import (
    generate_request_id
)
from src.observability.logger import app_logger


def build_graph():
    """
    Build and compile the state graph.
    """

    graph = StateGraph(QueryState)

    # Nodes
    graph.add_node(
        "context_loader",
        nodes.context_loader_node
    )
    graph.add_node("planner", nodes.planner_node)
    graph.add_node(
        "internal_retrieval",
        nodes.internal_retrieval_node
    )
    graph.add_node(
        "web_research",
        nodes.web_research_node
    )
    graph.add_node("memory", nodes.memory_node)
    graph.add_node("hybrid", nodes.hybrid_node)
    graph.add_node("direct", nodes.direct_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("verify", nodes.verify_node)
    graph.add_node("retry", nodes.retry_node)
    graph.add_node("format", nodes.format_node)
    graph.add_node("persist", nodes.persist_node)

    # Edges
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "internal_retrieval": "internal_retrieval",
            "web_research": "web_research",
            "memory": "memory",
            "hybrid": "hybrid",
            "direct": "direct"
        }
    )

    for evidence_node in (
        "internal_retrieval",
        "web_research",
        "memory",
        "hybrid",
        "direct"
    ):
        graph.add_edge(evidence_node, "generate")

    graph.add_edge("generate", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "retry": "retry",
            "format": "format"
        }
    )

    graph.add_edge("retry", "generate")
    graph.add_edge("format", "persist")
    graph.add_edge("persist", END)

    compiled = graph.compile()

    app_logger.success(
        "Dynamic-RAG graph compiled"
    )

    return compiled


# Compile once at import time.
dynamic_rag_graph = build_graph()


def _extract_final_response(result) -> FinalResponse:
    """
    LangGraph may return the final state as a dict
    of channel values or as the state model. Handle
    both.
    """

    if isinstance(result, dict):
        final = result.get("final_response")
    else:
        final = getattr(
            result,
            "final_response",
            None
        )

    if final is None:
        return FinalResponse(
            answer="",
            status="error"
        )

    return final


def run_query(
    query: str,
    session_id: Optional[str] = None
) -> FinalResponse:
    """
    Run one query through the full Dynamic-RAG
    graph and return the FinalResponse.
    """

    request_id = generate_request_id()

    session_id = session_id or f"session_{request_id}"

    initial_state = {
        "request_id": request_id,
        "session_id": session_id,
        "query_text": query
    }

    result = dynamic_rag_graph.invoke(
        initial_state
    )

    return _extract_final_response(result)
