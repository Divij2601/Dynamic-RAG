"""
LangGraph node functions for Dynamic-RAG.

Each node takes the QueryState and returns a dict
of field updates that LangGraph merges back into
the state. Nodes have a single bounded responsibility
and degrade gracefully when an optional dependency
(web, memory, Mongo) is unavailable.
"""

import time
from typing import Dict, Any

from src.graph.state import (
    QueryState,
    RetrievalMetrics,
    GenerationMetrics
)

from src.planner.planner import query_planner

from src.retrieval.hybrid import hybrid_retriever
from src.retrieval.reranker import reranker
from src.retrieval.evidence import evidence_builder

from src.web.search import web_search_agent
from src.web.evidence import web_evidence_builder

from src.memory.retriever import memory_retriever
from src.memory.store import conversation_store

from src.generation.prompt_builder import prompt_builder
from src.generation.generator import response_generator
from src.generation.verifier import faithfulness_verifier
from src.generation.response_builder import response_builder

from src.config import settings
from src.observability.logger import app_logger


ABSTAIN_MESSAGE = (
    "I could not find sufficient evidence "
    "to answer this confidently."
)

EVIDENCE_ROUTES = {
    "internal_rag",
    "web_research",
    "hybrid"
}


# ----------------------------------------------------
# Context + Planning
# ----------------------------------------------------

def context_loader_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Entry node. Kept light: request/session
    identifiers are already on the state. Hook
    point for preloading scope/metadata later.
    """

    app_logger.info(
        f"[{state.request_id}] context loaded "
        f"for session {state.session_id}"
    )

    return {}


def planner_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Classify the query and select a route.
    """

    planner_output = query_planner.plan(
        state.query_text
    )

    return {
        "planner_output": planner_output,
        "selected_route": planner_output.route
    }


# ----------------------------------------------------
# Evidence-gathering routes
# ----------------------------------------------------

def _retrieve_internal(query: str):
    """
    Hybrid retrieve -> rerank -> evidence.
    Retrieves a wider candidate pool
    (RERANK_TOP_K) before reranking down to
    FINAL_TOP_K so the reranker has real choices.
    """

    retrieval = hybrid_retriever.retrieve(
        query,
        top_k=settings.RERANK_TOP_K
    )

    reranked = reranker.rerank(
        query=query,
        retrieved_chunks=retrieval["results"]
    )

    return evidence_builder.build(
        reranked["results"]
    )


def internal_retrieval_node(
    state: QueryState
) -> Dict[str, Any]:

    start = time.perf_counter()

    evidence = _retrieve_internal(
        state.query_text
    )

    latency = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return {
        "internal_evidence": evidence,
        "retrieval_metrics": RetrievalMetrics(
            retrieval_latency_ms=latency
        )
    }


def web_research_node(
    state: QueryState
) -> Dict[str, Any]:

    start = time.perf_counter()

    try:
        results = web_search_agent.search(
            state.query_text
        )
        evidence = web_evidence_builder.build(
            results
        )

    except Exception as exc:
        app_logger.error(
            f"Web research failed: {exc!r}"
        )
        evidence = []

    latency = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return {
        "web_evidence": evidence,
        "retrieval_metrics": RetrievalMetrics(
            retrieval_latency_ms=latency
        )
    }


def _load_memory(state: QueryState) -> str:

    try:
        return memory_retriever.retrieve_context(
            session_id=state.session_id,
            query=state.query_text
        )
    except Exception as exc:
        app_logger.error(
            f"Memory retrieval failed: {exc!r}"
        )
        return ""


def memory_node(
    state: QueryState
) -> Dict[str, Any]:

    return {
        "memory_context": _load_memory(state)
    }


def hybrid_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Combine internal documents, web evidence and
    conversation memory for complex queries.
    """

    start = time.perf_counter()

    internal_evidence = _retrieve_internal(
        state.query_text
    )

    try:
        web_results = web_search_agent.search(
            state.query_text
        )
        web_evidence = web_evidence_builder.build(
            web_results
        )
    except Exception as exc:
        app_logger.error(
            f"Hybrid web portion failed: {exc!r}"
        )
        web_evidence = []

    latency = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return {
        "internal_evidence": internal_evidence,
        "web_evidence": web_evidence,
        "memory_context": _load_memory(state),
        "retrieval_metrics": RetrievalMetrics(
            retrieval_latency_ms=latency
        )
    }


def direct_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Direct reasoning: no retrieval.
    """

    return {}


# ----------------------------------------------------
# Generation + Verification
# ----------------------------------------------------

def _build_direct_prompt(
    query: str,
    memory_context: str
) -> str:
    """
    Prompt for non-grounded routes (direct
    reasoning and memory continuity). These tasks
    (rewrite, summarize, explain, follow-ups) do
    not depend on document/web evidence, so the
    strict abstain-if-no-evidence rule does not
    apply.
    """

    parts = [
        "You are Dynamic-RAG, a helpful assistant."
    ]

    if memory_context:
        parts.append(
            "Conversation memory (use only to "
            "resolve references and maintain "
            "continuity):\n" + memory_context
        )

    parts.append("User question:\n" + query)
    parts.append(
        "Provide a clear, direct, helpful answer."
    )

    return "\n\n".join(parts)


def generate_node(
    state: QueryState
) -> Dict[str, Any]:

    route = state.selected_route or ""

    if route in ("direct_generation", "memory"):

        # Non-grounded: direct reasoning / continuity.
        prompt = _build_direct_prompt(
            query=state.query_text,
            memory_context=state.memory_context or ""
        )

    else:

        # Grounded routes (internal / web / hybrid):
        # answer strictly from evidence, abstain if
        # evidence is insufficient.
        prompt = prompt_builder.build_prompt(
            query=state.query_text,

            internal_evidence=(
                state.internal_evidence or None
            ),

            web_evidence=(
                state.web_evidence or None
            ),

            memory_context=(
                state.memory_context or ""
            )
        )

        # On a retry, push the model to be stricter.
        if state.retry_count > 0:
            prompt += (
                "\n\nIMPORTANT: A previous answer was "
                "flagged as not fully supported by the "
                "evidence. Only state claims directly "
                "supported by the evidence above, and "
                "abstain if the support is insufficient."
            )

    start = time.perf_counter()

    answer = response_generator.generate(prompt)

    latency = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return {
        "candidate_answer": answer,
        "generation_metrics": GenerationMetrics(
            generation_latency_ms=latency
        )
    }


def verify_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Faithfulness verification for evidence-based
    routes. Routes without document/web evidence
    (direct, memory) skip grounding checks.
    """

    route = state.selected_route or ""

    evidence = (
        list(state.internal_evidence or [])
        + list(state.web_evidence or [])
    )

    if route not in EVIDENCE_ROUTES or not evidence:

        return {
            "verification_result": {
                "faithful": True,
                "grounded": True,
                "faithfulness_score": None,
                "unsupported_claims": [],
                "reasoning":
                "Route does not rely on document "
                "or web evidence; faithfulness "
                "check skipped."
            }
        }

    verdict = faithfulness_verifier.verify(
        query=state.query_text,
        answer=state.candidate_answer or "",
        evidence_items=evidence
    )

    return {"verification_result": verdict}


def retry_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Bounded retry counter. Routing back to
    generation happens via a graph edge.
    """

    new_count = state.retry_count + 1

    app_logger.info(
        f"[{state.request_id}] retry "
        f"{new_count}/{settings.MAX_RETRIES}"
    )

    return {"retry_count": new_count}


# ----------------------------------------------------
# Formatting + Persistence
# ----------------------------------------------------

def format_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Assemble the FinalResponse, deciding whether
    to return the answer or abstain.
    """

    route = state.selected_route or "unknown"

    evidence = (
        list(state.internal_evidence or [])
        + list(state.web_evidence or [])
    )

    answer = state.candidate_answer or ""
    verdict = state.verification_result or {}

    # Abstain if the model itself abstained, or if
    # an evidence-based answer remains ungrounded
    # after the retry budget is exhausted.
    abstained = (
        "sufficient evidence" in answer.lower()
        and "could not" in answer.lower()
    )

    if (
        route in EVIDENCE_ROUTES
        and verdict.get("grounded") is False
    ):
        abstained = True

    if abstained:
        final_answer = ABSTAIN_MESSAGE
        status = "abstained"
    else:
        final_answer = answer
        status = "success"

    response = response_builder.build(
        answer=final_answer,
        route=route,
        evidence_items=evidence,
        verification=verdict,
        status=status,
        query_id=state.request_id,
        session_id=state.session_id
    )

    return {"final_response": response}


def persist_node(
    state: QueryState
) -> Dict[str, Any]:
    """
    Persist the interaction to conversation memory
    and write a request trace. Both are best-effort:
    a storage failure must not fail the request.
    """

    response = state.final_response

    if response is None:
        return {}

    # Save to conversation memory (for continuity).
    try:
        conversation_store.save_interaction(
            session_id=state.session_id,
            query=state.query_text,
            answer=response.answer,
            route=response.route,
            confidence=response.confidence
        )
    except Exception as exc:
        app_logger.error(
            f"save_interaction failed: {exc!r}"
        )

    # Write a structured trace.
    try:
        from src.database.mongo_client import (
            mongo_client
        )

        sources = [
            {
                "source_type": s.source_type,
                "source_id": s.source_id,
                "chunk_id": s.chunk_id,
                "page": s.page,
                "score": s.score,
                "title": (
                    s.metadata.get("title")
                    if s.metadata else None
                )
            }
            for s in response.sources
        ]

        trace = {
            "request_id": state.request_id,
            "session_id": state.session_id,
            "query": state.query_text,
            "route": response.route,
            "retrieval_latency_ms":
            state.retrieval_metrics
            .retrieval_latency_ms,
            "generation_latency_ms":
            state.generation_metrics
            .generation_latency_ms,
            "faithfulness_score":
            response.faithfulness_score,
            "grounded": response.grounded,
            "confidence": response.confidence,
            "retry_count": state.retry_count,
            "status": response.status,
            "num_sources": len(response.sources),
            "sources": sources
        }

        mongo_client.get_database()[
            "traces"
        ].insert_one(trace)

    except Exception as exc:
        app_logger.error(
            f"Trace persistence failed: {exc!r}"
        )

    return {}
