import time

from fastapi import APIRouter, HTTPException

from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse
)

from src.graph.graph_builder import run_query
from src.memory.store import conversation_store
from src.database.mongo_client import mongo_client
from src.observability.logger import app_logger


router = APIRouter()


def _serialize_sources(sources) -> list:
    """
    Convert EvidenceItem objects into plain dicts
    for the API response.
    """

    serialized = []

    for s in sources:
        serialized.append({
            "source_type": s.source_type,
            "source_id": s.source_id,
            "chunk_id": s.chunk_id,
            "page": s.page,
            "score": s.score,
            "title": (
                s.metadata.get("title")
                if s.metadata else None
            )
        })

    return serialized


@router.post(
    "/chat/query",
    response_model=ChatResponse
)
def chat_query(request: ChatRequest):
    """
    Run a query through the Dynamic-RAG graph.

    Defined as a sync endpoint so FastAPI runs the
    (blocking) graph execution in a threadpool and
    does not stall the event loop.
    """

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="query must not be empty"
        )

    start = time.perf_counter()

    try:
        final = run_query(
            query=request.query,
            session_id=request.session_id
        )

    except Exception as exc:
        app_logger.error(
            f"Query execution failed: {exc!r}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {exc!r}"
        )

    latency_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    return ChatResponse(
        answer=final.answer,
        query_id=final.query_id,
        session_id=final.session_id,
        route=final.route,
        confidence=final.confidence,
        sources=_serialize_sources(final.sources),
        faithfulness_score=final.faithfulness_score,
        latency_ms=latency_ms,
        status=final.status
    )


@router.get("/chat/{session_id}")
def get_session(session_id: str):
    """
    Return the conversation history for a session.
    """

    history = conversation_store.get_recent_context(
        session_id=session_id,
        limit=50
    )

    messages = []

    for item in history:
        messages.append({
            "query": item.get("query"),
            "answer": item.get("answer"),
            "route": item.get("route"),
            "confidence": item.get("confidence"),
            "timestamp": (
                item["timestamp"].isoformat()
                if item.get("timestamp")
                else None
            )
        })

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages
    }


@router.get("/query/{query_id}/sources")
def get_query_sources(query_id: str):
    """
    Return the evidence sources associated with a
    previously answered query (by request id).
    """

    try:
        db = mongo_client.get_database()
        trace = db["traces"].find_one(
            {"request_id": query_id},
            {"_id": 0}
        )

    except Exception as exc:
        app_logger.error(
            f"Source lookup failed: {exc!r}"
        )
        raise HTTPException(
            status_code=500,
            detail="Source lookup failed"
        )

    if not trace:
        raise HTTPException(
            status_code=404,
            detail=f"No query found with id {query_id}"
        )

    return {
        "query_id": query_id,
        "route": trace.get("route"),
        "num_sources": trace.get("num_sources", 0),
        "sources": trace.get("sources", [])
    }
