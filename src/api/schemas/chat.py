from typing import Optional, List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    Chat query request
    """

    query: str

    session_id: Optional[str] = None

    document_scope: Optional[
        List[str]
    ] = None


class ChatResponse(BaseModel):
    """
    Final chat response
    """

    answer: str

    query_id: Optional[str] = None

    session_id: Optional[str] = None

    route: str

    confidence: float

    sources: list = []

    faithfulness_score: Optional[
        float
    ] = None

    latency_ms: Optional[
        float
    ] = None

    status: str = "success"